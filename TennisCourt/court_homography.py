# --- START OF FILE homography.py ---

from court_reference import CourtReference
import numpy as np
import cv2
from scipy.spatial import distance

# Initialize Court Reference globally (or pass it as an argument)
# Global initialization can be problematic if CourtReference state changes or is resource-intensive
# Consider passing an instance if used within a larger application.
print("Initializing CourtReference for homography...")
court_ref = CourtReference()
refer_kps_all = court_ref.get_keypoints_array().reshape((-1, 1, 2)) # Shape: (14, 1, 2)
print("CourtReference initialized.")

# Precompute a mapping from configuration ID to keypoint indices for efficiency
court_conf_indices = court_ref.court_conf # Use the dictionary directly

def get_trans_matrix(predicted_points, confidence_scores=None, min_inliers=4, reprojection_threshold=10.0):
    """
    Finds the best homography matrix mapping reference court points to predicted points.

    Args:
        predicted_points (list): A list of 14 tuples (x, y), where each tuple corresponds
                                 to a predicted keypoint. Use None for undetected points.
        confidence_scores (list, optional): A list of confidence scores for each predicted point.
                                            Used to select more reliable points for initial estimation.
        min_inliers (int): Minimum number of points required to compute homography.
        reprojection_threshold (float): Maximum allowed reprojection error (distance) for a point
                                        to be considered an inlier during RANSAC/robust estimation.

    Returns:
        np.ndarray or None: The 3x3 homography matrix if found, otherwise None.
    """
    matrix_trans = None
    best_num_inliers = -1 # Track the number of inliers for the best matrix
    min_median_dist = np.inf # Use median distance as a secondary sorting criterion

    # Prepare predicted points: Filter out None values and keep track of original indices
    valid_pred_pts = []
    valid_ref_pts = []
    original_indices = []
    for i, pt in enumerate(predicted_points):
        if pt is not None and pt[0] is not None and pt[1] is not None:
            valid_pred_pts.append(pt)
            # Get corresponding reference point using the index 'i'
            valid_ref_pts.append(np.squeeze(refer_kps_all[i])) # Squeeze from (1, 2) to (2,)
            original_indices.append(i)

    if len(valid_pred_pts) < min_inliers:
        # print(f"Not enough valid points ({len(valid_pred_pts)}) to compute homography.")
        return None # Not enough points

    # Convert points to NumPy arrays of shape (N, 2) as required by findHomography
    np_pred_pts = np.array(valid_pred_pts, dtype=np.float32)
    np_ref_pts = np.array(valid_ref_pts, dtype=np.float32)

    # --- Method 1: Using OpenCV's findHomography with RANSAC ---
    # RANSAC is generally preferred as it's robust to outliers.
    try:
        matrix_trans, mask_ransac = cv2.findHomography(
            np_ref_pts,                 # Source points (reference court)
            np_pred_pts,                # Destination points (predicted points)
            method=cv2.RANSAC,          # Use RANSAC for robustness
            ransacReprojThreshold=reprojection_threshold, # Max reprojection error
            maxIters=2000,              # Max RANSAC iterations
            confidence=0.99             # Confidence level
        )

        if matrix_trans is not None:
            num_inliers = np.sum(mask_ransac)
            # print(f"RANSAC Homography found with {num_inliers} inliers.")
            if num_inliers >= min_inliers:
                 # Optionally, refine the homography using only the inliers
                 inlier_ref_pts = np_ref_pts[mask_ransac.ravel() == 1]
                 inlier_pred_pts = np_pred_pts[mask_ransac.ravel() == 1]
                 matrix_trans_refined, _ = cv2.findHomography(inlier_ref_pts, inlier_pred_pts, method=0) # Use LM refinement
                 if matrix_trans_refined is not None:
                     matrix_trans = matrix_trans_refined
                 return matrix_trans
            else:
                # print(f"RANSAC found matrix but only {num_inliers} inliers (min required: {min_inliers}). Discarding.")
                matrix_trans = None # Discard if too few inliers
        # else:
        #    print("RANSAC failed to find a homography matrix.")

    except cv2.error as e:
        print(f"OpenCV error during findHomography (RANSAC): {e}")
        return None


    # --- Method 2: Iterating through predefined configurations (Fallback or Alternative) ---
    # This method might be useful if RANSAC fails or if specific point sets are known to be reliable.
    # It requires evaluating the quality of the resulting transformation.
    # print("RANSAC failed or produced insufficient inliers. Trying predefined configurations...")
    if matrix_trans is None: # Only run if RANSAC failed
        for conf_ind in court_conf_indices: # Iterate through configuration IDs (1 to 12)
            conf_point_indices = court_conf_indices[conf_ind] # Get the list of 4 indices for this config

            # Check if all 4 points for this configuration are present in the *predicted* points
            conf_pred_pts = []
            conf_ref_pts = []
            valid_conf = True
            for idx in conf_point_indices:
                if idx in original_indices:
                    pred_pt_index = original_indices.index(idx) # Find where this kp index is in the valid_pred_pts list
                    conf_pred_pts.append(valid_pred_pts[pred_pt_index])
                    conf_ref_pts.append(np.squeeze(refer_kps_all[idx]))
                else:
                    valid_conf = False # This configuration requires a point that wasn't detected
                    break

            if not valid_conf or len(conf_pred_pts) != 4:
                continue # Skip this configuration

            # Convert points for this configuration to NumPy arrays
            np_conf_pred = np.array(conf_pred_pts, dtype=np.float32)
            np_conf_ref = np.array(conf_ref_pts, dtype=np.float32)

            try:
                # Calculate homography using only these 4 points (less robust, method=0 is Direct Linear Transform)
                matrix_conf, _ = cv2.findHomography(np_conf_ref, np_conf_pred, method=0)

                if matrix_conf is not None:
                    # Evaluate this matrix using *all* valid points (not just the 4 used to compute it)
                    # Transform all reference points using the current matrix
                    transformed_ref_kps = cv2.perspectiveTransform(np.array(valid_ref_pts, dtype=np.float32).reshape(-1, 1, 2), matrix_conf)
                    if transformed_ref_kps is None: continue # Transform failed

                    transformed_ref_kps = np.squeeze(transformed_ref_kps) # Shape (N, 2)

                    # Calculate distances between transformed reference points and actual predictions
                    dists = []
                    current_inliers = 0
                    for i in range(len(valid_pred_pts)):
                        dist = distance.euclidean(valid_pred_pts[i], transformed_ref_kps[i])
                        if dist < reprojection_threshold: # Check if within threshold
                            current_inliers += 1
                        dists.append(dist)

                    if current_inliers >= min_inliers: # Check if enough points agree with this matrix
                        dist_median = np.median(dists) # Use median distance for robustness

                        # Update best matrix if this one has more inliers,
                        # or same number of inliers but smaller median distance
                        if current_inliers > best_num_inliers or \
                           (current_inliers == best_num_inliers and dist_median < min_median_dist):
                            # print(f"Conf {conf_ind}: Found better matrix. Inliers: {current_inliers}, Median Dist: {dist_median:.2f}")
                            matrix_trans = matrix_conf
                            best_num_inliers = current_inliers
                            min_median_dist = dist_median

            except cv2.error as e:
                # print(f"OpenCV error during findHomography (Config {conf_ind}): {e}")
                continue # Ignore errors for specific configurations


    # if matrix_trans is not None:
    #     print(f"Selected best homography. Method: {'Config-based' if best_num_inliers > -1 else 'RANSAC'}. Inliers: {best_num_inliers if best_num_inliers > -1 else np.sum(mask_ransac)}. Median Dist: {min_median_dist if min_median_dist != np.inf else 'N/A'}")
    # else:
    #     print("Failed to find a suitable homography matrix.")

    return matrix_trans

# --- END OF FILE homography.py ---