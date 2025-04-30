from court_reference_model import CourtReferenceModel
import numpy as np
import cv2
from scipy.spatial import distance

court_ref = CourtReferenceModel()
refer_kps = np.array(court_ref.key_points, dtype=np.float32).reshape((-1, 1, 2))

court_conf_ind = {}
for i in range(len(court_ref.court_conf)):
    conf = court_ref.court_conf[i+1]
    inds = []
    for j in range(4):
        inds.append(court_ref.key_points.index(conf[j]))
    court_conf_ind[i+1] = inds

def get_trans_matrix(points):
    """
    Determine the best homography matrix from court points using RANSAC.
    """
    matrix_trans = None
    dist_max = np.Inf
    ransac_thresh = 5.0

    for conf_ind in range(1, 13):
        conf = court_ref.court_conf[conf_ind]
        inds = court_conf_ind[conf_ind]

        # Check if all required points for this configuration are available (not None)
        required_points = [points[inds[0]], points[inds[1]], points[inds[2]], points[inds[3]]]
        if None not in required_points:
            # Only create src_pts and dst_pts if all points are valid
            src_pts = np.float32([conf[0], conf[1], conf[2], conf[3]])
            dst_pts = np.float32(required_points) # Use the checked list

            matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

            if matrix is not None:
                trans_kps = cv2.perspectiveTransform(refer_kps, matrix).squeeze(1)
                dists = []
                num_valid_points = 0
                for i in range(len(points)):
                    # Check if the point exists *and* is not one of the source points used
                    if points[i] is not None and i not in inds:
                        # Ensure trans_kps[i] is valid before calculating distance
                        if trans_kps is not None and i < len(trans_kps) and np.all(np.isfinite(trans_kps[i])):
                            dists.append(distance.euclidean(points[i], trans_kps[i]))
                            num_valid_points += 1
                        else:
                            # Handle cases where transformation failed for this specific point
                            pass # Or handle differently if needed

                if num_valid_points > 0:
                    dist_median = np.median(dists)
                    if dist_median < dist_max:
                        matrix_trans = matrix
                        dist_max = dist_median
                # Keep the first valid matrix if no other points are available for comparison
                elif num_valid_points == 0 and matrix_trans is None:
                    matrix_trans = matrix
                    dist_max = 0 # Set dist_max to 0 as there's nothing to compare against yet

    return matrix_trans


