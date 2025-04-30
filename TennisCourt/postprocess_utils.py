# --- START OF FILE postprocess.py ---

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance
from utils import line_intersection

def find_heatmap_peak(heatmap, sigma=1.5):
    """Finds the peak coordinates in a heatmap.

    Args:
        heatmap (np.array): Single channel heatmap.
        sigma (float): Standard deviation for Gaussian filter applied before peak finding.

    Returns:
        tuple: (x, y) coordinates of the peak, or (None, None) if no peak found.
    """
    # Apply Gaussian filter to smooth the heatmap and find a more stable peak
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    # Find the coordinates of the maximum value
    max_val = np.max(heatmap)
    if max_val == 0: # Avoid issues if heatmap is all zeros
        return None, None
    # Find indices of max value. Use unravel_index to get 2D coordinates.
    # Note: argmax might return the first occurence if there are multiple max values.
    y_pred, x_pred = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return x_pred, y_pred

def postprocess(heatmap, scale_factor=2, threshold=0.1):
    """
    Extracts keypoint coordinates from a heatmap by finding the peak.

    Args:
        heatmap (np.array): A single heatmap (H, W). Should be normalized (e.g., sigmoid output).
        scale_factor (int): Factor to scale the coordinates back to original image size
                           (relative to the heatmap size). E.g., if heatmap is 360x640
                           and original input was 720x1280, scale=2.
        threshold (float): Minimum peak value required to consider the point valid.

    Returns:
        tuple: (x_pred, y_pred) scaled coordinates, or (None, None) if peak below threshold.
    """
    # Ensure heatmap is float for filtering and peak value comparison
    heatmap = heatmap.astype(np.float32)

    max_val = np.max(heatmap)

    # Only proceed if the peak is above the threshold
    if max_val < threshold:
        return None, None

    # Find peak coordinates in the heatmap's resolution
    # Use a small sigma for smoothing, adjust if needed
    x_hm, y_hm = find_heatmap_peak(heatmap, sigma=1)

    if x_hm is None or y_hm is None:
        return None, None

    # Scale coordinates back to the target resolution (e.g., original image resolution)
    x_pred = x_hm * scale_factor
    y_pred = y_hm * scale_factor

    return x_pred, y_pred


# --- Refine KPS, Detect Lines, Merge Lines remain the same ---

def refine_kps(img, y_ct, x_ct, crop_size=40): # Note: OpenCV uses (y, x) for indexing
    """Refines keypoint location using line intersection in a local crop."""
    # Ensure input coordinates are integers for slicing
    x_ct, y_ct = int(x_ct), int(y_ct)

    refined_x_ct, refined_y_ct = x_ct, y_ct # Default to original if refinement fails

    img_height, img_width = img.shape[:2]

    # Calculate crop boundaries, ensuring they are within image limits
    x_min = max(x_ct - crop_size, 0)
    x_max = min(x_ct + crop_size, img_width) # Width corresponds to x-axis
    y_min = max(y_ct - crop_size, 0)
    y_max = min(y_ct + crop_size, img_height) # Height corresponds to y-axis

    # Check if crop dimensions are valid
    if x_min >= x_max or y_min >= y_max:
        print("Warning: Invalid crop dimensions in refine_kps.")
        return refined_y_ct, refined_x_ct # Return original coords

    img_crop = img[y_min:y_max, x_min:x_max] # Crop is [height_slice, width_slice]

    # Check if crop is empty
    if img_crop.size == 0:
        print("Warning: Empty crop in refine_kps.")
        return refined_y_ct, refined_x_ct # Return original coords

    lines = detect_lines(img_crop) # Detect lines within the crop

    if lines is not None and len(lines) > 1: # Need at least 2 lines for intersection
        lines = merge_lines(lines)
        if len(lines) == 2:
            # line_intersection expects (x1, y1, x2, y2) format
            # Coordinates are relative to the crop
            inters = line_intersection(lines[0], lines[1])
            if inters:
                # inters returns (x, y) relative to the crop
                new_x_crop, new_y_crop = map(int, inters)

                # Check if intersection is within crop bounds
                crop_h, crop_w = img_crop.shape[:2]
                if 0 < new_x_crop < crop_w and 0 < new_y_crop < crop_h:
                    # Translate back to original image coordinates
                    refined_x_ct = x_min + new_x_crop
                    refined_y_ct = y_min + new_y_crop
                    # print(f"Refined {x_ct},{y_ct} to {refined_x_ct},{refined_y_ct}") # Debugging

    # Return refined coordinates in (y, x) order consistent with input, or original if failed
    return refined_y_ct, refined_x_ct


def detect_lines(image):
    """Detects lines in an image crop using HoughLinesP."""
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image # Assume already grayscale if not 3 channels

    # Apply adaptive thresholding or Canny edge detection for potentially better line extraction
    # Option 1: Adaptive Thresholding
    # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Option 2: Canny Edge Detection
    # Use Canny for better edge definition before HoughLinesP
    edges = cv2.Canny(gray, threshold1=50, threshold2=150, apertureSize=3)

    # Use a simple threshold (as before) if Canny/Adaptive doesn't work well
    # _, binary = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)

    # Detect lines using HoughLinesP on the edge map
    # Adjust parameters: rho, theta precision, threshold, minLineLength, maxLineGap
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=20, minLineLength=10, maxLineGap=10)

    if lines is None:
        return [] # Return empty list if no lines found

    # Squeeze removes unnecessary dimensions if only one line is found
    lines = np.squeeze(lines, axis=1) # Squeeze the axis where lines are stacked

    # Handle case where squeeze might over-reduce for a single line detection
    if lines.ndim == 1 and lines.shape[0] == 4:
        lines = np.expand_dims(lines, axis=0) # Reshape back to (1, 4)

    return lines


def merge_lines(lines, angle_thresh=10, dist_thresh=15):
    """Merges lines that are close and have similar angles."""
    if len(lines) < 2:
        return lines

    # Calculate angle and representative point (midpoint) for each line
    line_props = []
    for x1, y1, x2, y2 in lines:
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        line_props.append({'coords': [x1, y1, x2, y2], 'angle': angle, 'mid': (mid_x, mid_y), 'merged': False})

    merged_lines_coords = []
    num_lines = len(line_props)

    for i in range(num_lines):
        if line_props[i]['merged']:
            continue

        current_group = [line_props[i]['coords']]
        line_props[i]['merged'] = True

        for j in range(i + 1, num_lines):
            if line_props[j]['merged']:
                continue

            # Compare angle difference (handle wrap-around 180 degrees)
            angle_diff = abs(line_props[i]['angle'] - line_props[j]['angle'])
            angle_diff = min(angle_diff, 180 - angle_diff)

            # Compare distance between midpoints
            dist = distance.euclidean(line_props[i]['mid'], line_props[j]['mid'])

            if angle_diff < angle_thresh and dist < dist_thresh:
                current_group.append(line_props[j]['coords'])
                line_props[j]['merged'] = True

        # Average the coordinates of the lines in the group
        if len(current_group) > 1:
            avg_line = np.mean(np.array(current_group), axis=0).astype(np.int32)
            merged_lines_coords.append(avg_line)
        else:
            # Keep the original line if it wasn't merged
            merged_lines_coords.append(np.array(current_group[0], dtype=np.int32))

    # Ensure the output format is consistent (list of numpy arrays)
    # Handle potential edge case where merging might result in fewer than expected lines
    return merged_lines_coords


# --- END OF FILE postprocess.py ---