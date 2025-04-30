# --- START OF FILE utils.py ---

import numpy as np
from sympy import Line, Point, Segment # Import Segment as well
import sympy
import cv2 # Import OpenCV for GaussianBlur

# --- Heatmap Generation ---

def gaussian2D(shape, sigma=1):
    """Generates a 2D Gaussian kernel."""
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    # Set values below machine epsilon to zero
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    """Draws a Gaussian heatmap centered at 'center' with 'radius'."""
    diameter = 2 * radius + 1
    # Calculate sigma based on diameter (common practice: sigma = diameter / 6)
    sigma = diameter / 6
    gaussian = gaussian2D((diameter, diameter), sigma=sigma)

    # Ensure center coordinates are integers
    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    # Calculate boundaries for placing the Gaussian, handling edges
    left = min(x, radius)
    right = min(width - 1 - x, radius) # Adjust right/bottom boundary checks
    top = min(y, radius)
    bottom = min(height - 1 - y, radius)

    # Select the area of the heatmap and the corresponding part of the Gaussian
    heatmap_roi = heatmap[y - top : y + bottom + 1, x - left : x + right + 1]
    gaussian_roi = gaussian[radius - top : radius + bottom + 1, radius - left : radius + right + 1]

    # Ensure shapes match before applying maximum
    if heatmap_roi.shape == gaussian_roi.shape and heatmap_roi.size > 0:
        np.maximum(heatmap_roi, gaussian_roi * k, out=heatmap_roi) # Use np.maximum for element-wise max
    # else: print(f"Warning: Shape mismatch or empty ROI in draw_umich_gaussian. Center: {center}, Radius: {radius}, Heatmap shape: {heatmap.shape}")


    return heatmap # Return modified heatmap (though modification is inplace)


# --- Geometry ---

def line_intersection(line1, line2, tolerance=1e-6):
    """
    Finds the intersection point of two lines defined by four points.

    Args:
        line1 (tuple): (x1, y1, x2, y2) defining the first line.
        line2 (tuple): (x3, y3, x4, y4) defining the second line.
        tolerance (float): Tolerance for checking parallel lines.

    Returns:
        tuple: (x, y) coordinates of the intersection, or None if lines are parallel or coincident.
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Calculate denominator
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # Check if lines are parallel (denominator is close to zero)
    if abs(denom) < tolerance:
        return None # Lines are parallel or coincident

    # Calculate numerators
    t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    u_num = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3))

    # Calculate t and u parameters
    t = t_num / denom
    u = u_num / denom

    # Calculate intersection point
    intersect_x = x1 + t * (x2 - x1)
    intersect_y = y1 + t * (y2 - y1)

    # Optional check: if the intersection lies within the segments defined by the points
    # if 0 <= t <= 1 and 0 <= u <= 1:
    #     return intersect_x, intersect_y
    # else: # Intersection is outside the segments
    #     return None

    # For line intersection (not segment intersection), the point is always valid if not parallel
    return intersect_x, intersect_y

    # --- Alternative using SymPy (more robust for edge cases but potentially slower) ---
    # try:
    #     l1 = Line(Point(line1[0], line1[1]), Point(line1[2], line1[3]))
    #     l2 = Line(Point(line2[0], line2[1]), Point(line2[2], line2[3]))
    #
    #     intersection = l1.intersection(l2)
    #
    #     if intersection:
    #         # Check if intersection is a single Point object
    #         if isinstance(intersection[0], sympy.geometry.point.Point2D):
    #             point = intersection[0].coordinates
    #             # Convert SymPy Float to Python float
    #             return float(point[0]), float(point[1])
    #         # Handle cases where intersection might be a line segment (coincident lines)
    #         # else: return None # Or handle as needed
    #     return None # No intersection
    # except Exception as e:
    #     print(f"Error during SymPy line intersection: {e}")
    #     return None


# --- Image Checks ---

def is_point_in_image(x, y, input_width=1280, input_height=720):
    """Checks if a point (x, y) is within the image boundaries."""
    # Check if x or y are None first
    if x is None or y is None:
        return False
    # Check if coordinates are numerical and finite
    if not (np.isfinite(x) and np.isfinite(y)):
        return False
    # Check boundaries (inclusive of 0, exclusive of width/height max)
    # Adjusted to be inclusive of max boundary based on common indexing needs. Check usage.
    # Usually, valid coordinates are 0 <= x < width and 0 <= y < height
    # Sticking to original "<=" based on its usage in validation/test
    res = (x >= 0) and (x <= input_width) and (y >= 0) and (y <= input_height)
    return res

# --- Gaussian Radius Calculation (Original - kept for reference, but may not be actively used) ---
def gaussian_radius(det_size, min_overlap=0.7):
    """Calculates Gaussian radius based on object size and minimum overlap.
       Seems related to CenterNet's radius calculation.
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(max(b1 ** 2 - 4 * a1 * c1, 0)) # Use max(..., 0) to avoid sqrt of negative
    r1 = (b1 + sq1) / (2 * a1) # Denominator is 2*a1

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(max(b2 ** 2 - 4 * a2 * c2, 0))
    r2 = (b2 + sq2) / (2 * a2) # Denominator is 2*a2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(max(b3 ** 2 - 4 * a3 * c3, 0))
    # Check if a3 is zero or close to zero before dividing
    r3 = (b3 + sq3) / (2 * a3) if abs(a3) > 1e-6 else float('inf')

    # Return the minimum valid radius, ensuring it's non-negative
    return max(0, min(r1, r2, r3))


# --- END OF FILE utils.py ---