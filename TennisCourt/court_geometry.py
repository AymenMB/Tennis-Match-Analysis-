# --- START OF FILE court_reference.py ---

import cv2
import numpy as np
# import matplotlib.pyplot as plt # Only needed if saving images

class CourtReference:
    """
    Defines the standard court layout in a reference pixel space.
    Coordinates seem specific to a particular template image resolution/layout.
    """
    def __init__(self):
        # --- Key Point Definitions ---
        # Using tuples for coordinates (x, y)
        self.baseline_top = ((286, 561), (1379, 561))
        self.baseline_bottom = ((286, 2935), (1379, 2935))
        self.net_coords = ((286, 1748), (1379, 1748)) # Renamed from 'net' to avoid conflict
        self.left_court_line = ((286, 561), (286, 2935))
        self.right_court_line = ((1379, 561), (1379, 2935))
        self.left_inner_line = ((423, 561), (423, 2935)) # Service line side T
        self.right_inner_line = ((1242, 561), (1242, 2935)) # Service line side T
        self.middle_line = ((832, 1110), (832, 2386)) # Center service line
        self.top_inner_line = ((423, 1110), (1242, 1110)) # Service line (top half)
        self.bottom_inner_line = ((423, 2386), (1242, 2386)) # Service line (bottom half)

        # Optional extra points (intersections or midpoints)
        # Ensure these are floats if necessary for calculations
        self.top_extra_part = (832.5, 580) # Midpoint top baseline? No, slightly offset. Check usage.
        self.bottom_extra_part = (832.5, 2910) # Midpoint bottom baseline? Slightly offset. Check usage.

        # --- Ordered Keypoints for Homography ---
        # This specific order corresponds to the 14 points used in the dataset/model output
        # Index mapping based on dataset_example.png and likely model output channels 0-13:
        # 0: Top Left Baseline Corner
        # 1: Top Right Baseline Corner
        # 2: Bottom Left Baseline Corner
        # 3: Bottom Right Baseline Corner
        # 4: Top Left Service 'T' (Intersection of left_inner_line and top_inner_line)
        # 5: Bottom Left Service 'T' (Intersection of left_inner_line and bottom_inner_line)
        # 6: Top Right Service 'T' (Intersection of right_inner_line and top_inner_line)
        # 7: Bottom Right Service 'T' (Intersection of right_inner_line and bottom_inner_line)
        # 8: Top Center Service Line Point
        # 9: Bottom Center Service Line Point
        # 10: Left Net Post Intersection (Approximate - where left_court_line crosses net)
        # 11: Right Net Post Intersection (Approximate - where right_court_line crosses net)
        # 12: Center Net Top (Approximate - where middle_line crosses net) - Needs calculation if not explicit
        # 13: Court Center (Intersection of middle_line and center horizontal line) - Needs calculation

        # Define keypoints based on the line definitions above
        # Assuming the order shown in `imgs/dataset_example.png` (0-13 clockwise/structured)
        self.key_points = [
            self.left_court_line[0],    # 0: Top-Left Corner (baseline_top[0])
            self.right_court_line[0],   # 1: Top-Right Corner (baseline_top[1])
            self.left_court_line[1],    # 2: Bottom-Left Corner (baseline_bottom[0])
            self.right_court_line[1],   # 3: Bottom-Right Corner (baseline_bottom[1])

            self.left_inner_line[0],    # 4: Top-Left Inner Corner (near baseline)
            self.left_inner_line[1],    # 5: Bottom-Left Inner Corner (near baseline)
            self.right_inner_line[0],   # 6: Top-Right Inner Corner (near baseline)
            self.right_inner_line[1],   # 7: Bottom-Right Inner Corner (near baseline)

            self.top_inner_line[0],     # 8: Left Service Line Top Point (top_inner_line[0])
            self.top_inner_line[1],     # 9: Right Service Line Top Point (top_inner_line[1])
            self.bottom_inner_line[0],  # 10: Left Service Line Bottom Point (bottom_inner_line[0])
            self.bottom_inner_line[1],  # 11: Right Service Line Bottom Point (bottom_inner_line[1])

            self.middle_line[0],        # 12: Center Service Line Top Point
            self.middle_line[1]         # 13: Center Service Line Bottom Point
        ]
        # Ensure the number of keypoints matches model output (14)
        assert len(self.key_points) == 14, f"Expected 14 keypoints, but got {len(self.key_points)}"


        # Border points for mask generation (optional)
        self.border_points = [*self.baseline_top, *self.baseline_bottom[::-1]] # Clockwise

        # --- Court Configurations for Homography Robustness ---
        # Defines sets of 4 keypoints used to estimate homography.
        # Each key is an index into the `self.key_points` list.
        # These configurations should represent geometrically stable sets of points.
        self.court_conf = {
            1: [0, 1, 3, 2], # Outer baseline corners
            2: [4, 6, 7, 5], # Inner baseline corners (ends of inner side lines)
            3: [8, 9, 11, 10], # Service line corners
            4: [0, 1, 9, 8], # Top half baseline + service line tops
            5: [2, 3, 11, 10], # Bottom half baseline + service line bottoms
            6: [0, 4, 12, 8], # Top-left quad (approx) using center service top
            7: [1, 6, 12, 9], # Top-right quad (approx)
            8: [2, 5, 13, 10], # Bottom-left quad (approx) using center service bottom
            9: [3, 7, 13, 11], # Bottom-right quad (approx)
            10: [4, 8, 12, 6], # Top inner rectangle using service T's and center service top (Careful with order: 4, 6, 12, 8?) check geom
            11: [5, 10, 13, 7],# Bottom inner rectangle (5, 7, 13, 10?) check geom
            12: [0, 1, 6, 4] # Another top baseline + inner corners combination
            # Add more configurations if needed
        }
        # Validate configuration indices
        max_kp_index = len(self.key_points) - 1
        for conf_id, indices in self.court_conf.items():
            assert len(indices) == 4, f"Configuration {conf_id} must have 4 points."
            for idx in indices:
                assert 0 <= idx <= max_kp_index, f"Index {idx} in configuration {conf_id} is out of bounds (0-{max_kp_index})."


        # --- Court Dimensions (based on the reference coordinates) ---
        self.line_width = 1 # Used for drawing reference court image
        # Calculate dimensions based on key points if needed, or use provided constants
        self.court_width = self.baseline_top[1][0] - self.baseline_top[0][0] # 1379 - 286 = 1093? No, 1117 provided. Check points.
        # Let's recalculate based on provided width/height if points are approximate
        # Provided:
        self.court_width = 1117
        self.court_height = 2408 # From baseline top Y to baseline bottom Y: 2935 - 561 = 2374? No, 2408 provided.
        self.top_bottom_border = 549 # Margin size above/below court lines
        self.right_left_border = 274 # Margin size left/right of court lines

        # Total dimensions of the reference image to be generated
        self.court_total_width = self.court_width + self.right_left_border * 2
        self.court_total_height = self.court_height + self.top_bottom_border * 2
        # Note: The keypoint coordinates must be consistent with these dimensions and borders.
        # Example: Top-left point (286, 561) should correspond to (right_left_border, top_bottom_border)
        # Check: 286 == 274? No. 561 == 549? No.
        # It seems the coordinates are absolute within a larger canvas, not relative to the borders.
        # Recalculate total width/height based on max coordinate values found in lines?
        all_coords = np.array(self.get_all_line_coords())
        max_x, max_y = np.max(all_coords, axis=0)
        min_x, min_y = np.min(all_coords, axis=0)
        # Use these if generating the court dynamically, otherwise trust the provided total dimensions.
        # Let's assume the provided total width/height are correct for the canvas size.
        # self.court_total_width = max_x + self.right_left_border # ? Needs clarification
        # self.court_total_height = max_y + self.top_bottom_border # ? Needs clarification

        # Generate the court reference image upon initialization
        self.court = self.build_court_reference()

    def get_all_line_coords(self):
        """Returns a list of all coordinates used in line definitions."""
        coords = []
        lines_tuples = [
            self.baseline_top, self.baseline_bottom, self.net_coords, self.left_court_line,
            self.right_court_line, self.left_inner_line, self.right_inner_line,
            self.middle_line, self.top_inner_line, self.bottom_inner_line
        ]
        for line in lines_tuples:
            coords.extend(line)
        return coords

    def build_court_reference(self):
        """
        Create court reference image using the line positions.
        The size is determined by the maximum coordinate values plus some margin,
        or using the pre-calculated total width/height.
        Using pre-calculated total width/height.
        """
        # Ensure coordinates are integers for drawing
        def to_int_tuple(point):
            return tuple(map(int, point))

        # Create a blank canvas
        # Use calculated total size. Ensure it's large enough for all points.
        # Add a safety margin just in case
        safety_margin = 10
        canvas_height = self.court_total_height + safety_margin
        canvas_width = self.court_total_width + safety_margin
        court = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

        # Draw lines with thickness
        line_thickness = 5 # Make lines thicker for better visibility / matching real courts
        line_color = 255 # White lines on black background

        cv2.line(court, to_int_tuple(self.baseline_top[0]), to_int_tuple(self.baseline_top[1]), line_color, line_thickness)
        cv2.line(court, to_int_tuple(self.baseline_bottom[0]), to_int_tuple(self.baseline_bottom[1]), line_color, line_thickness)
        cv2.line(court, to_int_tuple(self.net_coords[0]), to_int_tuple(self.net_coords[1]), line_color, line_thickness)
        cv2.line(court, to_int_tuple(self.top_inner_line[0]), to_int_tuple(self.top_inner_line[1]), line_color, line_thickness)
        cv2.line(court, to_int_tuple(self.bottom_inner_line[0]), to_int_tuple(self.bottom_inner_line[1]), line_color, line_thickness)
        cv2.line(court, to_int_tuple(self.left_court_line[0]), to_int_tuple(self.left_court_line[1]), line_color, line_thickness)
        cv2.line(court, to_int_tuple(self.right_court_line[0]), to_int_tuple(self.right_court_line[1]), line_color, line_thickness)
        cv2.line(court, to_int_tuple(self.left_inner_line[0]), to_int_tuple(self.left_inner_line[1]), line_color, line_thickness)
        cv2.line(court, to_int_tuple(self.right_inner_line[0]), to_int_tuple(self.right_inner_line[1]), line_color, line_thickness)
        cv2.line(court, to_int_tuple(self.middle_line[0]), to_int_tuple(self.middle_line[1]), line_color, line_thickness)

        # Optional: Dilate lines further if needed
        # kernel = np.ones((5, 5), dtype=np.uint8)
        # court = cv2.dilate(court, kernel)

        # Store the generated court image
        self.court = court
        return court

    def get_important_lines(self):
        """
        Returns all defined lines as pairs of points.
        """
        lines = [
            self.baseline_top, self.baseline_bottom, self.net_coords, self.top_inner_line,
            self.bottom_inner_line, self.left_court_line, self.right_court_line,
            self.left_inner_line, self.right_inner_line, self.middle_line
        ]
        return lines

    def get_keypoints_array(self):
        """Returns the keypoints as a NumPy array."""
        return np.array(self.key_points, dtype=np.float32)

    def get_extra_parts(self):
        """Returns extra defined points."""
        parts = [self.top_extra_part, self.bottom_extra_part]
        return parts

    def save_all_court_configurations(self, output_dir='court_configurations'):
        """
        Draws the reference court and highlights the points used in each homography configuration.
        Saves the images to the specified directory.
        """
        os.makedirs(output_dir, exist_ok=True)
        base_court_bgr = cv2.cvtColor(self.court, cv2.COLOR_GRAY2BGR) # Convert to BGR for color drawing

        for i, conf_indices in self.court_conf.items():
            c = base_court_bgr.copy() # Work on a copy
            # Highlight the 4 points for this configuration
            for kp_index in conf_indices:
                p = tuple(map(int, self.key_points[kp_index]))
                cv2.circle(c, p, radius=15, color=(0, 0, 255), thickness=-1) # Red filled circles
                cv2.putText(c, str(kp_index), (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

            # Draw all keypoints faintly for context
            for idx, p_all in enumerate(self.key_points):
                 p_int = tuple(map(int, p_all))
                 cv2.circle(c, p_int, radius=8, color=(0, 255, 0), thickness=2) # Green circles


            save_path = os.path.join(output_dir, f'court_conf_{i}.png')
            cv2.imwrite(save_path, c)
            print(f"Saved configuration {i} visualization to {save_path}")


    def get_court_mask(self, mask_type=0):
        """
        Get a mask of the court area based on different types.
        Coordinates are relative to the generated reference court image.
        """
        mask = np.zeros_like(self.court, dtype=np.uint8) # Start with black mask

        # Define court polygon using outer corners (indices 0, 1, 3, 2)
        # Ensure points are int32 for fillPoly
        outer_poly = np.array([self.key_points[0], self.key_points[1], self.key_points[3], self.key_points[2]], dtype=np.int32)

        if mask_type == 0: # Full court area including lines
            cv2.fillPoly(mask, [outer_poly], 255)
        elif mask_type == 1:  # Bottom half court (below net)
            # Define polygon for bottom half
            bottom_poly = np.array([
                self.key_points[2], self.key_points[3], # Bottom baseline
                tuple(map(int, self.net_coords[1])), tuple(map(int, self.net_coords[0])) # Net line
            ], dtype=np.int32)
            cv2.fillPoly(mask, [bottom_poly], 255)
        elif mask_type == 2:  # Top half court (above net)
             # Define polygon for top half
            top_poly = np.array([
                self.key_points[0], self.key_points[1], # Top baseline
                tuple(map(int, self.net_coords[1])), tuple(map(int, self.net_coords[0])) # Net line
            ], dtype=np.int32)
            cv2.fillPoly(mask, [top_poly], 255)
        elif mask_type == 3: # Court playing area only (inside baselines and sidelines)
            # Same as type 0 for this definition
             cv2.fillPoly(mask, [outer_poly], 255)

        # Ensure mask is binary 0 or 1 if needed downstream, otherwise 0 or 255 is fine
        # return mask / 255 if needed
        return mask


if __name__ == '__main__':
    import os
    print("Initializing CourtReference...")
    c = CourtReference()
    print("Court reference image generated.")
    print(f"Reference court image shape: {c.court.shape}")
    print(f"Number of keypoints defined: {len(c.key_points)}")

    # Example: Save the generated court reference image
    ref_img_path = 'court_reference_generated.png'
    cv2.imwrite(ref_img_path, c.court)
    print(f"Saved generated reference court image to {ref_img_path}")

    # Example: Save visualizations of homography configurations
    print("Generating homography configuration visualizations...")
    c.save_all_court_configurations()
    print("Visualizations saved.")

    # Example: Get a mask and save it
    mask_play_area = c.get_court_mask(mask_type=3)
    mask_path = 'court_mask_play_area.png'
    cv2.imwrite(mask_path, mask_play_area)
    print(f"Saved play area mask to {mask_path}")

# --- END OF FILE court_reference.py ---