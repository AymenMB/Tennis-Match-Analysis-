import cv2
import numpy as np
import torch
from track_net import Track
import torch.nn.functional as F
from tqdm import tqdm
from post_process import PostProcess
from homography import get_trans_matrix, refer_kps

class CourtDetectionNet:
    def __init__(self, path_model=None,  device='cuda'):
        self.model = Track(out_channels=15)
        self.device = device
        if path_model:
            self.model.load_state_dict(torch.load(path_model, map_location=device))
            self.model = self.model.to(device)
            self.model.eval()
            
    def infer_model(self, frames, confidence_threshold=0.5): # Added confidence_threshold parameter
        output_width = 640
        output_height = 360
        scale = 2
        
        kps_res = []
        matrixes_res = []
        for num_frame, image in enumerate(tqdm(frames)):
            img = cv2.resize(image, (output_width, output_height))
            inp = (img.astype(np.float32) / 255.)
            inp = torch.tensor(np.rollaxis(inp, 2, 0))
            inp = inp.unsqueeze(0)

            out = self.model(inp.float().to(self.device))[0]
            pred = F.sigmoid(out).detach().cpu().numpy()

            points = []
            for kps_num in range(14):
                # Check confidence before processing heatmap
                heatmap_raw = pred[kps_num]
                max_confidence = np.max(heatmap_raw) # Get max confidence from the raw heatmap

                if max_confidence > confidence_threshold:
                    # Use the previous HoughCircles logic
                    heatmap = (heatmap_raw * 255).astype(np.uint8)
                    # Apply thresholding
                    ret, heatmap_thresh = cv2.threshold(heatmap, 170, 255, cv2.THRESH_BINARY)
                    # Detect circles
                    circles = cv2.HoughCircles(heatmap_thresh, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=2,
                                               minRadius=10, maxRadius=25) # Use heatmap_thresh here
                    
                    if circles is not None:
                        # Extract coordinates from the first detected circle
                        x_pred = circles[0][0][0] * scale
                        y_pred = circles[0][0][1] * scale
                        
                        # Apply refinement as before for specific keypoints
                        if kps_num not in [8, 12, 9]: 
                            refined_coords = PostProcess.refine_kps(image, int(y_pred), int(x_pred), crop_size=40)
                            if refined_coords:
                                x_pred, y_pred = refined_coords # refine_kps returns y, x
                        points.append((x_pred, y_pred))                
                    else:
                        # No circle found even if confidence was high
                        points.append(None)
                else:
                    # Confidence below threshold
                    points.append(None)

            # --- Matrix calculation and inversion remains the same ---
            matrix_trans = get_trans_matrix(points) 
            transformed_points = None
            inv_matrix_trans = None # Initialize inverse matrix
            if matrix_trans is not None:
                if refer_kps is not None:
                    try:
                        refer_kps_float = refer_kps.astype(np.float32)
                        # Calculate transformed points using the original matrix (image -> ref)
                        transformed_points = cv2.perspectiveTransform(refer_kps_float, matrix_trans)
                    except cv2.error as e:
                        print(f"Frame {num_frame}: Error during perspectiveTransform: {e}")
                        transformed_points = None
                
                # Calculate the inverse matrix (ref -> image)
                inv_matrix_result = cv2.invert(matrix_trans)
                if inv_matrix_result[0]:
                    inv_matrix_trans = inv_matrix_result[1] # Store the inverse matrix
                else:
                    # Failed to invert matrix
                    inv_matrix_trans = None
            
            # Store the transformed points (image coords) and the inverse matrix (ref->image)
            kps_res.append(transformed_points)
            matrixes_res.append(inv_matrix_trans) # Append the inverse matrix
            
        return matrixes_res, kps_res