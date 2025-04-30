# --- person_detector.py (Minimally Modified for GPU & New Libraries) ---

import torchvision
import cv2
import torch
from court_reference_model import CourtReferenceModel
# from scipy import signal # Seems unused, can be commented out
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm
# *** CHANGE 1: Import weights enum ***
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

class PersonDetection:
    # Modify __init__ to accept and store the confidence threshold
    def __init__(self, device='cpu', person_min_score=0.85): # Added person_min_score parameter
        """Initializes the PersonDetection."""
        self.device = device
        # Store the minimum score threshold
        self.person_min_score = person_min_score 
        print(f"Initializing PersonDetection on device: {self.device} with min_score: {self.person_min_score}")

        # *** CHANGE 4: Use weights API instead of pretrained=True ***
        print("Loading Faster R-CNN ResNet50 FPN model with default weights...")
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

        # *** CHANGE 5: Store the required preprocessing transforms ***
        self.preprocess = weights.transforms()

        # *** CHANGE 6: Move model to the correct device ('cuda' or 'cpu') ***
        print(f"Moving model to device: {self.device}")
        self.detection_model = self.detection_model.to(self.device) # Use self.device here
        self.detection_model.eval()
        print("Model loaded and set to evaluation mode.")

        # --- Court reference setup (unchanged) ---
        self.court_ref = CourtReferenceModel()
        self.ref_top_court = self.court_ref.get_court_mask(2)
        self.ref_bottom_court = self.court_ref.get_court_mask(1)
        # --- Other attributes (unchanged) ---
        self.point_person_top = None
        self.point_person_bottom = None
        self.counter_top = 0
        self.counter_bottom = 0


    def detect(self, image): # Removed person_min_score from here, uses self.person_min_score now
        """Detects persons in a single image."""
        PERSON_LABEL = 1 # COCO label for person

        # 1. Preprocessing
        # Ensure uint8, convert BGR->RGB (needed for standard transforms)
        if image.dtype != np.uint8: image = image.astype(np.uint8)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert to tensor [C, H, W]
        input_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1)
        # Apply the standard transforms, converting to float is usually handled within
        input_tensor = self.preprocess(input_tensor) # Apply transforms

        # *** CHANGE 7: Move input tensor to the correct device ***
        # Wrap in a list for the model and move the tensor to self.device
        batch = [input_tensor.to(self.device)] # Use self.device here

        # 2. Inference
        with torch.no_grad():
            # Model and batch are now on the same device (CPU or GPU)
            preds = self.detection_model(batch) # Pass the processed batch

        # 3. Postprocessing (move results back to CPU)
        persons_boxes = []
        probs = []
        # Original code correctly moved results to CPU before numpy conversion
        for box, label, score in zip(preds[0]['boxes'].cpu(), preds[0]['labels'].cpu(), preds[0]['scores'].cpu()):
            # Use the stored threshold
            if label == PERSON_LABEL and score > self.person_min_score: 
                persons_boxes.append(box.detach().numpy())
                probs.append(score.detach().numpy()) # Keep original .numpy() output
        return persons_boxes, probs

    def detect_top_and_bottom_players(self, image, inv_matrix, filter_players=False):
        # Added basic error handling for matrix inversion
        try:
            matrix = cv2.invert(inv_matrix)[1] # This is ref->image matrix
        except cv2.error:
            # print(f"Warning: cv2.invert failed for matrix.")
            return [], [] # Return empty if inversion fails

        # Added basic error handling for warpPerspective
        img_height, img_width = image.shape[:2]
        try:
            mask_top_court = cv2.warpPerspective(self.ref_top_court, matrix, (img_width, img_height), flags=cv2.INTER_NEAREST) # Use INTER_NEAREST for masks
            mask_bottom_court = cv2.warpPerspective(self.ref_bottom_court, matrix, (img_width, img_height), flags=cv2.INTER_NEAREST)
        except cv2.error as e:
            # print(f"Warning: cv2.warpPerspective failed: {e}")
            return [], []

        person_bboxes_top, person_bboxes_bottom = [], []

        # Calls self.detect() which now uses the stored threshold
        bboxes, probs = self.detect(image) 

        if len(bboxes) > 0:
            person_points = [[int((bbox[2] + bbox[0]) / 2), int(bbox[3])] for bbox in bboxes]
            person_bboxes = list(zip(bboxes, person_points))

            # Classify players (added bounds check for safety)
            for pt in person_bboxes:
                 px, py = pt[1]
                 if 0 <= py < img_height and 0 <= px < img_width:
                     if mask_top_court[py, px] > 0: # Check > 0
                         person_bboxes_top.append(pt)
                     elif mask_bottom_court[py, px] > 0:
                         person_bboxes_bottom.append(pt)

            if filter_players:
                # filter_players uses the ref->image matrix
                person_bboxes_top, person_bboxes_bottom = self.filter_players(person_bboxes_top, person_bboxes_bottom, matrix)

        return person_bboxes_top, person_bboxes_bottom

    def filter_players(self, person_bboxes_top, person_bboxes_bottom, matrix):
        """
        Leave one person at the top and bottom of the tennis court
        """
        if matrix is None: return person_bboxes_top, person_bboxes_bottom
        if len(self.court_ref.key_points) < 14: return person_bboxes_top, person_bboxes_bottom

        # Uses keypoints 12 & 13
        refer_kps = np.array(self.court_ref.key_points[12:], dtype=np.float32).reshape((-1, 1, 2))
        center_top_court = None
        center_bottom_court = None
        try:
            # Uses ref->image matrix
            trans_kps = cv2.perspectiveTransform(refer_kps, matrix)
            if trans_kps is not None and len(trans_kps) >= 2: 
                # Check if the transformed points are valid (e.g., not NaN)
                if np.all(np.isfinite(trans_kps[0][0])) and np.all(np.isfinite(trans_kps[1][0])):
                    center_top_court = trans_kps[0][0]
                    center_bottom_court = trans_kps[1][0]
                else:
                     raise cv2.error("Transformed keypoints contain non-finite values")
            else: 
                raise cv2.error("Transform failed or returned insufficient points")
        except cv2.error as e:
            print(f"Warning: perspectiveTransform failed in filter_players: {e}")
            return person_bboxes_top, person_bboxes_bottom 

        # Proceed only if center points are valid
        if center_top_court is not None and len(person_bboxes_top) > 1:
            try:
                dists = [distance.euclidean(x[1], center_top_court) for x in person_bboxes_top]
                ind = np.argmin(dists) # Use numpy's argmin
                person_bboxes_top = [person_bboxes_top[ind]]
            except Exception as e: # Catch potential errors during distance calculation
                 print(f"Warning: Error calculating distances for top players: {e}")
        
        if center_bottom_court is not None and len(person_bboxes_bottom) > 1:
            try:
                dists = [distance.euclidean(x[1], center_bottom_court) for x in person_bboxes_bottom]
                ind = np.argmin(dists) # Use numpy's argmin
                person_bboxes_bottom = [person_bboxes_bottom[ind]]
            except Exception as e: # Catch potential errors during distance calculation
                 print(f"Warning: Error calculating distances for bottom players: {e}")

        return person_bboxes_top, person_bboxes_bottom

    def track_players(self, frames, matrix_all, filter_players=False):
        persons_top = []
        persons_bottom = []
        num_frames = len(frames)
        num_matrices = len(matrix_all)
        print("Tracking players...") # Added print statement for clarity

        # Use tqdm for progress bar
        for num_frame in tqdm(range(num_frames)):
            img = frames[num_frame]
            # Get matrix for frame i, assume matrix_all is inv_matrix (ref->image) as per original call structure
            inv_matrix = matrix_all[num_frame] if num_frame < num_matrices else None

            # Call detect_top_and_bottom_players which handles None matrix internally
            person_top, person_bottom = self.detect_top_and_bottom_players(img, inv_matrix, filter_players)

            persons_top.append(person_top)
            persons_bottom.append(person_bottom)

        print("Finished tracking players.") # Added print statement
        return persons_top, persons_bottom