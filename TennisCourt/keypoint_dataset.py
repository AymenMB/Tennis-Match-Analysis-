
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import json
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import draw_umich_gaussian, line_intersection, is_point_in_image

# Define Albumentations pipelines
# Input images are 1280x720, network expects 640x360
# Output heatmaps are 360x640 based on network stride
# Augmentations applied *before* final resize to network input size for better quality

# --- Configuration ---
TARGET_INPUT_WIDTH = 1280 # Original image width for loading
TARGET_INPUT_HEIGHT = 720 # Original image height for loading
NETWORK_INPUT_WIDTH = 640 # Width the network expects
NETWORK_INPUT_HEIGHT = 360 # Height the network expects
HEATMAP_SCALE = 1 # Scale factor from network output to heatmap size (assuming 1 if output is 360x640)
# If network output was e.g. 180x320, HEATMAP_SCALE would be 2
OUTPUT_WIDTH = NETWORK_INPUT_WIDTH // HEATMAP_SCALE
OUTPUT_HEIGHT = NETWORK_INPUT_HEIGHT // HEATMAP_SCALE
NUM_JOINTS = 14 # Base keypoints from dataset
TOTAL_KEYPOINTS = NUM_JOINTS + 1 # Including derived center point
HP_RADIUS = 5 # Radius for Gaussian heatmap generation (Adjust based on OUTPUT resolution)

# --- Augmentations ---
# Keypoint format for Albumentations: [x, y] normalized to [0, 1] or pixel values
# Here we use pixel values, specify format='xy' and image shape

train_transform = A.Compose([
    # Color/Intensity Augmentations (applied first)
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),

    # Geometric Augmentations (applied before final resize)
    # Use small values to avoid distorting the court too much
    A.Affine(
        scale=(0.95, 1.05),      # Slight zoom in/out
        translate_percent=(-0.05, 0.05), # Slight translation
        rotate=(-3, 3),         # Slight rotation
        shear=(-3, 3),          # Slight shear
        p=0.7
    ),

    # Occlusion-like Augmentation
    A.CoarseDropout(max_holes=4, max_height=40, max_width=40, min_holes=1, min_height=20, min_width=20, fill_value=0, p=0.3),

    # Final Resize to Network Input Size
    A.Resize(height=NETWORK_INPUT_HEIGHT, width=NETWORK_INPUT_WIDTH, interpolation=cv2.INTER_LINEAR),

    # Normalization & Tensor Conversion
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # Imagenet stats commonly used
    ToTensorV2(), # Converts image to CxHxW tensor and scales to [0, 1] if not normalized
], keypoint_params=A.KeypointParams(format='xy', label_fields=[], remove_invisible=False)) # label_fields needed if keypoints have labels

val_transform = A.Compose([
    A.Resize(height=NETWORK_INPUT_HEIGHT, width=NETWORK_INPUT_WIDTH, interpolation=cv2.INTER_LINEAR),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], keypoint_params=A.KeypointParams(format='xy', label_fields=[], remove_invisible=False))


class courtDataset(Dataset):

    def __init__(self, mode, hp_radius=HP_RADIUS):
        self.mode = mode
        assert mode in ['train', 'val'], 'incorrect mode'
        self.input_height = TARGET_INPUT_HEIGHT # Original load size
        self.input_width = TARGET_INPUT_WIDTH
        self.output_height = OUTPUT_HEIGHT
        self.output_width = OUTPUT_WIDTH
        self.num_joints = NUM_JOINTS
        self.total_keypoints = TOTAL_KEYPOINTS
        self.hp_radius = hp_radius
        # self.scale = TARGET_INPUT_HEIGHT / OUTPUT_HEIGHT # Overall scale factor from original to heatmap

        self.path_dataset = './data'
        self.path_images = os.path.join(self.path_dataset, 'images')
        json_path = os.path.join(self.path_dataset, f'data_{mode}.json')
        if not os.path.exists(json_path):
             raise FileNotFoundError(f"Annotation file not found: {json_path}")
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        print(f'mode = {mode}, len = {len(self.data)}')

        # Select the appropriate transform pipeline
        self.transform = train_transform if mode == 'train' else val_transform

    # Optional: Filter data (example remains the same)
    def filter_data(self):
        # This function filters based on original keypoints being within original image bounds
        # Might need adjustment if filtering based on transformed points is desired
        new_data = []
        for i in range(len(self.data)):
            kps_array = np.array(self.data[i]['kps'])
            # Check for None or invalid coordinates before min/max
            valid_kps = np.array([kp for kp in kps_array if kp[0] is not None and kp[1] is not None])
            if len(valid_kps) < self.num_joints: # Skip if too many keypoints are missing initially
                 continue
            max_elems = valid_kps.max(axis=0)
            min_elems = valid_kps.min(axis=0)
            if max_elems[0] < self.input_width and min_elems[0] >= 0 and \
               max_elems[1] < self.input_height and min_elems[1] >= 0:
                new_data.append(self.data[i])
        print(f"Filtered data from {len(self.data)} to {len(new_data)}")
        self.data = new_data
        return new_data

    def __getitem__(self, index):
        item = self.data[index]
        img_id = item['id']
        img_name = img_id + '.png'
        img_path = os.path.join(self.path_images, img_name)

        if not os.path.exists(img_path):
            print(f"Warning: Image file not found: {img_path}")
            # Handle missing file: return dummy data or skip? Returning dummy for now.
            # Adjust dimensions as needed
            dummy_inp = torch.zeros((3, NETWORK_INPUT_HEIGHT, NETWORK_INPUT_WIDTH), dtype=torch.float32)
            dummy_hm = torch.zeros((self.total_keypoints, self.output_height, self.output_width), dtype=torch.float32)
            dummy_kps = np.full((self.num_joints, 2), -1, dtype=int) # Use -1 to indicate invalid
            return dummy_inp, dummy_hm, dummy_kps, img_id

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to read image: {img_path}")
            dummy_inp = torch.zeros((3, NETWORK_INPUT_HEIGHT, NETWORK_INPUT_WIDTH), dtype=torch.float32)
            dummy_hm = torch.zeros((self.total_keypoints, self.output_height, self.output_width), dtype=torch.float32)
            dummy_kps = np.full((self.num_joints, 2), -1, dtype=int)
            return dummy_inp, dummy_hm, dummy_kps, img_id

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Albumentations expects RGB

        # Load original keypoints, handle potential None values
        original_kps = []
        for kp in item['kps']:
            if kp[0] is None or kp[1] is None:
                 # Append a placeholder (e.g., [-1, -1]) for Albumentations
                 # It needs a point for every index, even if invalid/invisible
                 original_kps.append([-1, -1])
            else:
                 original_kps.append([float(kp[0]), float(kp[1])])

        # Apply transformations
        # Keypoints are provided in pixel coordinates (format='xy')
        transformed = self.transform(image=img, keypoints=original_kps)
        inp = transformed['image'] # This is already a normalized tensor
        transformed_kps = transformed['keypoints'] # These are keypoints *after* augmentation and resize

        # --- Create Target Heatmaps ---
        hm_hp = np.zeros((self.total_keypoints, self.output_height, self.output_width), dtype=np.float32)
        draw_gaussian = draw_umich_gaussian # Function assumed from utils.py

        # Scale factor from network input size (after resize) to heatmap size
        hm_scale_x = self.output_width / NETWORK_INPUT_WIDTH
        hm_scale_y = self.output_height / NETWORK_INPUT_HEIGHT

        valid_transformed_kps_for_center = [] # Store valid points for center calculation

        for i in range(self.num_joints):
            # Get keypoint coordinates *after* transformation and resize
            x_kp_net, y_kp_net = transformed_kps[i]

            # Scale coordinates to the heatmap resolution
            x_pt_hm = int(x_kp_net * hm_scale_x)
            y_pt_hm = int(y_kp_net * hm_scale_y)

            # Check if the point is valid and within heatmap bounds *after* scaling
            if x_kp_net >= 0 and y_kp_net >= 0 and \
               x_pt_hm >= 0 and x_pt_hm < self.output_width and \
               y_pt_hm >= 0 and y_pt_hm < self.output_height:
                draw_gaussian(hm_hp[i], (x_pt_hm, y_pt_hm), self.hp_radius)
                # Store the valid point *at network input scale* for center calculation later
                valid_transformed_kps_for_center.append({'id': i, 'pt': (x_kp_net, y_kp_net)})
            # Else: Point is outside bounds or was invalid initially, leave heatmap as 0

        # Calculate center point using *transformed* valid keypoints (indices 0, 1, 2, 3)
        kps_dict = {kp['id']: kp['pt'] for kp in valid_transformed_kps_for_center}
        if all(k in kps_dict for k in [0, 1, 2, 3]):
            p0 = kps_dict[0]
            p1 = kps_dict[1]
            p2 = kps_dict[2]
            p3 = kps_dict[3]
            center_coords_net = line_intersection(
                (p0[0], p0[1], p3[0], p3[1]),
                (p1[0], p1[1], p2[0], p2[1])
            )
            if center_coords_net:
                x_ct_net, y_ct_net = center_coords_net
                # Scale center point to heatmap resolution
                x_ct_hm = int(x_ct_net * hm_scale_x)
                y_ct_hm = int(y_ct_net * hm_scale_y)
                # Check bounds before drawing
                if x_ct_hm >= 0 and x_ct_hm < self.output_width and y_ct_hm >= 0 and y_ct_hm < self.output_height:
                    draw_gaussian(hm_hp[self.num_joints], (x_ct_hm, y_ct_hm), self.hp_radius)

        # Return original KPs (before augmentation) for evaluation purposes if needed
        # Or return transformed KPs if evaluation should happen on augmented scale
        # Sticking to original evaluation: return original KPs (but handle None/-1)
        eval_kps = np.array([[kp[0] if kp[0]!=-1 else -1, kp[1] if kp[1]!=-1 else -1] for kp in original_kps], dtype=int)

        return inp, torch.from_numpy(hm_hp), eval_kps, img_id # Return torch tensor for heatmap

    def __len__(self):
        return len(self.data)

# --- END OF FILE dataset.py ---