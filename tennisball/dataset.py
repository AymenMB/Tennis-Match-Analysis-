import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import cv2
import math
import numpy as np

class TennisTrackDataset(Dataset):
    def __init__(self, mode, input_height=360, input_width=640, dataset_dir='./data/tennistrack'):
        self.path_dataset = dataset_dir
        assert mode in ['train', 'val'], 'incorrect mode'
        label_file = os.path.join(self.path_dataset, f'labels_{mode}.csv')
        if not os.path.exists(label_file):
            # Add absolute path to error message for easier debugging
            abs_path = os.path.abspath(label_file)
            raise FileNotFoundError(f"Label file not found: {label_file} (Absolute path: {abs_path})")
        self.data = pd.read_csv(label_file)
        print(f'mode = {mode}, samples = {self.data.shape[0]}, dataset_dir = {self.path_dataset}')
        self.height = input_height
        self.width = input_width

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path, path_prev, path_preprev, path_gt, x, y, status, vis = row

        path = os.path.join(self.path_dataset, path)
        path_prev = os.path.join(self.path_dataset, path_prev)
        path_preprev = os.path.join(self.path_dataset, path_preprev)
        path_gt = os.path.join(self.path_dataset, path_gt)

        # Ground truth coordinates (relative to input H/W) and visibility
        # Keep them as float for potential future use, but primarily used for validation metrics
        x_coord_gt = -1.0
        y_coord_gt = -1.0
        if pd.notna(x) and vis != 0:
            # Ensure coordinates are relative to the *input* size for consistency
            # (Assuming the coords in csv are for original 1280x720)
            # Note: gt_gen.py creates heatmaps based on original coords, but we load them and resize.
            # The x,y from csv are still useful for validation comparison.
            # If coords in csv are already for 360x640, remove scaling. Check Label.csv source.
            # Assuming they are for original size:
            # scale_x = self.width / 1280.0
            # scale_y = self.height / 720.0
            # x_coord_gt = float(x) * scale_x
            # y_coord_gt = float(y) * scale_y
            # *** If Label.csv coords are already for input size H/W, just use: ***
            x_coord_gt = float(x)
            y_coord_gt = float(y)


        inputs = self.get_input(path, path_prev, path_preprev)
        # Get GT heatmap as float tensor (1, H, W), normalized [0, 1]
        outputs = self.get_output(path_gt) # <<<< CHANGED: gets float heatmap

        # Return GT coordinates and visibility along with input/output tensors
        return inputs, outputs, x_coord_gt, y_coord_gt, int(vis)

    def get_output(self, path_gt):
        # Returns the GT heatmap as a normalized float tensor (1, H, W)
        heatmap_shape = (1, self.height, self.width)
        if not os.path.exists(path_gt):
            print(f"Warning: Ground truth file not found: {path_gt}. Returning zeros.")
            return np.zeros(heatmap_shape, dtype=np.float32)

        img = cv2.imread(path_gt)
        if img is None:
            print(f"Warning: Failed to read ground truth file: {path_gt}. Returning zeros.")
            return np.zeros(heatmap_shape, dtype=np.float32)

        img = cv2.resize(img, (self.width, self.height))
        img = img[:, :, 0] # Take single channel (grayscale heatmap)

        # Normalize heatmap to [0.0, 1.0] and ensure float32
        img = img.astype(np.float32) / 255.0

        # Add channel dimension: (H, W) -> (1, H, W)
        img = np.expand_dims(img, axis=0)
        return img # Return float heatmap for MSELoss

    def get_input(self, path, path_prev, path_preprev):
        # Returns the 3 stacked frames as a float tensor (9, H, W)
        img = self._read_and_resize(path)
        img_prev = self._read_and_resize(path_prev)
        img_preprev = self._read_and_resize(path_preprev)

        if img is None or img_prev is None or img_preprev is None:
            print(f"Warning: Could not read one or more input images ({path}, {path_prev}, {path_preprev}). Returning zeros.")
            return np.zeros((9, self.height, self.width), dtype=np.float32)

        # Concatenate along the channel axis (H, W, C) -> (H, W, 9)
        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        # Normalize to [0.0, 1.0]
        imgs = imgs.astype(np.float32) / 255.0
        # Roll axis: (H, W, C) -> (C, H, W)
        imgs = np.rollaxis(imgs, 2, 0)
        return imgs

    def _read_and_resize(self, img_path):
        if not os.path.exists(img_path):
            print(f"Warning: Input image file not found: {img_path}")
            return None
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to read input image: {img_path}")
            return None
        img = cv2.resize(img, (self.width, self.height))
        return img