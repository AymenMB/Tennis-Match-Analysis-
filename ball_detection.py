from track_net import Track
import torch
import cv2
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm
from scipy.ndimage import maximum_filter

class BallDetection:
    def __init__(self, path_model=None, device='cuda'):
        self.model = Track(input_channels=9, out_channels=256)
        self.device = device
        if path_model:
            self.model.load_state_dict(torch.load(path_model, map_location=device))
            self.model = self.model.to(device)
            self.model.eval()
        self.width = 640
        self.height = 360

    def infer_model(self, frames):
        """ Run pretrained model on a consecutive list of frames
        :params
            frames: list of consecutive video frames
        :return
            ball_track: list of detected ball points
        """
        ball_track = [(None, None)]*2
        prev_pred = [None, None]
        for num in tqdm(range(2, len(frames))):
            img = cv2.resize(frames[num], (self.width, self.height))
            img_prev = cv2.resize(frames[num-1], (self.width, self.height))
            img_preprev = cv2.resize(frames[num-2], (self.width, self.height))
            imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
            imgs = imgs.astype(np.float32)/255.0
            imgs = np.rollaxis(imgs, 2, 0)
            inp = np.expand_dims(imgs, axis=0)

            out = self.model(torch.from_numpy(inp).float().to(self.device))
            output = out.argmax(dim=1).detach().cpu().numpy()
            x_pred, y_pred = self.postprocess(output, prev_pred)
            prev_pred = [x_pred, y_pred]
            ball_track.append((x_pred, y_pred))
        return ball_track

    def postprocess(self, feature_map, prev_pred, threshold=0.3, scale=2, max_dist=80):
        """
        :params
            feature_map: feature map with shape (1,360,640) from the model
            prev_pred: [x,y] coordinates of ball prediction from previous frame
            threshold: minimum confidence threshold to consider a detection
            scale: scale for conversion to original shape (720,1280)
            max_dist: maximum distance from previous ball detection to remove outliers
        :return
            x,y ball coordinates
        """
        # Assuming feature_map represents probabilities or confidence scores
        feature_map = feature_map.reshape((self.height, self.width))

        # Find the coordinates of the maximum value
        y_idx, x_idx = np.unravel_index(np.argmax(feature_map), feature_map.shape)
        max_confidence = feature_map[y_idx, x_idx]

        x, y = None, None
        if max_confidence > threshold: # Check if the max confidence is above threshold
            x_temp = x_idx * scale
            y_temp = y_idx * scale

            # Apply distance filtering if a previous prediction exists
            if prev_pred[0] is not None:
                dist = distance.euclidean((x_temp, y_temp), prev_pred)
                if dist < max_dist:
                    x, y = x_temp, y_temp
            else:
                # If no previous prediction, accept the first confident detection
                x, y = x_temp, y_temp
                
        return x, y