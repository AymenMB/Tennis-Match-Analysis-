# --- START OF FILE infer_in_video.py ---

import os
import cv2
import numpy as np
import torch
from court_keypoint_net import CourtKeypointNet
import torch.nn.functional as F
from tqdm import tqdm
from postprocess import postprocess, refine_kps # Use updated postprocess
from homography import get_trans_matrix, court_ref # Use updated homography
import argparse

# Configuration (Similar to image inference)
NETWORK_INPUT_WIDTH = 640
NETWORK_INPUT_HEIGHT = 360
HEATMAP_THRESHOLD = 0.1
REFINE_KPS_EXCLUDE = [8, 9, 12, 13]
DRAW_RADIUS = 5
DRAW_THICKNESS = -1
DRAW_COLOR = (0, 0, 255) # Red BGR

def read_video(path_video):
    """ Read video file """
    cap = cv2.VideoCapture(path_video)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {path_video}")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    print("Reading video frames...")
    while True:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    print(f"Read {len(frames)} frames at {fps} FPS.")
    return frames, fps

def write_video(frames_out, fps, path_output_video):
    """ Write frames to video file """
    if not frames_out:
        print("No frames to write.")
        return
    height, width = frames_out[0].shape[:2]
    # Ensure output directory exists
    output_dir = os.path.dirname(path_output_video)
    if output_dir:
         os.makedirs(output_dir, exist_ok=True)
    # Use a common codec like MP4V for .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Or 'XVID', 'DIVX'
    out = cv2.VideoWriter(path_output_video, fourcc, fps, (width, height))
    print(f"Writing video to {path_output_video}...")
    for frame in tqdm(frames_out, desc="Writing Video"):
        out.write(frame)
    out.release()
    print("Video writing complete.")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='path to trained model (.pt file)')
    parser.add_argument('--input_path', type=str, required=True, help='path to input video')
    parser.add_argument('--output_path', type=str, required=True, help='path to save output video')
    parser.add_argument('--use_refine_kps', action='store_true', help='whether to use refine kps postprocessing')
    parser.add_argument('--use_homography', action='store_true', help='whether to use homography postprocessing')
    args = parser.parse_args()

    # Load Model (same logic as image inference)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("Loading model...")
    model = CourtKeypointNet(out_channels=15)
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if not isinstance(model, torch.nn.DataParallel) and all(key.startswith('module.') for key in state_dict.keys()):
         state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    elif isinstance(model, torch.nn.DataParallel) and not all(key.startswith('module.') for key in state_dict.keys()):
         state_dict = {'module.' + k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("Model loaded.")

    # Read Video
    frames, fps = read_video(args.input_path)
    if not frames:
        print("No frames found in video. Exiting.")
        exit()

    frames_out = []
    orig_h, orig_w = frames[0].shape[:2]

    # Process each frame
    print("Processing video frames...")
    for frame_orig in tqdm(frames, desc="Processing Frames"):
        # Preprocess frame (Resize, Normalize, Tensor)
        img_resized = cv2.resize(frame_orig, (NETWORK_INPUT_WIDTH, NETWORK_INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
        inp = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        inp = (inp.astype(np.float32) / 255.)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        inp = (inp - mean) / std
        inp = torch.from_numpy(np.rollaxis(inp, 2, 0)).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            out_logits = model(inp)
        pred_hm_sig = torch.sigmoid(out_logits)
        pred_hm_np = pred_hm_sig.squeeze(0).cpu().numpy()

        # Post-process: Extract points relative to network input size
        points_pred_resized = []
        heatmap_h, heatmap_w = pred_hm_np.shape[1], pred_hm_np.shape[2]
        scale_hm_to_net_x = NETWORK_INPUT_WIDTH / heatmap_w
        scale_hm_to_net_y = NETWORK_INPUT_HEIGHT / heatmap_h
        for kps_num in range(14):
            heatmap = pred_hm_np[kps_num]
            x_net, y_net = postprocess(heatmap,
                                       scale_factor=(scale_hm_to_net_x, scale_hm_to_net_y),
                                       threshold=HEATMAP_THRESHOLD)
            points_pred_resized.append((x_net, y_net))

        # Optional: Refine Keypoints (scaled to original frame size)
        points_to_use_orig = [None] * 14
        scale_net_to_orig_x = orig_w / NETWORK_INPUT_WIDTH
        scale_net_to_orig_y = orig_h / NETWORK_INPUT_HEIGHT
        if args.use_refine_kps:
            for i, pt_net in enumerate(points_pred_resized):
                if pt_net[0] is not None and i not in REFINE_KPS_EXCLUDE:
                    x_orig_est = pt_net[0] * scale_net_to_orig_x
                    y_orig_est = pt_net[1] * scale_net_to_orig_y
                    y_refined, x_refined = refine_kps(frame_orig, int(y_orig_est), int(x_orig_est))
                    points_to_use_orig[i] = (x_refined, y_refined)
                elif pt_net[0] is not None:
                     points_to_use_orig[i] = (pt_net[0] * scale_net_to_orig_x, pt_net[1] * scale_net_to_orig_y)
        else:
            # Scale initial predictions if not refining
            points_to_use_orig = []
            for pt_net in points_pred_resized:
                 if pt_net[0] is not None:
                     points_to_use_orig.append((pt_net[0] * scale_net_to_orig_x, pt_net[1] * scale_net_to_orig_y))
                 else:
                     points_to_use_orig.append(None)

        # Optional: Apply Homography
        final_points = points_to_use_orig
        if args.use_homography:
            matrix_trans = get_trans_matrix(points_to_use_orig)
            if matrix_trans is not None:
                refer_kps = court_ref.get_keypoints_array().reshape((-1, 1, 2))
                transformed_points = cv2.perspectiveTransform(refer_kps, matrix_trans)
                if transformed_points is not None:
                    final_points = [tuple(map(float, np.squeeze(p))) if p is not None else None for p in transformed_points]

        # Draw Results on the original frame
        frame_out = frame_orig.copy()
        for j, point in enumerate(final_points):
            if point is not None and point[0] is not None:
                point_int = tuple(map(int, point))
                if 0 <= point_int[0] < orig_w and 0 <= point_int[1] < orig_h:
                    cv2.circle(frame_out, point_int, radius=DRAW_RADIUS, color=DRAW_COLOR, thickness=DRAW_THICKNESS)

        frames_out.append(frame_out)

    # Write Output Video
    write_video(frames_out, fps, args.output_path)

    print("Video processing finished.")

# --- END OF FILE infer_in_video.py ---