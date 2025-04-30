from ball_model import BallTrackerNet # Uses updated model
import torch
import cv2
from utils_general import postprocess
from tqdm import tqdm
import numpy as np
import argparse
from itertools import groupby
from scipy.spatial import distance
import os
import time # For timing

def read_video(path_video):
    cap = cv2.VideoCapture(path_video)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {path_video}")
        return [], 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    print(f"Read {len(frames)} frames at {fps} FPS from {path_video}")
    return frames, fps

def infer_model(frames, model, device, input_height=360, input_width=640):
    # Get original dimensions for scaling output coords later
    if not frames: return [], []
    orig_height, orig_width = frames[0].shape[:2]
    scale_x = orig_width / input_width
    scale_y = orig_height / input_height

    dists = [-1.0]*2 # Store distances between consecutive *scaled* points
    # Store coordinates scaled back to original video dimensions
    ball_track_scaled = [(None, None)]*2
    model.to(device)
    model.eval()

    inference_times = []

    # Need at least 3 frames (0, 1, 2) to start prediction for frame 2
    for num in tqdm(range(2, len(frames)), desc="Inferring model"):
        start_time = time.time()

        # Prepare input frames
        img = cv2.resize(frames[num], (input_width, input_height))
        img_prev = cv2.resize(frames[num-1], (input_width, input_height))
        img_preprev = cv2.resize(frames[num-2], (input_width, input_height))

        # Stack and normalize
        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32) / 255.0
        imgs = np.rollaxis(imgs, 2, 0) # HWC to CHW (9, H, W)
        inp = np.expand_dims(imgs, axis=0) # Add batch dimension (1, 9, H, W)

        with torch.no_grad():
            # Model output is heatmap (1, 1, H, W)
            out_heatmap = model(torch.from_numpy(inp).float().to(device))

        # Process output heatmap
        # Remove batch and channel dim: (1, 1, H, W) -> (H, W)
        heatmap_np = out_heatmap.squeeze(0).squeeze(0).cpu().numpy()

        # Find coordinates (x, y) in the resized space (input_width, input_height)
        x_pred_resized, y_pred_resized = postprocess(heatmap_np, threshold=0.5) # Use updated postprocess

        end_time = time.time()
        inference_times.append(end_time - start_time)

        # Scale coordinates back to original video dimensions
        x_pred_scaled, y_pred_scaled = None, None
        if x_pred_resized is not None:
            x_pred_scaled = x_pred_resized * scale_x
            y_pred_scaled = y_pred_resized * scale_y

        ball_track_scaled.append((x_pred_scaled, y_pred_scaled))

        # Calculate distance between consecutive *scaled* points
        dist = -1.0
        if ball_track_scaled[-1][0] is not None and ball_track_scaled[-2][0] is not None:
            dist = distance.euclidean(ball_track_scaled[-1], ball_track_scaled[-2])
        dists.append(dist)

    avg_inf_time = np.mean(inference_times) if inference_times else 0
    print(f"Average inference time per frame: {avg_inf_time:.4f} seconds")
    # Return coordinates scaled to original video size
    return ball_track_scaled, dists

# --- Functions remove_outliers, split_track, interpolation remain the same ---
# (Make sure they handle the (None, None) format correctly, which they should)
def remove_outliers(ball_track, dists, max_dist = 100):
    """ Remove outliers from model prediction based on distance in original video scale.
    :params
        ball_track: list of detected ball points (scaled to original video)
        dists: list of euclidean distances between two neighbouring ball points (scaled)
        max_dist: maximum distance (pixels) between two neighbouring ball points in original video
    :return
        ball_track: list of ball points with outliers removed
    """
    outliers = [i for i, d in enumerate(dists) if d > max_dist and i > 0] # Find indices where dist > max_dist
    num_removed = 0
    # Iterate backwards to avoid index issues when removing
    for i in sorted(outliers, reverse=True):
        # Simple outlier removal: If a point is too far from the previous one, remove it.
        # More complex logic could consider the next point too.
        # Check if the point itself exists
         if ball_track[i][0] is not None:
            # Basic check: if distance to previous is large, remove current point
            # We could also check distance to next point if available
            next_dist = dists[i+1] if i + 1 < len(dists) else -1

            # If both prev and next dists are large, definitely remove
            if (dists[i] > max_dist) and (next_dist > max_dist or next_dist == -1):
                 ball_track[i] = (None, None)
                 num_removed +=1
            # If only prev dist is large, maybe interpolate later? For now, remove.
            elif dists[i] > max_dist:
                 ball_track[i] = (None, None)
                 num_removed +=1

    print(f"Removed {num_removed} potential outliers based on max_dist={max_dist}.")
    return ball_track

def split_track(ball_track, max_gap=4, max_dist_gap=80, min_track=5):
    """ Split ball track into several subtracks for interpolation.
        Operates on the scaled ball_track.
    """
    list_det = [0 if x[0] is not None else 1 for x in ball_track] # 0 = detected, 1 = None
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]

    cursor = 0
    min_value = 0
    result = [] # List to store [start, end] indices of subtracks
    for i, (k, l) in enumerate(groups):
        # k=1 means a gap of None values of length l
        if k == 1 and l > 0 and i > 0 and i < len(groups) - 1:
            # Check if indices are valid before accessing ball_track
            prev_idx = cursor - 1
            next_idx = cursor + l
            if prev_idx >= 0 and next_idx < len(ball_track) and \
               ball_track[prev_idx][0] is not None and ball_track[next_idx][0] is not None:
                # Calculate distance between points bordering the gap
                dist = distance.euclidean(ball_track[prev_idx], ball_track[next_idx])
                # If gap is too long OR points are too far apart for the gap length
                if l >= max_gap or (dist / l > max_dist_gap):
                    # Finalize the previous subtrack if it's long enough
                    if cursor - min_value >= min_track:
                        result.append([min_value, cursor]) # Add [start, end) of the subtrack
                    # Start the next subtrack after the gap
                    min_value = next_idx # Start next track at the first point after the gap
            else:
                # If points bordering the gap don't exist, end the current track here
                 if cursor - min_value >= min_track:
                     result.append([min_value, cursor])
                 min_value = cursor + l # Skip the gap

        cursor += l # Move cursor past the current group

    # Add the last subtrack if it's long enough
    if len(ball_track) - min_value >= min_track:
        result.append([min_value, len(ball_track)])

    print(f"Split track into {len(result)} subtracks for interpolation.")
    return result

def interpolation(coords):
    """ Interpolate None values within a subtrack using linear interpolation.
    :params
        coords: list of (x, y) coordinates for one subtrack, may contain (None, None)
    :return
        track: list of interpolated coordinates (float values)
    """
    # Separate x and y, replacing None with np.nan
    x = np.array([c[0] if c[0] is not None else np.nan for c in coords], dtype=float)
    y = np.array([c[1] if c[1] is not None else np.nan for c in coords], dtype=float)

    # Helper to find indices of NaNs and non-NaNs
    def nan_helper(arr):
        is_nan = np.isnan(arr)
        indices = lambda z: z.nonzero()[0]
        return is_nan, indices

    # Interpolate x
    nans_x, x_indices = nan_helper(x)
    if np.any(nans_x) and not np.all(nans_x): # Check if there are NaNs to interpolate and some non-NaNs exist
        x[nans_x] = np.interp(x_indices(nans_x), x_indices(~nans_x), x[~nans_x])

    # Interpolate y
    nans_y, y_indices = nan_helper(y)
    if np.any(nans_y) and not np.all(nans_y):
        y[nans_y] = np.interp(y_indices(nans_y), y_indices(~nans_y), y[~nans_y])

    # Combine back into list of tuples
    track = list(zip(x, y))
    return track


def write_track(frames, ball_track, path_output_video, fps, trace=7):
    """ Write .avi file with detected ball tracks overlaid on original frames.
    :params
        frames: list of original video frames
        ball_track: list of ball coordinates (scaled to original video size)
        path_output_video: path to output video (.avi)
        fps: frames per second
        trace: number of frames for the trajectory trace effect
    """
    if not frames:
        print("Error: No frames to write.")
        return
    height, width = frames[0].shape[:2]
    # Ensure output directory exists
    output_dir = os.path.dirname(path_output_video)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Use XVID codec, generally more compatible than DIVX
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path_output_video, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not open video writer for path: {path_output_video}")
        print("Ensure the codec 'XVID' is supported. You might need to install codec packs.")
        return

    for num in tqdm(range(len(frames)), desc="Writing video"):
        frame = frames[num].copy() # Work on a copy to avoid modifying original frames list
        # Draw the trajectory trace
        for i in range(trace):
            trace_idx = num - i
            if trace_idx >= 0 and trace_idx < len(ball_track):
                coords = ball_track[trace_idx]
                if coords[0] is not None and coords[1] is not None:
                    # Ensure coordinates are integers for drawing
                    x = int(round(coords[0]))
                    y = int(round(coords[1]))
                    # Check if coords are within frame bounds
                    if 0 <= x < width and 0 <= y < height:
                        # Draw circle: smaller radius, decreasing intensity for older points
                        radius = 5 # Fixed small radius
                        color_intensity = max(0, 255 - i * (255 // trace)) # Fades out
                        # Draw with BGR color (Red)
                        cv2.circle(frame, (x, y), radius=radius, color=(0, 0, color_intensity), thickness=-1) # Filled circle
                else:
                    # If a point in the trace is None, stop drawing the trace for this frame
                    break
        out.write(frame)

    out.release()
    print(f"Output video saved to: {path_output_video}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='path to trained model .pt file (expecting 1 output channel)')
    parser.add_argument('--video_path', type=str, required=True, help='path to input video (.mp4, .avi, etc.)')
    parser.add_argument('--video_out_path', type=str, required=True, help='path to output video (.avi)')
    parser.add_argument('--input_height', type=int, default=360, help='Model input height')
    parser.add_argument('--input_width', type=int, default=640, help='Model input width')
    parser.add_argument('--extrapolation', action='store_true', help='whether to use ball track interpolation for gaps')
    parser.add_argument('--outlier_max_dist', type=float, default=100.0, help='Maximum distance (pixels) between consecutive frames to not be considered an outlier')
    parser.add_argument('--interp_max_gap', type=int, default=5, help='Maximum number of consecutive missing frames to interpolate')
    parser.add_argument('--interp_max_dist', type=float, default=80.0, help='Maximum distance per frame allowed within a gap for interpolation')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model architecture (ensure out_channels=1)
    model = BallTrackerNet(out_channels=1)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model weights from: {args.model_path}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Ensure the model path is correct and the saved state dict matches the BallTrackerNet(out_channels=1) architecture.")
        exit()
    model = model.to(device)
    model.eval()

    print(f"Reading video: {args.video_path}")
    frames, fps = read_video(args.video_path)

    if frames:
        print(f"Running inference...")
        ball_track, dists = infer_model(frames, model, device, height=args.input_height, width=args.input_width)

        print(f"Removing outliers (max_dist={args.outlier_max_dist})...")
        ball_track = remove_outliers(ball_track, dists, max_dist=args.outlier_max_dist)

        if args.extrapolation:
            print(f"Performing interpolation (max_gap={args.interp_max_gap}, max_dist_gap={args.interp_max_dist})...")
            # Split the track into subtracks based on gaps/distances
            subtracks_indices = split_track(ball_track, max_gap=args.interp_max_gap, max_dist_gap=args.interp_max_dist)
            # Interpolate within each valid subtrack
            for start, end in subtracks_indices:
                 if end - start > 1: # Need at least 2 points to interpolate
                    interpolated_subtrack = interpolation(ball_track[start:end])
                    ball_track[start:end] = interpolated_subtrack
            print("Interpolation finished.")

        print(f"Writing output video to: {args.video_out_path}")
        write_track(frames, ball_track, args.video_out_path, fps)
    else:
        print("Exiting due to error reading video.")