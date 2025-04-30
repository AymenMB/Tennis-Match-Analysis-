# --- START OF FILE infer_in_image.py ---

import cv2
import numpy as np
import torch
from tennistrack import BallTrackerNet
import torch.nn.functional as F
from postprocess import postprocess, refine_kps # Use updated postprocess
from homography import get_trans_matrix, court_ref # Use updated homography, need court_ref for refer_kps
import argparse
import os

# Configuration
NETWORK_INPUT_WIDTH = 640
NETWORK_INPUT_HEIGHT = 360
POSTPROCESS_SCALE_FACTOR = 2 # Scale from heatmap (360x640) to original image coords (assuming input is 720x1280, adjust if needed)
# However, the script reads original image, resizes to 640x360 for network, then draws on original.
# Need to scale predictions back to original image size.
# Let's calculate scale dynamically.
HEATMAP_THRESHOLD = 0.1 # Threshold for postprocess
REFINE_KPS_EXCLUDE = [8, 9, 12, 13] # Indices of points not suitable for refinement (e.g., center line points without clear intersections)
DRAW_RADIUS = 5
DRAW_THICKNESS = -1 # Filled circle
DRAW_COLOR = (0, 0, 255) # Red BGR

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='path to trained model (.pt file)')
    parser.add_argument('--input_path', type=str, required=True, help='path to input image')
    parser.add_argument('--output_path', type=str, required=True, help='path to save output image')
    parser.add_argument('--use_refine_kps', action='store_true', help='whether to use refine kps postprocessing')
    parser.add_argument('--use_homography', action='store_true', help='whether to use homography postprocessing')
    parser.add_argument('--draw_refined', action='store_true', help='draw refined points in a different color')
    args = parser.parse_args()

    # Load Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("Loading model...")
    # Ensure the model architecture matches the saved weights
    model = BallTrackerNet(out_channels=15) # 14 kps + 1 center
    # Load state dict, handling DataParallel wrapper if necessary
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint) # Handle checkpoints saved with extra info
    # Remove `module.` prefix if saved using DataParallel and loading without it
    if not isinstance(model, torch.nn.DataParallel) and all(key.startswith('module.') for key in state_dict.keys()):
         state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    elif isinstance(model, torch.nn.DataParallel) and not all(key.startswith('module.') for key in state_dict.keys()):
         state_dict = {'module.' + k: v for k, v in state_dict.items()} # Add prefix if needed
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval() # Set to evaluation mode
    print("Model loaded.")

    # Load Image
    print(f"Loading image: {args.input_path}")
    image_orig = cv2.imread(args.input_path)
    if image_orig is None:
        raise FileNotFoundError(f"Could not read image file: {args.input_path}")
    orig_h, orig_w = image_orig.shape[:2]

    # Preprocess Image for Network
    # Resize to network input size
    img_resized = cv2.resize(image_orig, (NETWORK_INPUT_WIDTH, NETWORK_INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
    # Convert BGR to RGB, normalize, and convert to tensor (similar to dataset)
    inp = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    inp = (inp.astype(np.float32) / 255.)
    # Use standard imagenet normalization (should match training)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    inp = (inp - mean) / std
    inp = torch.from_numpy(np.rollaxis(inp, 2, 0)).unsqueeze(0) # HWC -> CHW -> NCHW

    # Inference
    print("Running inference...")
    with torch.no_grad():
        out_logits = model(inp.to(device))
    pred_hm_sig = torch.sigmoid(out_logits) # Apply sigmoid
    pred_hm_np = pred_hm_sig.squeeze(0).cpu().numpy() # Remove batch dim, move to CPU/NumPy -> (15, H, W)
    print("Inference complete.")

    # Post-processing: Extract points from heatmaps
    points_pred_resized = [] # Points relative to resized network input (640x360)
    # Scale factor from heatmap coords to network input coords (should be 1 if heatmap size = network size)
    heatmap_h, heatmap_w = pred_hm_np.shape[1], pred_hm_np.shape[2]
    scale_hm_to_net_x = NETWORK_INPUT_WIDTH / heatmap_w
    scale_hm_to_net_y = NETWORK_INPUT_HEIGHT / heatmap_h

    for kps_num in range(14): # Process the 14 keypoints
        heatmap = pred_hm_np[kps_num]
        # Find peak in heatmap coords, scale to network input coords
        x_net, y_net = postprocess(heatmap,
                                   scale_factor=(scale_hm_to_net_x, scale_hm_to_net_y), # Pass tuple for different x/y scales if needed
                                   threshold=HEATMAP_THRESHOLD)
        points_pred_resized.append((x_net, y_net)) # Store points relative to 640x360

    # --- Optional: Refine Keypoints ---
    points_refined_orig = [None] * 14 # Store refined points scaled to original image size
    if args.use_refine_kps:
        print("Applying keypoint refinement...")
        # Refinement needs the original image resolution
        scale_net_to_orig_x = orig_w / NETWORK_INPUT_WIDTH
        scale_net_to_orig_y = orig_h / NETWORK_INPUT_HEIGHT

        for i, pt_net in enumerate(points_pred_resized):
            if pt_net[0] is not None and i not in REFINE_KPS_EXCLUDE:
                # Scale point to original image resolution for refinement function
                x_orig_est = pt_net[0] * scale_net_to_orig_x
                y_orig_est = pt_net[1] * scale_net_to_orig_y

                # Refine expects (y, x) order for image indexing
                y_refined, x_refined = refine_kps(image_orig, int(y_orig_est), int(x_orig_est))
                points_refined_orig[i] = (x_refined, y_refined) # Store back in (x, y) format
            elif pt_net[0] is not None:
                 # If point is valid but excluded from refinement, scale it to original size
                 points_refined_orig[i] = (pt_net[0] * scale_net_to_orig_x, pt_net[1] * scale_net_to_orig_y)
            # else: point remains None
        points_to_use = points_refined_orig # Use refined points (at original scale) for homography/drawing
        print("Refinement done.")
    else:
        # If not refining, scale the initial predictions to the original image size
        scale_net_to_orig_x = orig_w / NETWORK_INPUT_WIDTH
        scale_net_to_orig_y = orig_h / NETWORK_INPUT_HEIGHT
        points_to_use = []
        for pt_net in points_pred_resized:
             if pt_net[0] is not None:
                 points_to_use.append((pt_net[0] * scale_net_to_orig_x, pt_net[1] * scale_net_to_orig_y))
             else:
                 points_to_use.append(None)


    # --- Optional: Apply Homography ---
    final_points = points_to_use # Start with points from previous step (at original scale)
    if args.use_homography:
        print("Applying homography...")
        # Homography requires points in (x, y) format
        matrix_trans = get_trans_matrix(points_to_use) # Pass points at original scale
        if matrix_trans is not None:
            print("Homography matrix found. Transforming reference points...")
            # Get reference keypoints (already scaled to its own reference image size)
            refer_kps = court_ref.get_keypoints_array().reshape((-1, 1, 2)) # Shape (14, 1, 2)
            # Transform reference points to the *current image's* coordinate system
            transformed_points = cv2.perspectiveTransform(refer_kps, matrix_trans)
            if transformed_points is not None:
                 # Squeeze and convert to list of tuples, handle potential None on failure
                 final_points = [tuple(map(float, np.squeeze(p))) if p is not None else None for p in transformed_points]
                 print("Homography applied.")
            else:
                 print("Warning: perspectiveTransform failed.")
        else:
            print("Homography matrix not found or failed. Using previous points.")


    # --- Draw Results on Original Image ---
    image_out = image_orig.copy()
    print("Drawing final points...")

    # Draw initial/refined points if requested and different from final points
    if args.draw_refined and args.use_homography and args.use_refine_kps:
         for i, pt in enumerate(points_to_use): # points_to_use holds refined points at original scale
              if pt is not None and pt[0] is not None:
                   pt_int = tuple(map(int, pt))
                   if 0 <= pt_int[0] < orig_w and 0 <= pt_int[1] < orig_h:
                       cv2.circle(image_out, pt_int, radius=DRAW_RADIUS, color=(255, 150, 0), thickness=DRAW_THICKNESS-1) # Blueish

    # Draw final points (potentially after homography)
    for j, point in enumerate(final_points):
        if point is not None and point[0] is not None:
            # Ensure points are tuples of integers for drawing
            point_int = tuple(map(int, point))
            # Check if point is within image bounds before drawing
            if 0 <= point_int[0] < orig_w and 0 <= point_int[1] < orig_h:
                cv2.circle(image_out, point_int, radius=DRAW_RADIUS, color=DRAW_COLOR, thickness=DRAW_THICKNESS)
                # Optionally draw index number
                cv2.putText(image_out, str(j), (point_int[0] + 5, point_int[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Save Output Image
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(args.output_path, image_out)
    print(f"Output image saved to: {args.output_path}")

# --- END OF FILE infer_in_image.py ---