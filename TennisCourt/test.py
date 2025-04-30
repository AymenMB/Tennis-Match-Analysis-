# --- START OF FILE test.py ---

import numpy as np
import cv2
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.spatial import distance

from utils import is_point_in_image
from postprocess import postprocess, refine_kps # Use updated versions
from homography import get_trans_matrix, court_ref # Use updated versions
from dataset import courtDataset # Need dataset for loading test data
from tennistrack import BallTrackerNet # Need model definition
import argparse

# Configuration
MAX_DIST_THRESHOLD = 7 # Pixel distance threshold for TP/FP evaluation
POSTPROCESS_SCALE_FACTOR = 2 # Scale from heatmap coords to original image coords
HEATMAP_THRESHOLD = 0.1 # Threshold for considering a heatmap peak valid
NUM_KEYPOINTS_TO_EVAL = 14 # Number of keypoints to evaluate
REFINE_KPS_EXCLUDE = [8, 9, 12, 13] # Indices to exclude from refinement

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for evaluation') # Increase default slightly
    parser.add_argument('--model_path', type=str, required=True, help='path to trained model (.pt file)')
    parser.add_argument('--use_refine_kps', action='store_true', help='whether to use refine kps postprocessing during eval')
    parser.add_argument('--use_homography', action='store_true', help='whether to use homography postprocessing during eval')
    parser.add_argument('--num_workers', type=int, default=4, help='number of dataloader workers')
    parser.add_argument('--dataset_mode', type=str, default='val', choices=['train', 'val', 'test'], help='Which dataset split to test on')
    args = parser.parse_args()

    # Setup Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load Dataset (Using 'val' split as per original code, but configurable)
    print(f"Loading dataset (mode: {args.dataset_mode})...")
    eval_dataset = courtDataset(args.dataset_mode)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False, # No shuffling during evaluation
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    print("Dataset loaded.")

    # Load Model (same logic as inference scripts)
    print("Loading model...")
    model = BallTrackerNet(out_channels=15)
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if not isinstance(model, torch.nn.DataParallel) and all(key.startswith('module.') for key in state_dict.keys()):
         state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    elif isinstance(model, torch.nn.DataParallel) and not all(key.startswith('module.') for key in state_dict.keys()):
         state_dict = {'module.' + k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval() # Set to evaluation mode
    print("Model loaded.")

    # Evaluation Loop
    tp, tn, fp, fn = 0, 0, 0, 0
    all_distances = [] # Store distances for median/mean calculation
    total_valid_gt_points = 0
    total_points_evaluated = 0

    print("Starting evaluation...")
    progress_bar = tqdm(enumerate(eval_loader), total=len(eval_loader), desc=f"Testing ({args.dataset_mode})")

    with torch.no_grad():
        for iter_id, batch in progress_bar:
            # Unpack batch
            inputs = batch[0].to(device, non_blocking=True)
            # gt_hm = batch[1] # Ground truth heatmaps not directly used for metric calculation here
            gt_kps_orig = batch[2].numpy() # Ground truth keypoints in original image resolution (N, 14, 2)
            img_ids = batch[3] # Image IDs/names

            batch_size = inputs.shape[0]

            # Forward pass
            out_logits = model(inputs)
            pred_hm_sig = torch.sigmoid(out_logits)
            pred_hm_np = pred_hm_sig.cpu().numpy() # (N, 15, H, W)

            # Process each item in the batch
            for bs in range(batch_size):
                # Load original image if needed for refinement
                img_orig = None
                if args.use_refine_kps:
                    img_path = os.path.join(eval_dataset.path_images, img_ids[bs] + '.png')
                    if os.path.exists(img_path):
                         img_orig = cv2.imread(img_path)
                    else:
                         print(f"Warning: Could not load image {img_path} for refinement.")

                # Get predictions from heatmaps (relative to network input size)
                points_pred_resized = []
                heatmap_h, heatmap_w = pred_hm_np.shape[2], pred_hm_np.shape[3]
                # Scale factor from heatmap coords to network input coords (should be 1 if heatmap size = network size)
                # Let's assume they are the same based on TennisTrack architecture: 360x640
                scale_hm_to_net_x = eval_dataset.input_width / heatmap_w # Scale to original image size directly? No, scale heatmap->net input first
                scale_hm_to_net_y = eval_dataset.input_height / heatmap_h # Need dataset input H/W
                # Postprocess expects scale factor to original image, let's recalculate
                # Need original image size for scaling. Get from gt_kps_orig or load image? Assume 1280x720 for now.
                orig_w, orig_h = 1280, 720 # TODO: Get this dynamically if possible
                scale_x = orig_w / heatmap_w
                scale_y = orig_h / heatmap_h

                for kps_num in range(NUM_KEYPOINTS_TO_EVAL):
                    heatmap = pred_hm_np[bs, kps_num]
                    # Use postprocess to get coords scaled to *original* image size
                    x_pred_orig, y_pred_orig = postprocess(heatmap,
                                                           scale_factor=(scale_x, scale_y),
                                                           threshold=HEATMAP_THRESHOLD)
                    points_pred_resized.append((x_pred_orig, y_pred_orig)) # Store points at original scale

                # Apply refinement if enabled
                points_to_use = points_pred_resized # Start with initial predictions (at original scale)
                if args.use_refine_kps and img_orig is not None:
                    refined_points_orig = [None] * NUM_KEYPOINTS_TO_EVAL
                    for i, pt_orig in enumerate(points_pred_resized):
                         if pt_orig is not None and pt_orig[0] is not None and i not in REFINE_KPS_EXCLUDE:
                              # Refine kps expects (y, x) order
                              y_refined, x_refined = refine_kps(img_orig, int(pt_orig[1]), int(pt_orig[0]))
                              refined_points_orig[i] = (x_refined, y_refined) # Store back as (x, y)
                         else:
                              refined_points_orig[i] = pt_orig # Keep original if not refined or invalid
                    points_to_use = refined_points_orig

                # Apply homography if enabled
                final_points = points_to_use
                if args.use_homography:
                    matrix_trans = get_trans_matrix(points_to_use) # Pass points at original scale
                    if matrix_trans is not None:
                        refer_kps = court_ref.get_keypoints_array().reshape((-1, 1, 2))
                        transformed_points = cv2.perspectiveTransform(refer_kps, matrix_trans)
                        if transformed_points is not None:
                            final_points = [tuple(map(float, np.squeeze(p))) if p is not None else None for p in transformed_points]

                # Compare final points with ground truth for this image
                for i, point_pred in enumerate(final_points):
                    x_gt = gt_kps_orig[bs, i, 0]
                    y_gt = gt_kps_orig[bs, i, 1]

                    # Check if GT point is valid
                    gt_is_valid = is_point_in_image(x_gt, y_gt, orig_w, orig_h)

                    # Check if prediction is valid
                    pred_is_valid = point_pred is not None and point_pred[0] is not None and point_pred[1] is not None
                    x_pred, y_pred = (point_pred[0], point_pred[1]) if pred_is_valid else (None, None)

                    total_points_evaluated += 1
                    if gt_is_valid:
                         total_valid_gt_points += 1

                    # --- Calculate TP, FP, FN, TN ---
                    if pred_is_valid and gt_is_valid:
                        dst = distance.euclidean((x_pred, y_pred), (x_gt, y_gt))
                        all_distances.append(dst) # Record distance for valid points
                        if dst < MAX_DIST_THRESHOLD:
                            tp += 1
                        else:
                            fp += 1
                    elif pred_is_valid and not gt_is_valid:
                        fp += 1
                    elif not pred_is_valid and gt_is_valid:
                        fn += 1
                    elif not pred_is_valid and not gt_is_valid:
                        tn += 1

            # Update progress bar periodically (optional)
            # eps = 1e-15
            # current_precision = round(tp/(tp+fp+eps), 5)
            # current_accuracy = round((tp+tn)/(tp+tn+fp+fn+eps), 5)
            # progress_bar.set_postfix({'acc': f'{current_accuracy:.3f}', 'prec': f'{current_precision:.3f}'})

    progress_bar.close()

    # Calculate final metrics
    eps = 1e-15
    precision = round(tp / (tp + fp + eps), 5)
    recall = round(tp / (tp + fn + eps), 5)
    accuracy = round((tp + tn) / (tp + tn + fp + fn + eps), 5)
    f1_score = 2 * (precision * recall) / (precision + recall + eps)

    mean_dist = np.mean(all_distances) if all_distances else -1
    median_dist = np.median(all_distances) if all_distances else -1

    print("\n--- Evaluation Results ---")
    print(f"Dataset Split: {args.dataset_mode}")
    print(f"Options: RefineKPS={args.use_refine_kps}, Homography={args.use_homography}")
    print(f"Total Points Evaluated: {total_points_evaluated}")
    print(f"Valid Ground Truth Points: {total_valid_gt_points}")
    print("-" * 25)
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print("-" * 25)
    print(f"Precision: {precision:.5f}")
    print(f"Recall:    {recall:.5f}")
    print(f"Accuracy:  {accuracy:.5f}")
    print(f"F1 Score:  {f1_score:.5f}")
    print("-" * 25)
    print(f"Mean Distance (for TP+FP where GT valid): {mean_dist:.3f} pixels")
    print(f"Median Distance (for TP+FP where GT valid): {median_dist:.3f} pixels")
    print("--- End of Results ---")

# --- END OF FILE test.py ---