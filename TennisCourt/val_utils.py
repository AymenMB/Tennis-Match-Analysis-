# --- START OF FILE base_validator.py ---

import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm # Added TQDM
from utils import is_point_in_image
from scipy.spatial import distance
from postprocess import postprocess # Use the updated postprocess function
from court_keypoint_net import CourtKeypointNet
# from dataset import courtDataset # Not strictly needed here
# import argparse # Not needed here
# import torch.nn as nn # Not needed here

# Configuration
MAX_DIST_THRESHOLD = 7 # Pixel distance threshold for TP
POSTPROCESS_SCALE_FACTOR = 2 # Scale factor from heatmap coords to original image coords (adjust if needed)
HEATMAP_THRESHOLD = 0.1 # Threshold for considering a heatmap peak valid (adjust based on model output)
NUM_KEYPOINTS_TO_EVAL = 14 # Evaluate the original 14 keypoints

def val(model, val_loader, criterion, device, epoch):
    model.eval() # Ensure model is in evaluation mode
    losses = []
    tp, fp, fn, tn = 0, 0, 0, 0
    total_valid_gt_points = 0 # Count points that should be visible
    total_points_evaluated = 0

    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch} [Val]")

    with torch.no_grad(): # Disable gradient calculations for validation
        for iter_id, batch in progress_bar:
            # Unpack batch and move to device
            inputs = batch[0].to(device, non_blocking=True)
            gt_hm = batch[1].to(device, non_blocking=True)
            gt_kps_orig = batch[2] # Ground truth keypoints in original image resolution (NumPy array)
            # img_names = batch[3] # Not used directly here

            batch_size = inputs.shape[0]

            # Forward pass
            out_logits = model(inputs)
            pred_hm_sig = torch.sigmoid(out_logits) # Apply sigmoid to get "probability" heatmaps

            # Calculate loss (optional, but good for monitoring)
            loss = criterion(pred_hm_sig, gt_hm)
            losses.append(loss.item())

            # Process predictions and compare with ground truth
            pred_hm_np = pred_hm_sig.detach().cpu().numpy() # Move heatmaps to CPU as NumPy arrays

            for bs in range(batch_size):
                for kps_idx in range(NUM_KEYPOINTS_TO_EVAL): # Iterate through the 14 keypoints
                    heatmap = pred_hm_np[bs][kps_idx] # Get heatmap for this keypoint

                    # Use updated postprocess function to find peak
                    x_pred, y_pred = postprocess(heatmap,
                                                 scale_factor=POSTPROCESS_SCALE_FACTOR,
                                                 threshold=HEATMAP_THRESHOLD)

                    # Get ground truth coordinates for this keypoint
                    # Handle potential invalid coordinates (e.g., -1 from dataset loading)
                    x_gt = gt_kps_orig[bs, kps_idx, 0].item()
                    y_gt = gt_kps_orig[bs, kps_idx, 1].item()

                    # Determine if GT point is valid (should be visible in original image)
                    gt_is_valid = is_point_in_image(x_gt, y_gt) # Assumes input_width/height defaults match original

                    # Determine if prediction is valid (found peak above threshold)
                    pred_is_valid = (x_pred is not None) and (y_pred is not None)

                    total_points_evaluated += 1
                    if gt_is_valid:
                        total_valid_gt_points += 1

                    # --- Calculate TP, FP, FN, TN based on validity and distance ---
                    if pred_is_valid and gt_is_valid:
                        # Both prediction and GT are valid: Check distance
                        dst = distance.euclidean((x_pred, y_pred), (x_gt, y_gt))
                        if dst < MAX_DIST_THRESHOLD:
                            tp += 1
                        else:
                            fp += 1 # Predicted, GT valid, but too far -> False Positive
                    elif pred_is_valid and not gt_is_valid:
                        # Prediction valid, GT invalid -> False Positive
                        fp += 1
                    elif not pred_is_valid and gt_is_valid:
                        # Prediction invalid, GT valid -> False Negative
                        fn += 1
                    elif not pred_is_valid and not gt_is_valid:
                        # Both invalid -> True Negative
                        tn += 1

            # Update progress bar with running metrics (optional)
            # eps = 1e-15
            # current_precision = round(tp / (tp + fp + eps), 5)
            # current_accuracy = round((tp + tn) / (tp + tn + fp + fn + eps), 5)
            # progress_bar.set_postfix({'loss': f'{loss.item():.5f}', 'acc': f'{current_accuracy:.3f}', 'prec': f'{current_precision:.3f}'})

    progress_bar.close()

    # Calculate final metrics for the epoch
    eps = 1e-15
    precision = round(tp / (tp + fp + eps), 5)
    recall = round(tp / (tp + fn + eps), 5) # Calculate recall as well
    # Accuracy calculation depends on definition. Standard classification accuracy:
    accuracy = round((tp + tn) / (tp + tn + fp + fn + eps), 5)
    # Or sometimes accuracy is defined as PCK (Percentage of Correct Keypoints) considering only valid GT points:
    # pck_accuracy = round(tp / (total_valid_gt_points + eps), 5) if total_valid_gt_points > 0 else 0.0

    avg_loss = np.mean(losses) if losses else 0.0

    print(f'Validation Summary: Avg Loss: {avg_loss:.5f}, TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}')
    print(f'Metrics: Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}')
    # print(f'PCK Accuracy (based on valid GT): {pck_accuracy}')

    # Return metrics needed for main training loop (loss and primary metric like accuracy)
    return avg_loss, tp, fp, fn, tn, precision, accuracy


# Example usage (commented out, as this is called from main.py)
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--batch_size', type=int, default=2, help='batch size')
#     parser.add_argument('--model_path', type=str, help='path to pretrained model')
#     args = parser.parse_args()
#
#     val_dataset = courtDataset('val') # Needs dataset definition
#     val_loader = torch.utils.data.DataLoader(
#         val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True
#     )
#
#     model = CourtKeypointNet(out_channels=15) # Needs model definition
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model.load_state_dict(torch.load(args.model_path, map_location=device))
#     model = model.to(device)
#     criterion = nn.MSELoss() # Needs loss definition
#
#     val_loss, tp, fp, fn, tn, precision, accuracy = val(model, val_loader, criterion, device, -1) # Epoch -1 for standalone run

# --- END OF FILE base_validator.py ---