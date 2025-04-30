import torch
import time
import numpy as np
import torch.nn as nn
import cv2
from scipy.spatial import distance
from tqdm import tqdm

def train(model, train_loader, optimizer, device, epoch, max_iters=200):
    start_time = time.time()
    losses = []
    # Use MSELoss for heatmap regression
    criterion = nn.MSELoss() # <<<< CHANGED: MSELoss
    model.train()

    train_iterator = tqdm(enumerate(train_loader), total=min(max_iters, len(train_loader)), desc=f"Training Epoch {epoch}")
    for iter_id, batch in train_iterator:
        if iter_id >= max_iters:
            break

        inputs = batch[0].float().to(device)
        # Ground truth heatmap (B, 1, H, W), float, normalized [0, 1]
        gt_heatmap = batch[1].float().to(device) # <<<< CHANGED: Load as float

        optimizer.zero_grad()
        # Model output heatmap (B, 1, H, W)
        out_heatmap = model(inputs)

        # Calculate loss between predicted and GT heatmaps
        loss = criterion(out_heatmap, gt_heatmap) # <<<< CHANGED: MSE loss calculation

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # Update tqdm description
        train_iterator.set_description(
            f'Train Epoch {epoch} | Iter [{iter_id+1}/{max_iters}] | Loss: {loss.item():.6f}'
        )

    duration = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f'train | epoch = {epoch}, mean_loss = {np.mean(losses):.6f}, time = {duration}')
    return np.mean(losses)


def validate(model, val_loader, device, epoch, min_dist=10): # Increased min_dist slightly
    losses = []
    tp = [0, 0, 0, 0] # True Positives per Visibility Class (index 0 unused)
    fp = [0, 0, 0, 0] # False Positives
    tn = [0, 0, 0, 0] # True Negatives (ball not present and not detected)
    fn = [0, 0, 0, 0] # False Negatives

    # Use MSELoss for validation loss consistency
    criterion = nn.MSELoss() # <<<< CHANGED: MSELoss
    model.eval()
    val_iterator = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Validating Epoch {epoch}")

    with torch.no_grad():
        for iter_id, batch in val_iterator:
            inputs = batch[0].float().to(device)
            # Ground truth heatmap (B, 1, H, W), float
            gt_heatmap = batch[1].float().to(device) # <<<< CHANGED: Load as float

            # Predicted heatmap (B, 1, H, W)
            out_heatmap = model(inputs)
            loss = criterion(out_heatmap, gt_heatmap)
            losses.append(loss.item())

            # --- Metrics Calculation ---
            # Get heatmaps on CPU as numpy arrays (B, 1, H, W) -> (B, H, W)
            pred_heatmaps_np = out_heatmap.squeeze(1).cpu().numpy() # Remove channel dim
            batch_size = pred_heatmaps_np.shape[0]
            h, w = val_loader.dataset.height, val_loader.dataset.width

            for i in range(batch_size):
                pred_map = pred_heatmaps_np[i] # Single heatmap (H, W)

                # Postprocess finds the center (x,y) from the predicted heatmap
                # Returns coords relative to input H, W
                x_pred, y_pred = postprocess(pred_map) # <<<< CHANGED: Pass heatmap directly

                # Ground truth coordinates (already relative to input H,W from dataset) and visibility
                x_gt = batch[2][i].item()
                y_gt = batch[3][i].item()
                vis = int(batch[4][i].item()) # Ensure vis is integer

                is_gt_present = (vis != 0 and x_gt >= 0 and y_gt >= 0)

                if x_pred is not None: # Model predicted a ball location
                    if is_gt_present: # GT ball is present
                        # Calculate distance between predicted and GT coordinates
                        dst = distance.euclidean((x_pred, y_pred), (x_gt, y_gt))
                        if dst < min_dist:
                            tp[vis] += 1 # Correct detection (True Positive for this visibility)
                        else:
                            fp[vis] += 1 # Incorrect location (False Positive for this visibility)
                    else: # GT ball is not present, but model predicted one
                        fp[0] += 1 # False Positive (predicted ball when none present)
                else: # Model did not predict a ball location (postprocess returned None)
                    if is_gt_present: # GT ball is present, but model missed it
                        fn[vis] += 1 # Missed detection (False Negative for this visibility)
                    else: # GT ball is not present, and model correctly didn't predict one
                        tn[0] += 1 # Correctly no detection (True Negative)

            # Update tqdm progress bar less frequently to avoid slowdown
            if iter_id % 50 == 0:
                 val_iterator.set_description(
                    f'Val Epoch {epoch} | Loss: {np.mean(losses):.4f} | TP: {sum(tp)} | FP: {sum(fp)} | FN: {sum(fn)}'
                 )

    eps = 1e-15
    total_tp = sum(tp[1:]) # TP for vis 1, 2, 3
    total_fp = sum(fp)     # FP for vis 0, 1, 2, 3
    total_fn = sum(fn[1:]) # FN for vis 1, 2, 3
    total_actual_positives = total_tp + total_fn # Ground truth balls present (vis 1, 2, 3)

    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_actual_positives + eps) # Recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall + eps)

    print(f'\n--- Validation Epoch {epoch} Results ---')
    print(f'Loss: {np.mean(losses):.6f}')
    print(f'TP: {total_tp} (Vis 1: {tp[1]}, Vis 2: {tp[2]}, Vis 3: {tp[3]})')
    print(f'FP: {total_fp} (Vis 0: {fp[0]}, Vis 1: {fp[1]}, Vis 2: {fp[2]}, Vis 3: {fp[3]})')
    print(f'FN: {total_fn} (Vis 1: {fn[1]}, Vis 2: {fn[2]}, Vis 3: {fn[3]})')
    print(f'TN: {tn[0]} (Correctly no ball detected when Vis=0)')
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1 Score:  {f1:.4f}')
    print('-------------------------------------')

    return np.mean(losses), precision, recall, f1


def postprocess(heatmap, threshold=0.5): # Added threshold
    """
    Finds the coordinates of the maximum value in a heatmap.
    :param heatmap: A 2D numpy array representing the heatmap (H, W).
    :param threshold: Minimum activation value to consider a detection.
    :return: (x, y) coordinates of the maximum value, or (None, None) if max value is below threshold.
    """
    # Find the maximum value and its index
    max_val = np.max(heatmap)

    if max_val < threshold: # <<<< ADDED Threshold check
        return None, None

    # Find the index (row, column) of the maximum value
    # row corresponds to y, column corresponds to x
    y_idx, x_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)

    # Return coordinates as (x, y) relative to heatmap dimensions
    return float(x_idx), float(y_idx) # <<<< CHANGED: Return float coords directly