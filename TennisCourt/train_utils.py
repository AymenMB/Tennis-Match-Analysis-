# --- START OF FILE base_trainer.py ---

import torch.nn.functional as F
import numpy as np
import torch
from tqdm import tqdm # Added TQDM for progress bar

# Configuration
PRINT_INTERVAL = 100 # Print loss every N iterations

def train(model, train_loader, optimizer, criterion, device, epoch, max_iters=1000):
    model.train() # Ensure model is in training mode
    losses = []
    # Determine the number of iterations: either max_iters or length of loader
    num_batches = len(train_loader)
    iters_this_epoch = min(max_iters, num_batches) if max_iters > 0 else num_batches

    # Use tqdm for progress bar
    progress_bar = tqdm(enumerate(train_loader), total=iters_this_epoch, desc=f"Epoch {epoch} [Train]")

    for iter_id, batch in progress_bar:
        # Stop epoch early if max_iters is reached
        if max_iters > 0 and iter_id >= max_iters:
            break

        # Unpack batch and move to device
        # Assuming batch structure: (input_tensor, target_heatmap, keypoints, img_id)
        inputs = batch[0].to(device, non_blocking=True) # Use non_blocking for potential speedup
        gt_hm_hp = batch[1].to(device, non_blocking=True)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs) # Model output is raw logits/values

        # Apply sigmoid to output before loss calculation (as in original code)
        pred_hm = torch.sigmoid(outputs)

        # Calculate loss
        loss = criterion(pred_hm, gt_hm_hp)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        # Record loss
        loss_item = loss.item()
        losses.append(loss_item)

        # Update progress bar description with current loss
        progress_bar.set_postfix({'loss': f'{loss_item:.5f}'})

        # Optional: Print loss periodically (less frequent than every iteration)
        # if (iter_id + 1) % PRINT_INTERVAL == 0:
        #     print(f'Train Epoch: {epoch} [{iter_id+1}/{iters_this_epoch}] Loss: {loss_item:.5f}')

    # Clean up progress bar
    progress_bar.close()

    # Return the average loss for the epoch
    return np.mean(losses) if losses else 0.0

# --- END OF FILE base_trainer.py ---