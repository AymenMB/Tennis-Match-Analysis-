# --- START OF FILE main.py ---

from dataset import courtDataset
import torch
import torch.nn as nn
from torch.optim import lr_scheduler # Import scheduler
from base_trainer import train
from base_validator import val
import os
from torch.utils.tensorboard import SummaryWriter # Use PyTorch's TensorBoard writer
from tennistrack import BallTrackerNet
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4, help='batch size (adjust based on GPU memory)') # Increased default slightly
    parser.add_argument('--exp_id', type=str, default='default_aug_lr_dropout', help='path to saving results') # Updated default name
    parser.add_argument('--num_epochs', type=int, default=200, help='total training epochs') # Reduced default slightly, adjust as needed
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate') # Common starting LR
    parser.add_argument('--val_intervals', type=int, default=5, help='number of epochs to run validation')
    parser.add_argument('--steps_per_epoch', type=int, default=1000, help='max number of steps per one epoch') # Max iters per epoch
    parser.add_argument('--num_workers', type=int, default=4, help='number of dataloader workers') # Increase workers if CPU allows
    parser.add_argument('--resume_path', type=str, default=None, help='path to checkpoint to resume training from') # Add resume option
    args = parser.parse_args()

    # Setup paths
    exps_path = os.path.join('./exps', args.exp_id)
    tb_path = os.path.join(exps_path, 'tensorboard_logs')
    model_save_path = os.path.join(exps_path, 'checkpoints')
    os.makedirs(tb_path, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)

    model_last_path = os.path.join(model_save_path, 'model_last.pt')
    model_best_path = os.path.join(model_save_path, 'model_best.pt')

    # Setup TensorBoard
    log_writer = SummaryWriter(tb_path)

    # Setup Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Datasets and Dataloaders
    print("Loading datasets...")
    train_dataset = courtDataset('train')
    # Optional: Filter dataset if needed (e.g., train_dataset.filter_data())
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True, # Use pin_memory if device is cuda
        drop_last=True # Drop last incomplete batch
    )

    val_dataset = courtDataset('val')
    # Optional: Filter dataset if needed (e.g., val_dataset.filter_data())
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2, # Can often use larger batch size for validation
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    print("Datasets loaded.")

    # Model
    print("Initializing model...")
    model = BallTrackerNet(out_channels=15) # Ensure correct output channels
    model = model.to(device)

    # If multiple GPUs are available, wrap the model
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    print("Model initialized.")

    # Loss Function
    criterion = nn.MSELoss() # Keep MSE loss for heatmap regression

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), weight_decay=1e-5) # Added slight weight decay

    # Learning Rate Scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=True) # Reduce LR if val_loss doesn't improve

    # Resume Training (Optional)
    start_epoch = 0
    val_best_accuracy = 0.0 # Initialize best accuracy

    if args.resume_path and os.path.exists(args.resume_path):
        print(f"Resuming training from {args.resume_path}")
        checkpoint = torch.load(args.resume_path, map_location=device)
        # Adjust for DataParallel wrapper if necessary
        if isinstance(model, nn.DataParallel):
             model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Load scheduler state if saved
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
             scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        # Load best accuracy if saved, otherwise default to 0
        val_best_accuracy = checkpoint.get('best_accuracy', 0.0)
        print(f"Resumed from epoch {start_epoch}, best accuracy so far: {val_best_accuracy:.5f}")


    # Training Loop
    print("Starting training...")
    for epoch in range(start_epoch, args.num_epochs):
        # --- Training Phase ---
        model.train() # Set model to training mode
        train_loss = train(model, train_loader, optimizer, criterion, device, epoch, args.steps_per_epoch)
        log_writer.add_scalar('Train/Loss', train_loss, epoch)
        log_writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], epoch) # Log LR
        print(f"Epoch {epoch}/{args.num_epochs} - Train Loss: {train_loss:.5f}")

        # --- Validation Phase ---
        if (epoch + 1) % args.val_intervals == 0: # Validate every N epochs
            model.eval() # Set model to evaluation mode
            val_loss, tp, fp, fn, tn, precision, accuracy = val(model, val_loader, criterion, device, epoch)

            print(f"Epoch {epoch}/{args.num_epochs} - Validation Loss: {val_loss:.5f}, Accuracy: {accuracy:.5f}, Precision: {precision:.5f}")
            log_writer.add_scalar('Val/Loss', val_loss, epoch)
            log_writer.add_scalar('Val/TP', tp, epoch)
            log_writer.add_scalar('Val/FP', fp, epoch)
            log_writer.add_scalar('Val/FN', fn, epoch)
            log_writer.add_scalar('Val/TN', tn, epoch)
            log_writer.add_scalar('Val/Precision', precision, epoch)
            log_writer.add_scalar('Val/Accuracy', accuracy, epoch)

            # Save best model based on validation accuracy
            if accuracy > val_best_accuracy:
                val_best_accuracy = accuracy
                print(f"*** New best model found (Accuracy: {accuracy:.5f})! Saving to {model_best_path} ***")
                save_state = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_accuracy': val_best_accuracy,
                    'loss': val_loss,
                }
                torch.save(save_state, model_best_path)

            # Step the scheduler based on validation loss
            scheduler.step(val_loss)

        # Save last model checkpoint periodically or at the end
        if (epoch + 1) % args.val_intervals == 0: # Save last checkpoint frequently
             print(f"Saving last model checkpoint to {model_last_path}")
             save_state = {
                 'epoch': epoch,
                 'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                 'best_accuracy': val_best_accuracy, # Include best accuracy in last checkpoint too
                 'loss': train_loss, # Save train loss for context
             }
             torch.save(save_state, model_last_path)


    log_writer.close()
    print("Training finished.")
    print(f"Best validation accuracy achieved: {val_best_accuracy:.5f}")

# --- END OF FILE main.py ---