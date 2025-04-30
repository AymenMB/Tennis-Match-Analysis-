from ball_model import BallTrackerNet
import torch
from dataset import TennisTrackDataset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler # Import scheduler
import os
from tensorboardX import SummaryWriter
from utils_general import train, validate
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Adjusted defaults: smaller batch size for Colab, slightly lower initial LR for AdamW
    parser.add_argument('--batch_size', type=int, default=2, help='batch size (reduce if memory issues)')
    parser.add_argument('--exp_id', type=str, default='tennistrack_mse_v1', help='path to saving results')
    parser.add_argument('--num_epochs', type=int, default=100, help='total training epochs (adjust as needed)') # Reduced default for quicker testing
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for AdamW') # <<<< CHANGED: Default LR for AdamW
    parser.add_argument('--val_intervals', type=int, default=1, help='number of epochs to run validation') # Validate more often
    parser.add_argument('--steps_per_epoch', type=int, default=500, help='max number of steps per one training epoch') # Increased steps
    parser.add_argument('--dataset_dir', type=str, default='./datasets/tennistrack', help='path to processed dataset')
    parser.add_argument('--num_workers', type=int, default=2, help='number of dataloader workers')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = TennisTrackDataset('train', dataset_dir=args.dataset_dir)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True # drop_last can help with BatchNorm issues if last batch is size 1
    )

    val_dataset = TennisTrackDataset('val', dataset_dir=args.dataset_dir)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Use the model with 1 output channel
    model = BallTrackerNet(out_channels=1).to(device)

    exps_path = './exps/{}'.format(args.exp_id)
    tb_path = os.path.join(exps_path, 'plots')
    if not os.path.exists(tb_path): os.makedirs(tb_path)
    log_writer = SummaryWriter(tb_path)
    model_last_path = os.path.join(exps_path, 'model_last.pt')
    model_best_path = os.path.join(exps_path, 'model_best.pt')

    # Use AdamW optimizer and ReduceLROnPlateau scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr) # <<<< CHANGED: AdamW
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5, verbose=True) # <<<< ADDED: Scheduler based on F1

    val_best_metric = -1.0 # Initialize best F1 score

    print(f"--- Starting Training ---")
    print(f"Experiment ID: {args.exp_id}")
    print(f"Dataset Dir: {args.dataset_dir}")
    print(f"Output Dir: {exps_path}")
    print(f"Epochs: {args.num_epochs}, Steps/Epoch: {args.steps_per_epoch}")
    print(f"Batch Size: {args.batch_size}, LR: {args.lr}, Optimizer: AdamW")
    print(f"-------------------------")

    for epoch in range(args.num_epochs):
        # Train one epoch
        train_loss = train(model, train_loader, optimizer, device, epoch, args.steps_per_epoch)
        log_writer.add_scalar('Train/loss', train_loss, epoch)
        log_writer.add_scalar('Train/learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # Validation block
        if (epoch + 1) % args.val_intervals == 0:
            val_loss, precision, recall, f1 = validate(model, val_loader, device, epoch)
            log_writer.add_scalar('Val/loss', val_loss, epoch)
            log_writer.add_scalar('Val/precision', precision, epoch)
            log_writer.add_scalar('Val/recall', recall, epoch)
            log_writer.add_scalar('Val/f1', f1, epoch)

            # Step the scheduler based on F1 score
            scheduler.step(f1) # <<<< ADDED: Step scheduler

            # Save best model based on F1 score
            if f1 > val_best_metric:
                val_best_metric = f1
                print(f"*** New best F1 score: {f1:.4f}. Saving model to {model_best_path} ***")
                torch.save(model.state_dict(), model_best_path)
            else:
                 print(f"F1 score {f1:.4f} did not improve from best {val_best_metric:.4f}")

            # Save last model
            torch.save(model.state_dict(), model_last_path)
            print(f"Saved last model checkpoint to {model_last_path}")

    log_writer.close()
    print("--- Training Finished ---")
    print(f"Best F1 score achieved: {val_best_metric:.4f}")
    print(f"Best model saved to: {model_best_path}")
    print(f"Last model saved to: {model_last_path}")