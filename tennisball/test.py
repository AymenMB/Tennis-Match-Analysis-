from ball_model import BallTrackerNet
import torch
from datasets import TennisTrackDataset
from utils_general import validate # Uses updated validate function
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4, help='batch size (can be larger for testing)') # Increased default
    parser.add_argument('--model_path', type=str, required=True, help='path to trained model .pt file')
    parser.add_argument('--dataset_dir', type=str, default='./datasets/tennistrack', help='path to processed dataset')
    parser.add_argument('--num_workers', type=int, default=2, help='number of dataloader workers')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    val_dataset = TennisTrackDataset('val', dataset_dir=args.dataset_dir)
    # Use a larger batch size for validation/testing if memory allows
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Load model architecture (ensure out_channels=1)
    model = BallTrackerNet(out_channels=1) # <<<< CHANGED: Specify out_channels=1
    # Load trained weights
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model weights from: {args.model_path}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Ensure the model path is correct and the saved state dict matches the BallTrackerNet(out_channels=1) architecture.")
        exit()

    model = model.to(device)

    print("--- Running Final Validation on Test Set ---")
    # The 'validate' function now calculates metrics based on the regression output
    val_loss, precision, recall, f1 = validate(model, val_loader, device, epoch=-1) # epoch=-1 indicates testing phase

    print("\n--- Test Set Evaluation Summary ---")
    print(f"Model Path: {args.model_path}")
    print(f"Validation Loss: {val_loss:.6f}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"F1 Score:        {f1:.4f}")
    print("---------------------------------")