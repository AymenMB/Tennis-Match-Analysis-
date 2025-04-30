import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
import argparse

# Import model
from tennisball.model import BallTrackerNet

class BallDetectionDataset(Dataset):
    def __init__(self, dataset_dir, labels_path):
        self.labels = pd.read_csv(labels_path)
        self.dataset_dir = dataset_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        
        # Load frames
        path1 = os.path.join(self.dataset_dir, row['path1'])
        path2 = os.path.join(self.dataset_dir, row['path2'])
        path3 = os.path.join(self.dataset_dir, row['path3'])
        
        # Extract ground truth coordinates
        x_gt = row['x-coordinate']
        y_gt = row['y-coordinate']
        
        # Check if coordinates are available (not NaN)
        if pd.isna(x_gt) or pd.isna(y_gt):
            x_gt = -1
            y_gt = -1
        
        # Load images
        img1 = self.transform(Image.open(path1).convert('RGB'))
        img2 = self.transform(Image.open(path2).convert('RGB'))
        img3 = self.transform(Image.open(path3).convert('RGB'))
        
        # Create 9-channel input (3 consecutive frames)
        input_tensor = torch.cat([img1, img2, img3], dim=0)
        
        return input_tensor, (float(x_gt), float(y_gt))

def postprocess(output_processed):
    """
    Process the output from the ball detection model to extract coordinates.
    
    Args:
        output_processed: The processed output (argmaxed across channels) from the model
        
    Returns:
        x_coord, y_coord: Predicted ball coordinates in original image space
    """
    # Check if the output is valid
    if output_processed is None:
        return None, None
    
    # Original image dimensions (from sample.tex - 1280x720)
    original_height, original_width = 720, 1280
    
    # Handle 1D array (flattened output)
    if len(output_processed.shape) == 1:
        # Find index of max value in flattened array
        idx = np.argmax(output_processed)
        
        # Calculate 2D coordinates assuming row-major flattening (height, width)
        # Assuming the 1D array of length 921600 represents a flattened 720x1280 grid
        y_max = idx // original_width
        x_max = idx % original_width
        
        # No need to scale since we're already at original resolution
        return x_max, y_max
    else:
        # Original logic for 2D heatmaps
        height, width = output_processed.shape
        y_max, x_max = np.unravel_index(np.argmax(output_processed), output_processed.shape)
        
        # Scale coordinates to original image size
        x_pred = int(x_max * original_width / width)
        y_pred = int(y_max * original_height / height)
        
        return x_pred, y_pred

def evaluate_model(model, data_loader, device):
    model.eval()
    true_positives = 0
    total_samples = 0
    valid_samples = 0
    
    with torch.no_grad():
        for batch_idx, (input_tensor, gt_coords) in enumerate(data_loader):
            # Move input to device
            input_tensor = input_tensor.to(device)
            
            # Forward pass
            output = model(input_tensor)
            
            if batch_idx == 0:
                print(f"Model output shape: {output.shape}")
            
            # Process batch - use argmax across channels like in infer_on_video.py
            output_processed = output.argmax(dim=1).detach().cpu().numpy()
            
            batch_size = input_tensor.size(0)
            for i in range(batch_size):
                # Extract ground truth coordinates
                x_gt, y_gt = gt_coords[0][i], gt_coords[1][i]
                
                # Skip samples without valid ground truth
                if x_gt < 0 or y_gt < 0:
                    continue
                
                valid_samples += 1
                
                # Use postprocess to extract ball coordinates like in inference
                x_pred, y_pred = postprocess(output_processed[i])
                
                if x_pred is not None and y_pred is not None:
                    # Evaluate distance
                    dist = np.sqrt((x_pred - x_gt)**2 + (y_pred - y_gt)**2)
                    threshold = 10  # 10-pixel threshold as in your code
                    
                    if dist <= threshold:
                        true_positives += 1
                
            total_samples += batch_size
    
    # Calculate metrics (handle division by zero)
    precision = true_positives / valid_samples if valid_samples > 0 else 0
    recall = precision  # Same as in your original code
    f1 = precision      # Same as in your original code
    
    return 0.0, precision, recall, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='path to model')
    parser.add_argument('--dataset_dir', type=str, required=True, help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and filter labels
    all_labels = pd.DataFrame()
    train_labels_path = os.path.join(args.dataset_dir, 'labels_train.csv')
    val_labels_path = os.path.join(args.dataset_dir, 'labels_val.csv')
    
    if os.path.exists(train_labels_path):
        train_labels = pd.read_csv(train_labels_path)
        all_labels = pd.concat([all_labels, train_labels])
        print(f"Loaded {len(train_labels)} training samples")
    
    if os.path.exists(val_labels_path):
        val_labels = pd.read_csv(val_labels_path)
        all_labels = pd.concat([all_labels, val_labels])
        print(f"Loaded {len(val_labels)} validation samples")
    
    # Filter for game10/Clip8 only with explicit path check
    game1_clip1_labels = all_labels[all_labels['gt_path'].str.contains(r'game10/Clip8/')]
    print(f"Found {len(game1_clip1_labels)} samples for game10/Clip8")
    
    if len(game1_clip1_labels) == 0:
        print("No data found for game10/Clip8. Please check dataset paths.")
        return
    
    # Create temporary CSV file for the filtered data
    temp_csv_path = os.path.join(args.dataset_dir, 'temp_game1_clip1.csv')
    game1_clip1_labels.to_csv(temp_csv_path, index=False)
    
    try:
        # Create dataset and dataloader
        dataset = BallDetectionDataset(args.dataset_dir, temp_csv_path)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        print(f"Created dataset with {len(dataset)} samples")
        
        # Load model
        model = BallTrackerNet()
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model = model.to(device)
        print(f"Successfully loaded model from {args.model_path}")
        
        # Evaluate model
        print("\nRunning evaluation on Game10/Clip8...")
        _, precision, recall, f1 = evaluate_model(model, dataloader, device)
        
        print("\n=== Evaluation Results on Game10/Clip8 ===")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)

if __name__ == "__main__":
    main()