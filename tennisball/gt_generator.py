import numpy as np
import pandas as pd
import os
import cv2
import argparse

def gaussian_kernel(size, variance):
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp(-(x**2+y**2)/float(2*variance))
    return g

def create_gaussian(size, variance):
    gaussian_kernel_array = gaussian_kernel(size, variance)
    gaussian_kernel_array =  gaussian_kernel_array * 255/gaussian_kernel_array[int(len(gaussian_kernel_array)/2)][int(len(gaussian_kernel_array)/2)]
    gaussian_kernel_array = gaussian_kernel_array.astype(int)
    return gaussian_kernel_array

def create_gt_images(path_input, path_output, size, variance, width, height):
    gaussian_kernel_array = create_gaussian(size, variance)
    game_folders = sorted([f for f in os.listdir(path_input) if os.path.isdir(os.path.join(path_input, f)) and f.startswith('game')])
    for game in game_folders:
        clips = os.listdir(os.path.join(path_input, game))
        for clip in clips:
            print('game = {}, clip = {}'.format(game, clip))

            path_out_game = os.path.join(path_output, game)
            if not os.path.exists(path_out_game):
                os.makedirs(path_out_game)

            path_out_clip = os.path.join(path_out_game, clip)    
            if not os.path.exists(path_out_clip):
                os.makedirs(path_out_clip)  

            path_labels = os.path.join(path_input, game, clip, 'Label.csv')
            if not os.path.exists(path_labels):
                print(f"Warning: Label.csv not found in {os.path.join(path_input, game, clip)}. Skipping clip.")
                continue
            labels = pd.read_csv(path_labels)
            required_cols = ['file name', 'visibility', 'x-coordinate', 'y-coordinate', 'status']
            if not all(col in labels.columns for col in required_cols):
                print(f"Warning: Missing required columns in {path_labels}. Skipping clip.")
                continue

            for idx in range(labels.shape[0]):
                row = labels.iloc[idx]
                file_name = row['file name']
                vis = row['visibility']
                x = row['x-coordinate']
                y = row['y-coordinate']

                heatmap = np.zeros((height, width, 3), dtype=np.uint8)
                if vis != 0 and pd.notna(x) and pd.notna(y):
                    x = int(x)
                    y = int(y)
                    for i in range(-size, size+1):
                        for j in range(-size, size+1):
                                if x+i<width and x+i>=0 and y+j<height and y+j>=0 :
                                    temp = gaussian_kernel_array[i+size][j+size]
                                    if temp > 0:
                                        heatmap[y+j,x+i] = (temp,temp,temp)

                cv2.imwrite(os.path.join(path_out_clip, file_name), heatmap) 
                
def create_gt_labels(path_input, path_output, train_rate=0.7):
    df_list = []
    game_folders = sorted([f for f in os.listdir(path_input) if os.path.isdir(os.path.join(path_input, f)) and f.startswith('game')])
    for game in game_folders:
        clips = os.listdir(os.path.join(path_input, game))
        for clip in clips:
            label_path = os.path.join(path_input, game, clip, 'Label.csv')
            if not os.path.exists(label_path):
                continue

            labels = pd.read_csv(label_path)
            required_cols = ['file name', 'visibility', 'x-coordinate', 'y-coordinate', 'status']
            if not all(col in labels.columns for col in required_cols):
                continue

            labels['gt_path'] = 'gts/' + game + '/' + clip + '/' + labels['file name']
            labels['path1'] = 'images/' + game + '/' + clip + '/' + labels['file name']

            if labels.shape[0] < 3:
                continue

            labels_target = labels.iloc[2:].copy()
            labels_target['path2'] = labels['path1'].iloc[1:-1].values
            labels_target['path3'] = labels['path1'].iloc[:-2].values

            df_list.append(labels_target)

    if not df_list:
        print("Error: No valid labels found to process.")
        return

    df = pd.concat(df_list, ignore_index=True)
    df = df[['path1', 'path2', 'path3', 'gt_path', 'x-coordinate', 'y-coordinate', 'status', 'visibility']]
    df = df.sample(frac=1, random_state=42)
    num_train = int(df.shape[0]*train_rate)
    df_train = df[:num_train]
    df_test = df[num_train:]
    df_train.to_csv(os.path.join(path_output, 'labels_train.csv'), index=False)
    df_test.to_csv(os.path.join(path_output, 'labels_val.csv'), index=False)
    print(f"Generated labels: {len(df_train)} train, {len(df_test)} val.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Ground Truth Heatmaps and Labels for TrackNet")
    parser.add_argument('--path_input', type=str, required=True, help='Path to the root folder of the raw dataset (containing game1, game2, etc.)')
    parser.add_argument('--path_output', type=str, required=True, help='Path to the output folder for processed data (gts, labels_train.csv, labels_val.csv)')
    parser.add_argument('--heatmap_size', type=int, default=20, help='Size parameter for Gaussian kernel')
    parser.add_argument('--heatmap_variance', type=int, default=10, help='Variance parameter for Gaussian kernel')
    parser.add_argument('--image_width', type=int, default=1280, help='Original image width for heatmap generation')
    parser.add_argument('--image_height', type=int, default=720, help='Original image height for heatmap generation')
    parser.add_argument('--train_rate', type=float, default=0.7, help='Proportion of data for training set')
    args = parser.parse_args()

    if not os.path.exists(args.path_output):
        os.makedirs(args.path_output)

    gts_path = os.path.join(args.path_output, 'gts')
    if not os.path.exists(gts_path):
        os.makedirs(gts_path)
        
    print("Creating Ground Truth Heatmap Images...")
    create_gt_images(args.path_input, gts_path,
                     args.heatmap_size, args.heatmap_variance,
                     args.image_width, args.image_height)

    print("\nCreating Training/Validation Label Files...")
    create_gt_labels(args.path_input, args.path_output, args.train_rate)

    print("\nGround truth generation finished.")





