import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import catboost as ctb
import argparse


def create_features(path_dataset, num_frames):
    # Find game directories more robustly
    game_folders = sorted([f for f in os.listdir(path_dataset) if os.path.isdir(os.path.join(path_dataset, f)) and f.startswith('game')])
    df_list = []  # More efficient to append to list and concat once
    print(f"Found {len(game_folders)} game folders in {path_dataset}")
    for game in tqdm(game_folders, desc="Processing games"):
        clips = os.listdir(os.path.join(path_dataset, game))
        for clip in clips:
            label_path = os.path.join(path_dataset, game, clip, 'Label.csv')
            if not os.path.exists(label_path):
                continue
            try:
                labels = pd.read_csv(label_path)
                # Check required columns
                required_cols = ['x-coordinate', 'y-coordinate', 'status']
                if not all(col in labels.columns for col in required_cols):
                    print(f"Warning: Missing columns in {label_path}, skipping.")
                    continue
            except Exception as e:
                print(f"Error reading {label_path}: {e}")
                continue

            # Ensure coordinates are numeric, coerce errors to NaN
            labels['x-coordinate'] = pd.to_numeric(labels['x-coordinate'], errors='coerce')
            labels['y-coordinate'] = pd.to_numeric(labels['y-coordinate'], errors='coerce')

            eps = 1e-15
            # Feature calculation loop
            for i in range(1, num_frames + 1):  # Loop up to num_frames
                labels[f'x_lag_{i}'] = labels['x-coordinate'].shift(i)
                labels[f'x_lag_inv_{i}'] = labels['x-coordinate'].shift(-i)
                labels[f'y_lag_{i}'] = labels['y-coordinate'].shift(i)
                labels[f'y_lag_inv_{i}'] = labels['y-coordinate'].shift(-i)
                labels[f'x_diff_{i}'] = abs(labels[f'x_lag_{i}'] - labels['x-coordinate'])
                labels[f'y_diff_{i}'] = labels[f'y_lag_{i}'] - labels['y-coordinate']
                labels[f'x_diff_inv_{i}'] = abs(labels[f'x_lag_inv_{i}'] - labels['x-coordinate'])
                labels[f'y_diff_inv_{i}'] = labels[f'y_lag_inv_{i}'] - labels['y-coordinate']
                labels[f'x_div_{i}'] = abs(labels[f'x_diff_{i}'] / (labels[f'x_diff_inv_{i}'] + eps))
                labels[f'y_div_{i}'] = labels[f'y_diff_{i}'] / (labels[f'y_diff_inv_{i}'] + eps)

            # Target: 1 if status is 2 (bounce), 0 otherwise
            labels['target'] = (labels['status'] == 2).astype(int)

            # Drop rows with NaNs created by shifts or original NaNs
            labels = labels.dropna(subset=['x-coordinate', 'y-coordinate'])  # Drop rows with NaN coords first
            # Drop rows where any lag/inv feature is NaN
            lag_cols = [col for col in labels.columns if 'lag' in col]
            labels = labels.dropna(subset=lag_cols)

            # Ensure status is integer
            labels['status'] = labels['status'].astype(int)
            df_list.append(labels)

    if not df_list:
        print("Error: No valid data found after processing features.")
        return pd.DataFrame()

    df = pd.concat(df_list, ignore_index=True)
    print(f"Created features dataframe with shape: {df.shape}")
    return df


def create_train_test(df, num_frames, test_size=0.25, random_state=42):
    # Define feature columns based on num_frames
    colnames_x = [f'x_diff_{i}' for i in range(1, num_frames + 1)] + \
                 [f'x_diff_inv_{i}' for i in range(1, num_frames + 1)] + \
                 [f'x_div_{i}' for i in range(1, num_frames + 1)]
    colnames_y = [f'y_diff_{i}' for i in range(1, num_frames + 1)] + \
                 [f'y_diff_inv_{i}' for i in range(1, num_frames + 1)] + \
                 [f'y_div_{i}' for i in range(1, num_frames + 1)]
    colnames = colnames_x + colnames_y

    # Ensure all feature columns exist in the dataframe
    missing_cols = [col for col in colnames if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns in DataFrame: {missing_cols}")

    # Check if 'target' column exists
    if 'target' not in df.columns:
        raise ValueError("Missing 'target' column in DataFrame")

    # Check for NaN/inf values in feature columns before splitting
    df_features = df[colnames]
    if df_features.isnull().values.any() or np.isinf(df_features.values).any():
        print("Warning: NaN or Inf values found in features. Consider imputation or review feature calculation.")
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=colnames + ['target'])

    # Split data
    X = df[colnames]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    print(f"Train target distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train CatBoost model for bounce detection")
    parser.add_argument('--path_dataset', type=str, required=True, help='Path to the root folder of the raw TrackNet dataset (containing game folders)')
    parser.add_argument('--path_save_model', type=str, required=True, help='Path for saving trained model (e.g., bounce_model.cbm)')
    parser.add_argument('--num_feature_frames', type=int, default=3, help='Number of lag/lead frames for features')
    parser.add_argument('--test_size', type=float, default=0.25, help='Proportion of data for test set')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for splitting data')
    args = parser.parse_args()

    print("Creating features...")
    df_features = create_features(args.path_dataset, args.num_feature_frames)

    if df_features.empty:
        print("Exiting: Feature creation resulted in an empty dataframe.")
    else:
        print("Splitting data into train/test sets...")
        X_train, y_train, X_test, y_test = create_train_test(df_features, args.num_feature_frames,
                                                             test_size=args.test_size, random_state=args.random_state)

        print("Training CatBoostClassifier...")
        train_pool = ctb.Pool(X_train, y_train)
        eval_pool = ctb.Pool(X_test, y_test)

        model_ctb = ctb.CatBoostClassifier(
            loss_function='Logloss',
            eval_metric='Accuracy',
            iterations=200,
            learning_rate=0.1,
            depth=4,
            l2_leaf_reg=1,
            random_seed=args.random_state,
            verbose=50
        )

        model_ctb.fit(train_pool, eval_set=eval_pool, early_stopping_rounds=20)

        print("\nEvaluating model on test set...")
        y_pred_bin = model_ctb.predict(X_test).astype(int)

        accuracy = accuracy_score(y_test, y_pred_bin)
        precision = precision_score(y_test, y_pred_bin)
        recall = recall_score(y_test, y_pred_bin)
        f1 = f1_score(y_test, y_pred_bin)
        cm = confusion_matrix(y_test, y_pred_bin)
        tn, fp, fn, tp = cm.ravel()

        print("\nTest Set Evaluation Metrics:")
        print(f"Confusion Matrix:\n{cm}")
        print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")

        print(f"\nSaving model to: {args.path_save_model}")
        os.makedirs(os.path.dirname(args.path_save_model), exist_ok=True)
        model_ctb.save_model(args.path_save_model, format="cbm")

        print("Bounce detection training finished.")

