"""
Local scoring test script.
Uses the public data to evaluate the trained model using checkpoints.
"""

import os
import sys

import numpy as np
import torch

from model import NeuralForecaster, create_model


def calculate_mse(array1, array2):
    """Calculate MSE between two arrays (only on timesteps 10-19)."""
    if array1.shape != array2.shape:
        raise ValueError(
            f"Shapes don't match: {array1.shape} vs {array2.shape}. Expected (N, 20, C)"
        )

    # Only score timesteps 10-19 (the predictions)
    array1 = torch.tensor(array1[:, 10:])
    array2 = torch.tensor(array2[:, 10:])

    mse = torch.nn.functional.mse_loss(array1, array2, reduction="mean")
    return mse.item()


def load_test_data(data_dir, dataset_name):
    """Load test data for a dataset."""
    file_map = {
        "affi": "train_data_affi.npz",
        "beignet": "train_data_beignet.npz",
        "affi_private": "train_data_affi_2024-03-20_private.npz",
        "beignet_private_1": "train_data_beignet_2022-06-01_private.npz",
        "beignet_private_2": "train_data_beignet_2022-06-02_private.npz",
    }

    filepath = os.path.join(data_dir, file_map[dataset_name])
    if not os.path.exists(filepath):
        print(f"  File not found: {filepath}")
        return None

    data = np.load(filepath)["arr_0"]
    return data


def get_monkey_name(dataset_name):
    """Get monkey name from dataset name."""
    if "affi" in dataset_name:
        return "affi"
    return "beignet"


def normalize_data(data, mean, std):
    """Normalize data using training stats."""
    return (data - mean) / (4 * std)


def denormalize_data(data, mean, std):
    """Denormalize data back to original scale."""
    # data shape: (N, T, C), mean/std shape: (1, 1, C, F)
    # We need mean/std for feature 0 only: (1, 1, C, 1) -> (C,)
    mean_f0 = mean[0, 0, :, 0]  # (C,)
    std_f0 = std[0, 0, :, 0]  # (C,)
    return data * (4 * std_f0) + mean_f0


def load_model_from_checkpoint(monkey_name, checkpoint_dir, device):
    """Load model from training checkpoint."""
    checkpoint_path = os.path.join(checkpoint_dir, f"best_model_{monkey_name}.pt")
    stats_path = os.path.join(checkpoint_dir, f"norm_stats_{monkey_name}.npz")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get number of channels from data
    if monkey_name == "affi":
        n_channels = 239
    else:
        n_channels = 89

    # Create model (memory_efficient=True to match training)
    model = create_model(n_channels, device, memory_efficient=True)

    # Load weights - handle DataParallel wrapping
    state_dict = checkpoint["model_state_dict"]

    # Check if the inner model needs the weights
    try:
        model.model.load_state_dict(state_dict)
    except RuntimeError:
        # Try loading into the full model
        model.load_state_dict(state_dict)

    # Load normalization stats
    stats = None
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        stats = {"mean": stats["mean"], "std": stats["std"]}

    return model, stats


def predict_with_model(model, data, stats, device):
    """Generate predictions using the model."""
    model.eval()

    # Keep original data for input feature 0 (first 10 timesteps)
    original_data = data.astype(np.float32)

    # Normalize data if stats available
    if stats is not None:
        data_norm = normalize_data(original_data, stats["mean"], stats["std"])
    else:
        data_norm = original_data

    predictions = []

    with torch.no_grad():
        # Process in batches to avoid memory issues
        batch_size = 32
        for i in range(0, len(data_norm), batch_size):
            batch = data_norm[i : i + batch_size]
            batch_orig = original_data[i : i + batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)

            # Forward pass
            x_input = batch_tensor[:, :10, :, :]
            main_pred, _ = model.model(x_input)

            # Denormalize predictions back to original scale
            main_pred_np = main_pred.cpu().numpy()
            if stats is not None:
                main_pred_np = denormalize_data(
                    main_pred_np, stats["mean"], stats["std"]
                )

            # Combine original input (first 10 steps) and denormalized prediction (last 10)
            input_feature0 = batch_orig[:, :10, :, 0]  # Original scale
            full_pred = np.concatenate([input_feature0, main_pred_np], axis=1)

            predictions.append(full_pred)

    return np.concatenate(predictions, axis=0)


def main():
    data_dir = "./public_data"
    checkpoint_dir = "./checkpoints"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets to evaluate
    datasets = [
        "affi",
        "beignet",
        "affi_private",
        "beignet_private_1",
        "beignet_private_2",
    ]

    # Load models
    print("\nLoading models...")
    models = {}
    stats = {}

    for monkey in ["affi", "beignet"]:
        print(f"  Loading {monkey} model...")
        try:
            model, norm_stats = load_model_from_checkpoint(
                monkey, checkpoint_dir, device
            )
            model.to(device)
            models[monkey] = model
            stats[monkey] = norm_stats
            print(f"    ✓ Loaded successfully")
        except Exception as e:
            print(f"    ✗ Failed to load: {e}")
            import traceback

            traceback.print_exc()
            models[monkey] = None
            stats[monkey] = None

    print()

    # Evaluate each dataset
    scores = {}

    for dataset_name in datasets:
        print(f"Evaluating {dataset_name}...")

        # Load data
        data = load_test_data(data_dir, dataset_name)
        if data is None:
            scores[dataset_name] = None
            continue

        print(f"  Data shape: {data.shape}")

        # Get appropriate model
        monkey = get_monkey_name(dataset_name)
        model = models.get(monkey)
        norm_stats = stats.get(monkey)

        if model is None:
            print(f"  ✗ No model available for {monkey}")
            scores[dataset_name] = None
            continue

        # Get ground truth (feature 0 only)
        ground_truth = data[:, :, :, 0].astype(np.float32)  # (N, 20, C)

        # Generate predictions
        print(f"  Generating predictions...")
        try:
            predictions = predict_with_model(model, data, norm_stats, device)
            print(f"  Predictions shape: {predictions.shape}")
        except Exception as e:
            print(f"  ✗ Prediction failed: {e}")
            import traceback

            traceback.print_exc()
            scores[dataset_name] = None
            continue

        # Calculate MSE
        try:
            mse = calculate_mse(ground_truth, predictions)
            scores[dataset_name] = mse
            print(f"  ✓ MSE: {mse:.6f}")
        except Exception as e:
            print(f"  ✗ MSE calculation failed: {e}")
            import traceback

            traceback.print_exc()
            scores[dataset_name] = None

        print()

    # Summary
    print("=" * 50)
    print("SCORING SUMMARY")
    print("=" * 50)

    valid_scores = []
    for dataset_name in datasets:
        mse = scores.get(dataset_name)
        if mse is not None:
            print(f"  {dataset_name}: {mse:.6f}")
            valid_scores.append(mse)
        else:
            print(f"  {dataset_name}: FAILED")

    if valid_scores:
        avg_mse = sum(valid_scores) / len(valid_scores)
        print()
        print(f"  Average MSE: {avg_mse:.6f}")

    print("=" * 50)


if __name__ == "__main__":
    main()
