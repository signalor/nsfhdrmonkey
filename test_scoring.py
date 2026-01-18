"""
Local scoring test script for the enhanced model.
Evaluates the trained model using competition metrics.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

from model import NeuralForecaster, create_model


def calculate_mse(array1, array2):
    """Calculate MSE between two arrays (only on timesteps 10-19)."""
    if array1.shape != array2.shape:
        raise ValueError(f"Shapes don't match: {array1.shape} vs {array2.shape}")

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

    filepath = os.path.join(data_dir, file_map.get(dataset_name, f"train_data_{dataset_name}.npz"))
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
    mean_f0 = mean[0, 0, :, 0]
    std_f0 = std[0, 0, :, 0]
    return data * (4 * std_f0) + mean_f0


def load_model_from_checkpoint(monkey_name, checkpoint_dir, device):
    """Load model from training checkpoint."""
    checkpoint_path = os.path.join(checkpoint_dir, f"best_model_{monkey_name}.pt")
    stats_path = os.path.join(checkpoint_dir, f"norm_stats_{monkey_name}.npz")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get channels
    if monkey_name == "affi":
        n_channels = 239
    else:
        n_channels = 89

    # Create model
    memory_efficient = checkpoint.get("config", {}).get("memory_efficient", False)
    model = create_model(n_channels, device, memory_efficient=memory_efficient)

    # Load weights
    state_dict = checkpoint["model_state_dict"]
    try:
        model.model.load_state_dict(state_dict)
    except RuntimeError:
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"Warning: Could not load state dict directly: {e}")
            # Try loading with strict=False
            model.model.load_state_dict(state_dict, strict=False)

    # Load stats
    stats = None
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        stats = {"mean": stats["mean"], "std": stats["std"]}

    return model, stats


def predict_with_model(model, data, stats, device):
    """Generate predictions using the model."""
    model.eval()
    original_data = data.astype(np.float32)

    # Normalize
    if stats is not None:
        data_norm = normalize_data(original_data, stats["mean"], stats["std"])
    else:
        data_norm = original_data

    predictions = []
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(data_norm), batch_size):
            batch = data_norm[i:i + batch_size]
            batch_orig = original_data[i:i + batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)

            # Predict
            x_input = batch_tensor[:, :10, :, :]
            main_pred, _, _ = model.model(x_input, return_intermediate=False)

            # Denormalize
            main_pred_np = main_pred.cpu().numpy()
            if stats is not None:
                main_pred_np = denormalize_data(main_pred_np, stats["mean"], stats["std"])

            # Combine with original input
            input_feature0 = batch_orig[:, :10, :, 0]
            full_pred = np.concatenate([input_feature0, main_pred_np], axis=1)
            predictions.append(full_pred)

    return np.concatenate(predictions, axis=0)


def main():
    data_dir = "./public_data"
    checkpoint_dir = "./checkpoints"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    datasets = ["affi", "beignet"]  # Start with main datasets

    # Load models
    print("\nLoading models...")
    models = {}
    stats = {}

    for monkey in ["affi", "beignet"]:
        print(f"  Loading {monkey} model...")
        try:
            model, norm_stats = load_model_from_checkpoint(monkey, checkpoint_dir, device)
            model.to(device)
            models[monkey] = model
            stats[monkey] = norm_stats
            print(f"    Loaded successfully")
        except Exception as e:
            print(f"    Failed: {e}")
            models[monkey] = None
            stats[monkey] = None

    # Evaluate
    scores = {}
    for dataset_name in datasets:
        print(f"\nEvaluating {dataset_name}...")

        data = load_test_data(data_dir, dataset_name)
        if data is None:
            continue

        # Use last 10% as test
        n = len(data)
        test_start = int(0.9 * n)
        test_data = data[test_start:]
        print(f"  Test samples: {len(test_data)}")

        monkey = get_monkey_name(dataset_name)
        model = models.get(monkey)
        norm_stats = stats.get(monkey)

        if model is None:
            print(f"  No model for {monkey}")
            continue

        ground_truth = test_data[:, :, :, 0].astype(np.float32)

        try:
            predictions = predict_with_model(model, test_data, norm_stats, device)
            mse = calculate_mse(ground_truth, predictions)
            scores[dataset_name] = mse
            print(f"  MSE: {mse:.2f}")
        except Exception as e:
            print(f"  Failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 50)
    print("SCORING SUMMARY")
    print("=" * 50)

    for name, mse in scores.items():
        print(f"  {name}: {mse:.2f}")

    if scores:
        avg = sum(scores.values()) / len(scores)
        print(f"\n  Average MSE: {avg:.2f}")


if __name__ == "__main__":
    main()
