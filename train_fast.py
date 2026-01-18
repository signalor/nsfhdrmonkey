"""
Fast Training script for Neural Forecasting Competition.
Simplified version that works with model_fast.py
"""

import argparse
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model_fast import NeuralForecaster, create_model, ForecastingLoss


# =============================================================================
# Data Loading
# =============================================================================


def normalize(data: np.ndarray, average: np.ndarray = None, std: np.ndarray = None):
    """Normalize data to approximately [-1, 1] range."""
    if data.ndim == 4:
        n, t, c, f = data.shape
        data = data.reshape((n * t, -1))
    else:
        n, t, c, f = None, None, None, None

    if average is None:
        average = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)

    combine_max = average + 4 * std
    combine_min = average - 4 * std
    norm_data = 2 * (data - combine_min) / (combine_max - combine_min + 1e-8) - 1

    if n is not None:
        norm_data = norm_data.reshape((n, t, c, f))

    return norm_data, average, std


class NeuralForecastDataset(Dataset):
    """Dataset for neural forecasting."""

    def __init__(
        self,
        data: np.ndarray,
        average: np.ndarray = None,
        std: np.ndarray = None,
    ):
        self.raw_data = data.copy()
        self.data, self.average, self.std = normalize(data, average, std)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)


def load_data(
    data_dir: str, monkey_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Load training data for a specific monkey."""

    train_file = os.path.join(data_dir, f"train_data_{monkey_name}.npz")
    data = np.load(train_file)["arr_0"]

    # Split into train/val
    n = len(data)
    train_end = int(n * 0.9)

    train_data = data[:train_end]
    val_data = data[train_end:]

    # Try to load additional data
    for f in os.listdir(data_dir):
        if f.startswith(f"train_data_{monkey_name}_") and f.endswith(".npz"):
            try:
                add_data = np.load(os.path.join(data_dir, f))["arr_0"]
                train_data = np.concatenate([train_data, add_data], axis=0)
                print(f"  Added {len(add_data)} samples from {f}")
            except Exception as e:
                print(f"  Warning: Could not load {f}: {e}")

    return train_data, val_data


# =============================================================================
# Training
# =============================================================================


def validate_model(
    model: NeuralForecaster,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Validate model and return MSE."""
    model.eval()
    total_mse = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model(batch)
            target = batch[:, :, :, 0]
            mse = F.mse_loss(pred[:, 10:, :], target[:, 10:, :], reduction="mean")
            total_mse += mse.item()
            n_batches += 1

    return total_mse / max(n_batches, 1)


def train_model(
    train_data: np.ndarray,
    val_data: np.ndarray,
    monkey_name: str,
    device: torch.device,
    batch_size: int = 16,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    save_dir: str = "./checkpoints",
):
    """Train model for a specific monkey."""

    print(f"\nData shapes for {monkey_name}:")
    print(f"  Train: {train_data.shape}")
    print(f"  Val: {val_data.shape}")

    n_channels = train_data.shape[2]
    print(f"  Channels: {n_channels}")

    # Create datasets
    train_dataset = NeuralForecastDataset(train_data)
    val_dataset = NeuralForecastDataset(
        val_data, train_dataset.average, train_dataset.std
    )

    # Save normalization stats
    os.makedirs(save_dir, exist_ok=True)
    np.savez(
        os.path.join(save_dir, f"norm_stats_{monkey_name}.npz"),
        average=train_dataset.average,
        std=train_dataset.std,
    )

    # Data loaders
    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Create model
    model = create_model(n_channels, device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    loss_fn = ForecastingLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )

    # Scheduler with warmup
    warmup_epochs = 5

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    best_val_mse = float("inf")
    patience = 15
    patience_counter = 0

    print(f"\nTraining {monkey_name}...")
    print("=" * 50)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        n_steps = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred = model(batch)
            target = batch[:, :, :, 0]
            loss, _ = loss_fn(pred, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_steps += 1

        avg_loss = total_loss / max(n_steps, 1)
        val_mse = validate_model(model, val_loader, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:3d}/{num_epochs} | "
            f"Loss: {avg_loss:.6f} | Val MSE: {val_mse:.6f} | LR: {lr:.2e}"
        )

        # Early stopping
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_mse": val_mse,
                    "n_channels": n_channels,
                },
                os.path.join(save_dir, f"best_model_{monkey_name}.pth"),
            )
            print(f"  -> Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # Load best model
    checkpoint = torch.load(
        os.path.join(save_dir, f"best_model_{monkey_name}.pth"),
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"\nBest Val MSE: {checkpoint['val_mse']:.6f}")

    return model


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Fast Train Neural Forecasting Model")
    parser.add_argument(
        "data_dir",
        type=str,
        nargs="?",
        default="./hackdata/neural-forecasting/public_data/",
        help="Path to data directory",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--save_dir", type=str, default="./checkpoints", help="Save directory"
    )
    parser.add_argument(
        "--monkey",
        type=str,
        default="all",
        choices=["all", "affi", "beignet"],
        help="Which monkey to train",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    monkeys = ["affi", "beignet"] if args.monkey == "all" else [args.monkey]

    for monkey in monkeys:
        print("\n" + "=" * 50)
        print(f"Training {monkey}")
        print("=" * 50)

        try:
            train_data, val_data = load_data(args.data_dir, monkey)
            train_model(
                train_data=train_data,
                val_data=val_data,
                monkey_name=monkey,
                device=device,
                batch_size=args.batch_size,
                num_epochs=args.epochs,
                learning_rate=args.lr,
                save_dir=args.save_dir,
            )
        except Exception as e:
            print(f"Error training {monkey}: {e}")
            import traceback
            traceback.print_exc()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()