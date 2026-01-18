"""
Enhanced Training script for Neural Forecasting Competition.
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import (
    EnhancedForecastingLoss,
    NeuralForecaster,
    create_model,
    validate,
    validate_competition,
)

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


def denormalize(data: np.ndarray, average: np.ndarray, std: np.ndarray):
    """Denormalize data back to original scale."""
    combine_max = average + 4 * std
    combine_min = average - 4 * std

    if data.ndim == 4:
        n, t, c, f = data.shape
        data = data.reshape((n * t, -1))
        denorm_data = (data + 1) / 2 * (combine_max - combine_min + 1e-8) + combine_min
        return denorm_data.reshape((n, t, c, f))
    else:
        return (data + 1) / 2 * (combine_max - combine_min + 1e-8) + combine_min


class NeuralForecastDataset(Dataset):
    """Dataset for neural forecasting."""

    def __init__(
        self,
        data: np.ndarray,
        average: np.ndarray = None,
        std: np.ndarray = None,
        augment: bool = False,
    ):
        self.raw_data = data.copy()
        self.augment = augment

        # Normalize
        self.data, self.average, self.std = normalize(data, average, std)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx].copy()

        if self.augment and self.training:
            data = self._augment(data)

        return torch.tensor(data, dtype=torch.float32)

    def _augment(self, data: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        # Add small noise
        if np.random.random() < 0.3:
            noise = np.random.randn(*data.shape) * 0.01
            data = data + noise

        # Channel dropout (zero out some channels)
        if np.random.random() < 0.2:
            n_drop = np.random.randint(1, max(2, data.shape[1] // 20))
            drop_idx = np.random.choice(data.shape[1], n_drop, replace=False)
            data[:, drop_idx, :] = 0

        return data

    @property
    def training(self):
        return self.augment


def load_data(
    data_dir: str, monkey_name: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load training data for a specific monkey."""

    # Main training file
    train_file = os.path.join(data_dir, f"train_data_{monkey_name}.npz")
    data = np.load(train_file)["arr_0"]

    # Split into train/val/test
    n = len(data)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # Try to load additional data
    additional_files = []
    for f in os.listdir(data_dir):
        if f.startswith(f"train_data_{monkey_name}_") and f.endswith(".npz"):
            additional_files.append(os.path.join(data_dir, f))

    for add_file in additional_files:
        try:
            add_data = np.load(add_file)["arr_0"]
            train_data = np.concatenate([train_data, add_data], axis=0)
            print(f"  Added {len(add_data)} samples from {os.path.basename(add_file)}")
        except Exception as e:
            print(f"  Warning: Could not load {add_file}: {e}")

    return train_data, val_data, test_data


# =============================================================================
# Training Functions
# =============================================================================


def get_memory_info():
    """Get system memory information."""
    mem = psutil.virtual_memory()
    return mem.total / (1024**3), mem.available / (1024**3)


def train_step_with_accumulation(
    model: NeuralForecaster,
    batch: torch.Tensor,
    loss_fn: EnhancedForecastingLoss,
    optimizer: torch.optim.Optimizer,
    accumulation_steps: int,
    step_in_accumulation: int,
    device: torch.device,
    use_amp: bool = False,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    """Single training step with gradient accumulation."""

    batch = batch.to(device)

    # Forward pass
    main_pred, aux_pred, intermediate = model.forward_with_intermediate(batch)

    # Prepare targets (future timesteps only, matching model output shape)
    target = batch[:, 10:, :, 0]  # Main target (feature 0, future only)
    aux_target = batch[:, 10:, :, :]  # Auxiliary target (all features, future only)

    # Compute loss
    loss, metrics = loss_fn(
        main_pred=main_pred,
        aux_pred=aux_pred,
        target=target,
        target_full=aux_target,
        intermediate_preds=intermediate,
    )

    # Scale loss for gradient accumulation
    loss = loss / accumulation_steps

    # Backward pass
    if use_amp and scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    # Update weights at the end of accumulation
    if (step_in_accumulation + 1) % accumulation_steps == 0:
        if use_amp and scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad()

    return metrics


def validate_model(
    model: NeuralForecaster,
    val_loader: DataLoader,
    device: torch.device,
    dataset: NeuralForecastDataset,
) -> Tuple[float, float]:
    """Validate model and return both normalized and competition-scale MSE."""
    model.eval()

    total_mse_norm = 0.0
    total_mse_raw = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model.predict(batch)
            target = batch[:, :, :, 0]

            # Normalized MSE
            mse_norm = F.mse_loss(pred[:, 10:, :], target[:, 10:, :], reduction="mean")
            total_mse_norm += mse_norm.item()

            # Denormalize for competition-scale MSE
            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()

            # Denormalize (need to handle the shape properly)
            B, T, C = pred_np.shape
            pred_flat = pred_np.reshape(B * T, C)
            target_flat = target_np.reshape(B * T, C)

            # Use only feature 0 stats
            avg_f0 = dataset.average[:, :C]
            std_f0 = dataset.std[:, :C]

            pred_denorm = denormalize(pred_flat, avg_f0, std_f0).reshape(B, T, C)
            target_denorm = denormalize(target_flat, avg_f0, std_f0).reshape(B, T, C)

            mse_raw = np.mean((pred_denorm[:, 10:, :] - target_denorm[:, 10:, :]) ** 2)
            total_mse_raw += mse_raw

            n_batches += 1

    return total_mse_norm / n_batches, total_mse_raw / n_batches


def train_model(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    monkey_name: str,
    device: torch.device,
    batch_size: int = 8,
    accumulation_steps: int = 4,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    memory_efficient: bool = True,
    use_amp: bool = False,
    save_dir: str = "./checkpoints",
) -> NeuralForecaster:
    """Train model for a specific monkey."""

    print(f"Data shapes for {monkey_name}:")
    print(f"  Train: {train_data.shape}")
    print(f"  Val: {val_data.shape}")
    print(f"  Test: {test_data.shape}")

    n_channels = train_data.shape[2]
    print(f"Detected {n_channels} channels")

    # Create datasets
    train_dataset = NeuralForecastDataset(train_data, augment=True)
    val_dataset = NeuralForecastDataset(
        val_data, train_dataset.average, train_dataset.std
    )
    test_dataset = NeuralForecastDataset(
        test_data, train_dataset.average, train_dataset.std
    )

    # Save normalization stats
    os.makedirs(save_dir, exist_ok=True)
    np.savez(
        os.path.join(save_dir, f"norm_stats_{monkey_name}.npz"),
        average=train_dataset.average,
        std=train_dataset.std,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = create_model(n_channels, device, memory_efficient=memory_efficient)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    loss_fn = EnhancedForecastingLoss(
        aux_weight=0.3,
        consistency_weight=0.1,
        intermediate_weight=0.2,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01, betas=(0.9, 0.999)
    )

    # Learning rate scheduler with warmup
    warmup_epochs = 5

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 0.5 * (
                1
                + np.cos(np.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs))
            )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Training loop
    best_val_mse = float("inf")
    patience = 15
    patience_counter = 0

    print(f"\nStarting training for {monkey_name}...")
    print("=" * 60)

    for epoch in range(num_epochs):
        model.train()
        epoch_metrics = {"total": 0, "mse": 0, "temporal": 0}
        n_steps = 0

        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            metrics = train_step_with_accumulation(
                model=model,
                batch=batch,
                loss_fn=loss_fn,
                optimizer=optimizer,
                accumulation_steps=accumulation_steps,
                step_in_accumulation=step,
                device=device,
                use_amp=use_amp,
                scaler=scaler,
            )

            for k, v in metrics.items():
                if k in epoch_metrics:
                    epoch_metrics[k] += v
            n_steps += 1

        # Average metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= max(n_steps, 1)

        # Validation
        val_mse_norm, val_mse_raw = validate_model(
            model, val_loader, device, train_dataset
        )

        # Update scheduler
        scheduler.step()

        # Logging
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:3d}/{num_epochs} | "
            f"Train Loss: {epoch_metrics['total']:.6f} | "
            f"Val MSE (norm): {val_mse_norm:.6f} | "
            f"Val MSE (raw): {val_mse_raw:.2f} | "
            f"LR: {current_lr:.2e}"
        )

        # Early stopping check
        if val_mse_norm < best_val_mse:
            best_val_mse = val_mse_norm
            patience_counter = 0

            # Save best model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mse": val_mse_norm,
                    "n_channels": n_channels,
                },
                os.path.join(save_dir, f"best_model_{monkey_name}.pth"),
            )
            print(f"  -> Saved new best model (Val MSE: {val_mse_norm:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # Load best model
    checkpoint = torch.load(
        os.path.join(save_dir, f"best_model_{monkey_name}.pth"),
        weights_only=False,  # Required for checkpoints containing numpy scalars
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # Final test evaluation
    test_mse_norm, test_mse_raw = validate_model(
        model, test_loader, device, train_dataset
    )
    print(f"\nFinal Test MSE (norm): {test_mse_norm:.6f}")
    print(f"Final Test MSE (raw): {test_mse_raw:.2f}")

    return model


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train Neural Forecasting Model")
    parser.add_argument(
        "data_dir",
        type=str,
        nargs="?",
        default="./hackdata/neural-forecasting/public_data/",
        help="Path to data directory",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
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

    # System info
    total_mem, avail_mem = get_memory_info()
    print(f"System RAM: {total_mem:.1f} GB total, {avail_mem:.1f} GB available")

    # Auto-configure based on available memory
    if args.batch_size is None:
        if avail_mem > 16:
            batch_size = 8
            accumulation_steps = 4
        elif avail_mem > 8:
            batch_size = 4
            accumulation_steps = 8
        else:
            batch_size = 2
            accumulation_steps = 16
    else:
        batch_size = args.batch_size
        accumulation_steps = max(1, 32 // batch_size)

    print(f"Batch size: {batch_size}, Grad accum: {accumulation_steps}")
    print(f"Effective batch size: {batch_size * accumulation_steps}")

    # Memory efficient mode for low memory systems
    # memory_efficient = avail_mem < 16
    # use_amp = torch.cuda.is_available() and avail_mem < 12
    use_amp = True
    memory_efficient = True
    print(f"Memory efficient: {memory_efficient}, AMP: {use_amp}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train models
    monkeys = ["affi", "beignet"] if args.monkey == "all" else [args.monkey]

    for monkey in monkeys:
        print("\n" + "=" * 60)
        print(f"Training model for {monkey}")
        print("=" * 60)

        try:
            train_data, val_data, test_data = load_data(args.data_dir, monkey)

            model = train_model(
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                monkey_name=monkey,
                device=device,
                batch_size=batch_size,
                accumulation_steps=accumulation_steps,
                num_epochs=args.epochs,
                learning_rate=args.lr,
                memory_efficient=memory_efficient,
                use_amp=use_amp,
                save_dir=args.save_dir,
            )

            print(f"\nCompleted training for {monkey}")

        except Exception as e:
            print(f"Error training {monkey}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
