"""
Training script for Neural Forecasting Competition.
Includes:
- Proper train/val/test splits
- Learning rate scheduling with warmup
- Early stopping
- Model checkpointing
- Logging and visualization
"""

import gc
import json
import os
import sys
from datetime import datetime
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import (
    ForecastingLoss,
    NeuralForecaster,
    create_model,
    validate,
    validate_competition,
)

# =============================================================================
# Training Step with Gradient Accumulation
# =============================================================================


def train_step_with_accumulation(
    model: NeuralForecaster,
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: ForecastingLoss,
    device: torch.device,
    grad_accum_steps: int,
    batch_idx: int,
    total_batches: int,
) -> dict:
    """
    Training step with gradient accumulation for memory efficiency.
    Only updates weights every grad_accum_steps batches.
    """
    batch = batch.to(device)

    # Forward pass with augmentation
    x_input = batch[:, :10, :, :]
    x_input_aug = model.augmentation.channel_dropout(x_input, p=0.1)
    x_input_aug = model.augmentation.time_noise(x_input_aug, std=0.03)

    # Forward through inner model (may be wrapped with DataParallel)
    main_pred, aux_pred = model.model(x_input_aug)

    # Targets
    target = batch[:, 10:, :, 0]  # Feature 0 of future timesteps
    target_full = batch[:, 10:, :, :]  # All features

    # Compute loss (scaled by accumulation steps)
    loss, metrics = loss_fn(main_pred, aux_pred, target, target_full)
    loss = loss / grad_accum_steps

    # Backward pass (gradients accumulate)
    loss.backward()

    # Only step optimizer every grad_accum_steps or at end of epoch
    if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == total_batches:
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Clear cache periodically to prevent fragmentation
        if torch.cuda.is_available() and batch_idx % (grad_accum_steps * 10) == 0:
            torch.cuda.empty_cache()

    return metrics


# =============================================================================
# Dataset
# =============================================================================


class NeuralForecastDataset(Dataset):
    """Dataset for neural forecasting."""

    def __init__(
        self,
        data: np.ndarray,
        normalize: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        """
        Args:
            data: (N, T, C, F) neural data
            normalize: whether to normalize
            mean, std: precomputed statistics for test set
        """
        self.data = data.astype(np.float32)
        self.normalize = normalize

        if normalize:
            if mean is None or std is None:
                # Compute statistics from this data
                self.mean, self.std = self._compute_stats(self.data)
            else:
                self.mean = mean
                self.std = std

            self.data = self._normalize(self.data)

    def _compute_stats(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and std across samples and time."""
        # Shape: (N, T, C, F) -> compute stats over N and T
        mean = np.mean(data, axis=(0, 1), keepdims=True)  # (1, 1, C, F)
        std = np.std(data, axis=(0, 1), keepdims=True) + 1e-8
        return mean, std

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize to roughly [-1, 1] range."""
        return (data - self.mean) / (4 * self.std)  # 4*std to avoid extreme values

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.data[idx])


def load_data(data_dir: str, monkey_name: str) -> Tuple[np.ndarray, ...]:
    """Load and split data for a specific monkey."""

    filepath = os.path.join(data_dir, f"train_data_{monkey_name}.npz")
    data = np.load(filepath)["arr_0"]

    # Shuffle with fixed seed for reproducibility
    np.random.seed(42)
    indices = np.random.permutation(len(data))
    data = data[indices]

    # Split: 80% train, 10% val, 10% test
    n = len(data)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    print(f"Data shapes for {monkey_name}:")
    print(f"  Train: {train_data.shape}")
    print(f"  Val: {val_data.shape}")
    print(f"  Test: {test_data.shape}")

    return train_data, val_data, test_data


def load_competition_test_data(
    data_dir: str, monkey_name: str, train_mean: np.ndarray, train_std: np.ndarray
) -> dict:
    """
    Load all test datasets for competition evaluation.

    Returns:
        dict of DataLoaders for each test dataset
    """
    test_loaders = {}

    # Base test dataset naming patterns
    if monkey_name == "affi":
        test_files = [
            ("affi", f"test_data_{monkey_name}.npz"),
            ("affi_d2", f"test_data_{monkey_name}_d2.npz"),
        ]
    else:  # beignet
        test_files = [
            ("beignet", f"test_data_{monkey_name}.npz"),
            ("beignet_d2", f"test_data_{monkey_name}_d2.npz"),
            ("beignet_d3", f"test_data_{monkey_name}_d3.npz"),
        ]

    for dataset_name, filename in test_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"Loading {dataset_name} test data...")
            data = np.load(filepath)["arr_0"]

            # Create dataset with training normalization stats
            dataset = NeuralForecastDataset(
                data, normalize=True, mean=train_mean, std=train_std
            )

            # Create dataloader
            loader = DataLoader(
                dataset,
                batch_size=32,  # Use larger batch size for inference
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )

            test_loaders[dataset_name] = loader
            print(f"  Loaded {len(dataset)} samples")
        else:
            print(f"Warning: {filename} not found, skipping {dataset_name}")

    return test_loaders


# =============================================================================
# Learning Rate Scheduler with Warmup
# =============================================================================


class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup and cosine decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        base_lr: float,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def _get_lr(self) -> float:
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            return self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine decay
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + np.cos(np.pi * progress)
            )


# =============================================================================
# Early Stopping
# =============================================================================


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# =============================================================================
# Training Loop
# =============================================================================


def train_model(
    monkey_name: str, data_dir: str, save_dir: str, device: torch.device, config: dict
) -> NeuralForecaster:
    """Complete training pipeline."""

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    train_data, val_data, test_data = load_data(data_dir, monkey_name)

    # Create datasets
    train_dataset = NeuralForecastDataset(train_data, normalize=True)
    val_dataset = NeuralForecastDataset(
        val_data, normalize=True, mean=train_dataset.mean, std=train_dataset.std
    )
    test_dataset = NeuralForecastDataset(
        test_data, normalize=True, mean=train_dataset.mean, std=train_dataset.std
    )

    # Save normalization stats
    np.savez(
        os.path.join(save_dir, f"norm_stats_{monkey_name}.npz"),
        mean=train_dataset.mean,
        std=train_dataset.std,
    )

    # Free raw data arrays to save memory
    del train_data, val_data, test_data
    gc.collect()

    # Create data loaders with memory-safe settings
    pin_memory = config.get("pin_memory", False)
    num_workers = config.get("num_workers", 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Detect number of channels from dataset
    sample = train_dataset[0]
    n_channels = sample.shape[1]  # Shape is (T, C, F)
    print(f"Detected {n_channels} channels")

    # Create model
    memory_efficient = config.get("memory_efficient", False)
    model = create_model(n_channels, device, memory_efficient=memory_efficient)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    if memory_efficient:
        print("Using memory-efficient model configuration")

    # Wrap inner model with DataParallel if multiple GPUs available
    num_gpus = config.get("num_gpus", 1)
    use_data_parallel = num_gpus > 1 and torch.cuda.is_available()
    if use_data_parallel:
        print(f"Using DataParallel across {num_gpus} GPUs")
        model.model = torch.nn.DataParallel(model.model)

    # Loss function
    loss_fn = ForecastingLoss(
        aux_weight=config["aux_weight"], consistency_weight=config["consistency_weight"]
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.98),
    )

    # Scheduler
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=config["warmup_epochs"],
        total_epochs=config["num_epochs"],
        base_lr=config["learning_rate"],
        min_lr=config["min_lr"],
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=config["patience"])

    # Training history
    history = {"train_loss": [], "val_loss": [], "learning_rate": []}

    best_val_loss = float("inf")

    print(f"\nStarting training for {monkey_name}...")
    print("=" * 60)

    grad_accum_steps = config.get("grad_accum_steps", 1)

    for epoch in range(config["num_epochs"]):
        # Training with gradient accumulation
        model.train()
        epoch_losses = []
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            metrics = train_step_with_accumulation(
                model,
                batch,
                optimizer,
                loss_fn,
                device,
                grad_accum_steps,
                batch_idx,
                len(train_loader),
            )
            epoch_losses.append(metrics["total_loss"])

        avg_train_loss = np.mean(epoch_losses)

        # Validation
        val_loss = validate(model, val_loader, device)

        # Update learning rate
        lr = scheduler.step()

        # Record history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(lr)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Handle DataParallel wrapper on inner model - save unwrapped state
            inner_model = model.model
            if hasattr(inner_model, "module"):
                inner_state = inner_model.module.state_dict()
            else:
                inner_state = inner_model.state_dict()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": inner_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": config,
                },
                os.path.join(save_dir, f"best_model_{monkey_name}.pt"),
            )

        # Print progress
        if epoch % 5 == 0 or epoch == config["num_epochs"] - 1:
            print(
                f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | LR: {lr:.2e}"
            )

        # Early stopping check
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    # Load best model for final evaluation
    checkpoint = torch.load(
        os.path.join(save_dir, f"best_model_{monkey_name}.pt"), weights_only=False
    )
    # Handle DataParallel wrapper on inner model when loading
    inner_model = model.model
    if hasattr(inner_model, "module"):
        inner_model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        inner_model.load_state_dict(checkpoint["model_state_dict"])

    # ========================================================================
    # COMPETITION EVALUATION - Multiple test datasets
    # ========================================================================
    print("\n" + "=" * 60)
    print("Competition Evaluation")
    print("=" * 60)

    # Load all competition test datasets
    test_loaders = load_competition_test_data(
        data_dir, monkey_name, train_dataset.mean, train_dataset.std
    )

    if test_loaders:
        # Run competition validation
        comp_results = validate_competition(model, test_loaders, device)

        # Print results in competition format
        print(f"\nCompetition Results for {monkey_name}:")
        for key, value in comp_results.items():
            if key.startswith("MSE_"):
                dataset_name = key.replace("MSE_", "")
                print(f"  {key}: {value:.2f}")
        print(f"  Total MSR: {comp_results['Total_MSR']:.2f}")

        # Save competition results
        with open(
            os.path.join(save_dir, f"competition_results_{monkey_name}.json"), "w"
        ) as f:
            json.dump(comp_results, f, indent=2)
    else:
        print("No additional test datasets found for competition evaluation")
        print("Using only validation set evaluation:")
        test_loss = validate(model, test_loader, device)
        print(f"Test MSE: {test_loss:.6f}")

    # Save training history
    with open(os.path.join(save_dir, f"history_{monkey_name}.json"), "w") as f:
        json.dump(history, f)

    # Plot training curves
    plot_training_curves(history, save_dir, monkey_name)

    return model


def plot_training_curves(history: dict, save_dir: str, monkey_name: str):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{monkey_name} - Training Curves")
    axes[0].legend()
    axes[0].set_yscale("log")

    # Learning rate
    axes[1].plot(history["learning_rate"])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("Learning Rate Schedule")
    axes[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"training_curves_{monkey_name}.png"), dpi=150)
    plt.close()


# =============================================================================
# Evaluation and Visualization
# =============================================================================


def evaluate_and_visualize(
    model: NeuralForecaster,
    test_loader: DataLoader,
    device: torch.device,
    save_dir: str,
    monkey_name: str,
    n_samples: int = 5,
):
    """Evaluate model and create visualizations."""

    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model.predict(batch)
            predictions.append(pred.cpu().numpy())
            targets.append(batch[:, :, :, 0].cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    # Compute metrics
    future_preds = predictions[:, 10:, :]
    future_targets = targets[:, 10:, :]

    mse = np.mean((future_preds - future_targets) ** 2)
    mae = np.mean(np.abs(future_preds - future_targets))

    # R2 score per channel
    ss_res = np.sum((future_targets - future_preds) ** 2, axis=(0, 1))
    ss_tot = np.sum(
        (future_targets - np.mean(future_targets, axis=(0, 1), keepdims=True)) ** 2,
        axis=(0, 1),
    )
    r2_per_channel = 1 - ss_res / (ss_tot + 1e-8)

    print(f"\nEvaluation Results for {monkey_name}:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Mean R2: {np.mean(r2_per_channel):.4f}")
    print(f"  Median R2: {np.median(r2_per_channel):.4f}")

    # Visualize sample predictions
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 3 * n_samples))

    for i in range(n_samples):
        sample_idx = i * (len(predictions) // n_samples)

        for j, ch in enumerate([0, targets.shape[2] // 2, targets.shape[2] - 1]):
            ax = axes[i, j]

            # Plot ground truth
            ax.plot(
                range(20),
                targets[sample_idx, :, ch],
                "b-",
                label="Ground Truth",
                linewidth=2,
            )

            # Plot prediction (future part)
            ax.plot(
                range(10, 20),
                predictions[sample_idx, 10:, ch],
                "r--",
                label="Prediction",
                linewidth=2,
            )

            # Vertical line at prediction start
            ax.axvline(x=10, color="gray", linestyle=":", alpha=0.7)

            ax.set_xlabel("Time Step")
            ax.set_ylabel("Signal")
            ax.set_title(f"Sample {sample_idx}, Channel {ch}")
            if i == 0 and j == 0:
                ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"predictions_{monkey_name}.png"), dpi=150)
    plt.close()

    return {"mse": mse, "mae": mae, "r2_mean": np.mean(r2_per_channel)}


# =============================================================================
# Main
# =============================================================================


def get_memory_limit_config() -> dict:
    """
    Detect available memory and return appropriate configuration.
    Returns conservative settings to prevent OOM crashes.
    """
    try:
        import psutil

        # Get available system RAM in GB
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        print(
            f"System RAM: {total_ram_gb:.1f} GB total, {available_ram_gb:.1f} GB available"
        )
    except ImportError:
        print("Warning: psutil not installed. Using conservative memory defaults.")
        print("Install with: pip install psutil")
        available_ram_gb = 8  # Assume modest system
        total_ram_gb = 16

    # Check GPU memory if available
    gpu_memory_gb = 0
    num_gpus = 0
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPUs available: {num_gpus}")
        for i in range(num_gpus):
            mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(
                f"  GPU {i}: {torch.cuda.get_device_properties(i).name} ({mem:.1f} GB)"
            )

    # Determine batch size and gradient accumulation based on available memory
    # Target effective batch size of 32, achieved via smaller batches + accumulation
    if torch.cuda.is_available():
        # GPU-based limits - be conservative to avoid OOM
        # Base batch size per GPU
        if gpu_memory_gb >= 24:
            batch_size_per_gpu = 16
        elif gpu_memory_gb >= 16:
            batch_size_per_gpu = 8
        elif gpu_memory_gb >= 10:
            batch_size_per_gpu = 4
        elif gpu_memory_gb >= 6:
            batch_size_per_gpu = 2
        else:
            batch_size_per_gpu = 1

        # Scale batch size by number of GPUs
        batch_size = batch_size_per_gpu * max(1, num_gpus)

        # Adjust gradient accumulation to maintain effective batch size of ~32
        effective_target = 32
        grad_accum_steps = max(1, effective_target // batch_size)
    else:
        # CPU-based limits
        if available_ram_gb >= 32:
            batch_size = 8
            grad_accum_steps = 4
        elif available_ram_gb >= 16:
            batch_size = 4
            grad_accum_steps = 8
        else:
            batch_size = 2
            grad_accum_steps = 16

    # Determine if we should use pin_memory (only if enough RAM)
    pin_memory = torch.cuda.is_available() and available_ram_gb >= 8

    # Determine number of workers (0 is safest for memory)
    num_workers = 0

    # Use memory-efficient model for GPUs with less than 24GB
    # The full model needs ~18GB+ for forward/backward pass
    memory_efficient = gpu_memory_gb < 24 if torch.cuda.is_available() else True

    print(
        f"Using batch_size={batch_size} with {grad_accum_steps} gradient accumulation steps"
    )
    print(f"Effective batch size: {batch_size * grad_accum_steps}")
    if memory_efficient:
        print("Memory-efficient model enabled (reduced d_model, layers)")

    return {
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
        "memory_efficient": memory_efficient,
        "num_gpus": num_gpus,
        "available_ram_gb": available_ram_gb,
        "gpu_memory_gb": gpu_memory_gb,
    }


def main():
    # Detect memory and get safe configuration
    mem_config = get_memory_limit_config()
    print(f"Using batch_size={mem_config['batch_size']} based on available memory")

    # Configuration
    config = {
        # Data - use detected batch size
        "batch_size": mem_config["batch_size"],
        "grad_accum_steps": mem_config["grad_accum_steps"],
        # Training
        "num_epochs": 50,
        "learning_rate": 3e-4,
        "min_lr": 1e-6,
        "weight_decay": 0.01,
        "warmup_epochs": 10,
        "patience": 10,
        # Loss weights
        "aux_weight": 0.3,
        "consistency_weight": 0.05,
        # Memory settings
        "pin_memory": mem_config["pin_memory"],
        "num_workers": mem_config["num_workers"],
        "memory_efficient": mem_config["memory_efficient"],
    }

    # Paths
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./public_data/"
    save_dir = "./checkpoints/"

    # Device setup with memory limits
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Limit GPU memory growth to prevent OOM
        # Reserve some memory for system/other processes
        torch.cuda.set_per_process_memory_fraction(0.85)
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Train for both monkeys
    for monkey_name in ["affi", "beignet"]:
        print(f"\n{'=' * 60}")
        print(f"Training model for {monkey_name}")
        print("=" * 60)

        model = train_model(
            monkey_name=monkey_name,
            data_dir=data_dir,
            save_dir=save_dir,
            device=device,
            config=config,
        )

        # Clean up memory between monkeys to prevent accumulation
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
