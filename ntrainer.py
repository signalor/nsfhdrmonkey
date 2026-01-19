import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Import the model architecture inline to ensure this script is standalone
# (Paste the RevIN, AttentionBlock, and ResidualBiGRU classes here exactly as in model.py)


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + 1e-10)
        x = x * self.stdev + self.mean
        return x


class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        weights = self.attention(x)
        context = torch.sum(weights * x, dim=1)
        return context


class ResidualBiGRU(nn.Module):
    def __init__(
        self, channel_size, feature_size=9, hidden_dim=128, num_layers=2, dropout=0.2
    ):
        super(ResidualBiGRU, self).__init__()
        self.revin = RevIN(num_features=1, affine=True)
        self.input_proj = nn.Linear(feature_size, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = AttentionBlock(hidden_dim * 2)
        self.output_proj = nn.Linear(hidden_dim * 2, 10)

    def forward(self, x):
        b, t, c, f = x.shape
        x_reshaped = x.permute(0, 2, 1, 3).reshape(b * c, t, f)
        x_emb = self.input_proj(x_reshaped)
        gru_out, _ = self.gru(x_emb)
        context = self.attention(gru_out)
        pred = self.output_proj(context)
        pred = pred.unsqueeze(-1)
        pred = pred.reshape(b, c, t, 1).permute(0, 2, 1, 3).squeeze(-1)
        return pred


# -----------------------------------------------------------------------------
# Data Loading & Utils
# -----------------------------------------------------------------------------
def normalize(data, average=[], std=[]):
    if data.ndim == 4:
        n, t, c, f = data.shape
        data_reshaped = data.reshape((n * t, -1))
    else:
        data_reshaped = data
    if len(average) == 0:
        average = np.mean(data_reshaped, axis=0, keepdims=True)
        std = np.std(data_reshaped, axis=0, keepdims=True)
    combine_max = average + 4 * std
    combine_min = average - 4 * std
    std_safe = np.where(std == 0, 1e-9, std)
    denominator = combine_max - combine_min
    denominator = np.where(denominator == 0, 1e-9, denominator)
    norm_data = 2 * (data_reshaped - combine_min) / denominator - 1
    if data.ndim == 4:
        norm_data = norm_data.reshape((n, t, c, f))
    return norm_data, average, std


class NeuroForcastDataset(Dataset):
    def __init__(self, neural_data, average=[], std=[]):
        self.data = neural_data
        self.norm_data, self.average, self.std = normalize(neural_data, average, std)
        self.norm_data = torch.from_numpy(self.norm_data).float()

    def __len__(self):
        return len(self.norm_data)

    def __getitem__(self, idx):
        return self.norm_data[idx]


def load_and_combine_data(file_paths):
    all_neural_data = []
    for filename in file_paths:
        data = np.load(filename)
        keys = list(data.keys())
        neural_data = data["neural_data"] if "neural_data" in keys else data[keys[0]]
        if neural_data.ndim != 4:
            print(
                f"Warning: Skipping {filename}. Expected 4D data, but got shape {neural_data.shape}."
            )
            continue
        all_neural_data.append(neural_data)
    combined_data = np.concatenate(all_neural_data, axis=0)
    idx = np.random.permutation(len(combined_data))
    combined_data = combined_data[idx]
    n_train = int(len(combined_data) * 0.8)
    n_test = int(len(combined_data) * 0.1)
    return (
        combined_data[:n_train],
        combined_data[n_train : n_train + n_test],
        combined_data[n_train + n_test :],
    )


def save_comparison_charts(model, dataset, device, charts_dir, monkey_name):
    model.eval()
    if not os.path.exists(charts_dir):
        os.makedirs(charts_dir)
    input_tensor = dataset[0].unsqueeze(0).to(device)
    input_seq = input_tensor[:, :10, :, :]
    target_full = input_tensor[:, :, :, 0].cpu().numpy().squeeze()
    with torch.no_grad():
        pred_seq = model(input_seq).cpu().numpy().squeeze()

    num_channels = min(9, target_full.shape[1])
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()
    time_input = np.arange(0, 10)
    time_pred = np.arange(10, 20)
    for i in range(num_channels):
        ax = axes[i]
        ax.plot(time_input, target_full[:10, i], label="Input", color="gray")
        ax.plot(time_pred, target_full[10:, i], label="Target", color="blue")
        ax.plot(
            time_pred, pred_seq[:, i], label="Prediction", color="red", linestyle="--"
        )
        ax.axvline(x=10, color="black", linestyle=":", alpha=0.5)
        ax.set_title(f"Ch {i}")
        if i == 0:
            ax.legend()
    plt.suptitle(f"{monkey_name} (Bi-GRU + Attention)")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, f"{monkey_name}_forecast.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--charts_dir", type=str, default="charts")
    parser.add_argument(
        "--monkey_name",
        type=str,
        default="affi",
        help="Name of the monkey to train (e.g., 'affi'). Trains all if not specified.",
    )
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--info", action="store_true", help="Log model info and exit without training."
    )
    args = parser.parse_args()
    print(args)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.charts_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    files = [
        f
        for f in os.listdir(args.data_dir)
        if f.endswith(".npz")
        and "test" not in f
        and "val" not in f
        and "masked" not in f
    ]

    monkey_files = {"affi": [], "beignet": []}
    for f in files:
        if "affi" in f.lower():
            monkey_files["affi"].append(os.path.join(args.data_dir, f))
        elif "beignet" in f.lower():
            monkey_files["beignet"].append(os.path.join(args.data_dir, f))

    if args.monkey_name:
        monkey_to_train = args.monkey_name.lower()
        if monkey_to_train not in monkey_files or not monkey_files[monkey_to_train]:
            print(f"Warning: No files found for monkey '{args.monkey_name}'. Skipping.")
            monkey_files = {}
        else:
            monkey_files = {monkey_to_train: monkey_files[monkey_to_train]}

    for monkey, file_paths in monkey_files.items():
        if not file_paths:
            continue
        print(f"\nTraining Subject: {monkey.upper()}")

        train_raw, test_raw, val_raw = load_and_combine_data(file_paths)
        train_dataset = NeuroForcastDataset(train_raw)
        avg, std = train_dataset.average, train_dataset.std
        np.savez(
            os.path.join(args.save_dir, f"train_data_average_std_{monkey}.npz"),
            average=avg,
            std=std,
        )
        val_dataset = NeuroForcastDataset(val_raw, average=avg, std=std)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        _, _, n_channels, n_features = train_raw.shape
        model = ResidualBiGRU(channel_size=n_channels, feature_size=n_features)
        model.to(device)

        if args.info:
            print(f"\n--- Model Information for {monkey.upper()} ---")
            print(model)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            print(f"\nTotal parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print("--------------------------------------------------")
            continue

        if args.multi_gpu and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device, non_blocking=True)
                input_seq = batch[:, :10, :, :]
                target_seq = batch[:, 10:, :, 0]

                optimizer.zero_grad()
                output = model(input_seq)
                loss = criterion(output, target_seq)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                running_loss += loss.item() * batch.size(0)

            train_loss = running_loss / len(train_loader.dataset)

            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device, non_blocking=True)
                    input_seq = batch[:, :10, :, :]
                    target_seq = batch[:, 10:, :, 0]
                    output = model(input_seq)
                    loss = criterion(output, target_seq)
                    val_running_loss += loss.item() * batch.size(0)
            val_loss = val_running_loss / len(val_loader.dataset)

            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                state = (
                    model.module.state_dict()
                    if isinstance(model, nn.DataParallel)
                    else model.state_dict()
                )
                torch.save(state, os.path.join(args.save_dir, f"model_{monkey}.pth"))
                print(f"  * Best model saved (Ep {epoch + 1})")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print("Early stopping.")
                    break

            print(f"Epoch {epoch + 1} | Train: {train_loss:.5f} | Val: {val_loss:.5f}")

        model.load_state_dict(
            torch.load(
                os.path.join(args.save_dir, f"model_{monkey}.pth"), weights_only=True
            )
        )
        save_comparison_charts(model, val_dataset, device, args.charts_dir, monkey)


if __name__ == "__main__":
    main()
