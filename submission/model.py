import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------
# 1. Architecture Classes (Must match ntrainer.py exactly)
# -----------------------------------------------------------------------------
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
        pred = pred.reshape(b, c, 10, 1).permute(0, 2, 1, 3).squeeze(-1)
        return pred


# -----------------------------------------------------------------------------
# 2. Helper Classes
# -----------------------------------------------------------------------------
def normalize(data, average=[], std=[]):
    original_shape = data.shape
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
    denominator = combine_max - combine_min
    denominator = np.where(denominator == 0, 1e-9, denominator)
    norm_data_reshaped = 2 * (data_reshaped - combine_min) / denominator - 1

    if len(original_shape) == 4:
        norm_data = norm_data_reshaped.reshape(original_shape)
    else:
        norm_data = norm_data_reshaped

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


# -----------------------------------------------------------------------------
# 3. Submission Interface (Matches trivial_submission.py)
# -----------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, monkey_name):
        super().__init__()
        self.monkey_name = monkey_name

        # 1. Load Stats immediately
        base = os.path.dirname(__file__)
        stats_path = os.path.join(base, f"train_data_average_std_{monkey_name}.npz")
        stats = np.load(stats_path)
        self.average = stats["average"]
        self.std = stats["std"]

        # 2. Initialize Model Architecture
        if monkey_name == "affi":
            self.channel_size = 239
        else:
            self.channel_size = 87

        self.model = ResidualBiGRU(channel_size=self.channel_size)
        self.model.to(device)

        # 3. Load Weights immediately (Standard practice for torch submissions)
        model_path = os.path.join(base, f"model_{monkey_name}.pth")
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def load(self):
        """
        Required by the competition interface.
        Since we loaded weights in __init__, we pass here.
        """
        pass

    def predict(self, x):
        """
        x: (Batch, 20, Channel, 9) -> The full input from the test file
        Returns: (Batch, 20, Channel) -> First 10 steps from input + 10 predicted steps
        """
        # The model should predict using the first 10 time steps.
        model_input = x[:, :10, :, :]

        # The final output needs the first 10 steps of the target feature (feature 0)
        output_prefix = x[:, :10, :, 0]

        # Normalize the 10-step input and create a dataset
        dataset = NeuroForcastDataset(model_input, average=self.average, std=self.std)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

        # Run predictions
        predictions_list = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                output = self.model(batch)  # Model outputs 10 predicted steps
                predictions_list.append(output.cpu().numpy())

        predicted_suffix = np.concatenate(predictions_list, axis=0)

        # Combine the input prefix with the predicted suffix
        full_output = np.concatenate([output_prefix, predicted_suffix], axis=1)

        return full_output
