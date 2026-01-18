"""
Submission model for Neural Forecasting Competition.
Spatial-Temporal Transformer with memory-efficient configuration.
"""

import math
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Normalization
# =============================================================================


def normalize(data, mean=None, std=None):
    """
    Normalize input data using training-compatible normalization.
    Uses: (data - mean) / (4 * std) to give roughly [-1, 1] range.
    """
    if data.ndim == 4:
        n, t, c, f = data.shape
    else:
        raise ValueError(f"Expected 4D data, got {data.ndim}D")

    if mean is None or std is None:
        # Compute stats across samples and time: (1, 1, C, F)
        mean = np.mean(data, axis=(0, 1), keepdims=True)
        std = np.std(data, axis=(0, 1), keepdims=True) + 1e-8

    norm_data = (data - mean) / (4 * std)
    return norm_data, mean, std


def denormalize(data, mean, std):
    """
    Denormalize data back to original scale.
    Reverses: (data - mean) / (4 * std)
    """
    return data * (4 * std) + mean


class NeuroForcastDataset(Dataset):
    """Dataset for neural forecasting with normalization."""

    def __init__(self, neural_data, use_graph=False, mean=None, std=None):
        self.data = neural_data.astype(np.float32)
        self.use_graph = use_graph

        if mean is None or std is None:
            self.data, self.mean, self.std = normalize(self.data)
        else:
            self.data, self.mean, self.std = normalize(self.data, mean, std)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        if not self.use_graph:
            data = data[:, :, 0]
        data = torch.tensor(data, dtype=torch.float32)
        return data


# =============================================================================
# Model Components
# =============================================================================


class RevIN(nn.Module):
    """Reversible Instance Normalization."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x: torch.Tensor):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        x = x * self.stdev + self.mean
        return x


class ConvTokenizer(nn.Module):
    """Multi-scale convolutional tokenizer."""

    def __init__(self, in_channels: int, d_model: int, kernel_sizes: list = [3, 5, 7]):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        d_model // len(kernel_sizes),
                        kernel_size=k,
                        padding=k // 2,
                        groups=1,
                    ),
                    nn.GELU(),
                    nn.BatchNorm1d(d_model // len(kernel_sizes)),
                )
                for k in kernel_sizes
            ]
        )
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        conv_outputs = [conv(x) for conv in self.convs]
        x = torch.cat(conv_outputs, dim=1)
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.norm(x)
        return x


class SpatialAttention(nn.Module):
    """Attention across channels at each timestep."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, D = x.shape
        x_flat = x.reshape(B * T, C, D)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        attn_out = self.dropout(attn_out)
        x_flat = self.norm(x_flat + attn_out)
        return x_flat.reshape(B, T, C, D)


class TemporalAttention(nn.Module):
    """Attention across time for each channel."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, C, D = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(B * C, T, D)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat, attn_mask=mask)
        attn_out = self.dropout(attn_out)
        x_flat = self.norm(x_flat + attn_out)
        return x_flat.reshape(B, C, T, D).permute(0, 2, 1, 3)


class FeedForward(nn.Module):
    """Standard transformer feed-forward block."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class SpatialTemporalBlock(nn.Module):
    """Single block of factorized spatial-temporal attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.spatial_attn = SpatialAttention(d_model, n_heads, dropout)
        self.temporal_attn = TemporalAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(
        self, x: torch.Tensor, temporal_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.spatial_attn(x)
        x = self.temporal_attn(x, temporal_mask)
        B, T, C, D = x.shape
        x = x.reshape(B * T * C, D)
        x = self.ff(x.unsqueeze(1)).squeeze(1)
        x = x.reshape(B, T, C, D)
        return x


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding for both time and space."""

    def __init__(self, max_time: int, max_channels: int, d_model: int):
        super().__init__()
        self.time_embed = nn.Parameter(torch.randn(1, max_time, 1, d_model) * 0.02)
        self.channel_embed = nn.Parameter(
            torch.randn(1, 1, max_channels, d_model) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, D = x.shape
        return x + self.time_embed[:, :T, :, :] + self.channel_embed[:, :, :C, :]


class FutureQueryDecoderLayer(nn.Module):
    """Single decoder layer with self-attention and cross-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        B, T_fut, C, D = x.shape
        _, T_enc, _, _ = encoder_output.shape
        x_flat = x.reshape(B, T_fut * C, D)
        enc_flat = encoder_output.reshape(B, T_enc * C, D)
        attn_out, _ = self.self_attn(x_flat, x_flat, x_flat)
        x_flat = self.norm1(x_flat + self.dropout(attn_out))
        attn_out, _ = self.cross_attn(x_flat, enc_flat, enc_flat)
        x_flat = self.norm2(x_flat + self.dropout(attn_out))
        x_flat = self.norm3(x_flat + self.ff(x_flat))
        return x_flat.reshape(B, T_fut, C, D)


class FutureQueryDecoder(nn.Module):
    """Decoder that uses learned queries to predict future timesteps."""

    def __init__(
        self,
        n_future: int,
        n_channels: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_future = n_future
        self.n_channels = n_channels
        self.future_queries = nn.Parameter(
            torch.randn(1, n_future, n_channels, d_model) * 0.02
        )
        self.layers = nn.ModuleList(
            [
                FutureQueryDecoderLayer(d_model, n_heads, dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, queries: torch.Tensor, encoder_output: torch.Tensor
    ) -> torch.Tensor:
        B = encoder_output.shape[0]
        x = queries.expand(B, -1, -1, -1)
        for layer in self.layers:
            x = layer(x, encoder_output)
        return self.norm(x)


class FeatureEmbedding(nn.Module):
    """Embeds the 9 features with learnable fusion."""

    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.target_proj = nn.Linear(1, d_model // 2)
        self.freq_proj = nn.Linear(n_features - 1, d_model // 2)
        self.fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target = x[..., 0:1]
        freq_bands = x[..., 1:]
        target_emb = self.target_proj(target)
        freq_emb = self.freq_proj(freq_bands)
        combined = torch.cat([target_emb, freq_emb], dim=-1)
        return self.fusion(combined)


class SpatialTemporalForecaster(nn.Module):
    """Complete spatial-temporal transformer model."""

    def __init__(
        self,
        n_channels: int,
        n_features: int = 9,
        n_input_steps: int = 10,
        n_output_steps: int = 10,
        d_model: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 1,
        d_ff: int = 256,
        dropout: float = 0.1,
        use_revin: bool = False,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_features = n_features
        self.n_input_steps = n_input_steps
        self.n_output_steps = n_output_steps
        self.d_model = d_model
        self.use_revin = use_revin

        if use_revin:
            self.revin = RevIN(n_channels)

        self.feature_embed = FeatureEmbedding(n_features, d_model, dropout)
        self.pos_encoding = LearnablePositionalEncoding(
            max_time=n_input_steps + n_output_steps,
            max_channels=n_channels,
            d_model=d_model,
        )
        self.encoder_layers = nn.ModuleList(
            [
                SpatialTemporalBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_encoder_layers)
            ]
        )
        self.decoder = FutureQueryDecoder(
            n_future=n_output_steps,
            n_channels=n_channels,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_decoder_layers,
            dropout=dropout,
        )
        self.output_proj = nn.Linear(d_model, 1)
        self.aux_output_proj = nn.Linear(d_model, n_features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C, F = x.shape
        if self.use_revin:
            x_target = x[..., 0]
            x_target = self.revin(x_target, mode="norm")
            x = torch.cat([x_target.unsqueeze(-1), x[..., 1:]], dim=-1)
        x = self.feature_embed(x)
        x = self.pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        future_queries = self.decoder.future_queries
        decoded = self.decoder(future_queries, x)
        main_output = self.output_proj(decoded).squeeze(-1)
        aux_output = self.aux_output_proj(decoded)
        if self.use_revin:
            main_output = self.revin(main_output, mode="denorm")
        return main_output, aux_output


# =============================================================================
# Model class (required interface for submission)
# =============================================================================


class Model(torch.nn.Module):
    """Submission model wrapper."""

    def __init__(self, monkey_name="beignet"):
        super(Model, self).__init__()
        self.monkey_name = monkey_name

        # Set channel count based on monkey
        if self.monkey_name == "beignet":
            self.n_channels = 89
        elif self.monkey_name == "affi":
            self.n_channels = 239
        else:
            raise ValueError(f"No such a monkey: {self.monkey_name}")

        # Memory-efficient model configuration
        d_model = 128
        n_heads = 4
        n_encoder_layers = 2
        n_decoder_layers = 1
        d_ff = 256

        self.model = SpatialTemporalForecaster(
            n_channels=self.n_channels,
            n_features=9,
            n_input_steps=10,
            n_output_steps=10,
            d_model=d_model,
            n_heads=n_heads,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            d_ff=d_ff,
            dropout=0.15,
            use_revin=True,
        )

        # Load normalization stats (saved from training as mean/std with shape (1, 1, C, F))
        base = os.path.dirname(__file__)
        try:
            data = np.load(
                os.path.join(base, f"train_data_average_std_{self.monkey_name}.npz")
            )
            self.mean = data["mean"]
            self.std = data["std"]
        except FileNotFoundError:
            print(
                f"Warning: train_data_average_std_{self.monkey_name}.npz not found. "
                "Will compute during predict."
            )
            self.mean = None
            self.std = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: (batch, time, channels, features) or (batch, time, channels)
        Returns:
            (batch, time, channels) predictions
        """
        # Handle case where only feature 0 is provided
        if x.dim() == 3:
            # Expand to include feature dimension (add dummy features)
            x = x.unsqueeze(-1).expand(-1, -1, -1, 9)

        x_input = x[:, :10, :, :]
        main_pred, _ = self.model(x_input)
        input_feature0 = x[:, :10, :, 0]
        full_pred = torch.cat([input_feature0, main_pred], dim=1)
        return full_pred

    def load(self):
        """Load model weights from file."""
        base = os.path.dirname(__file__)
        path = os.path.join(base, f"model_{self.monkey_name}.pth")
        state_dict = torch.load(
            path,
            map_location=torch.device(device),
            weights_only=True,
        )
        self.model.load_state_dict(state_dict)
        # Move entire model to device
        self.to(device)

    def predict(self, x):
        """
        Predict on input data.
        Args:
            x: numpy array of shape (N, T, C, F)
        Returns:
            numpy array of shape (N, T, C) with predictions on original scale
        """
        # Keep original data for first 10 timesteps (input)
        original_x = x.astype(np.float32)

        # Create dataset with normalization
        if self.mean is not None:
            dataset = NeuroForcastDataset(
                x, use_graph=True, mean=self.mean, std=self.std
            )
        else:
            dataset = NeuroForcastDataset(x, use_graph=True)

        self.eval()
        predictions = []

        with torch.no_grad():
            for i in range(len(dataset)):
                sample = dataset[i]  # (T, C, F) - normalized
                if sample.dim() == 3:
                    sample = sample.unsqueeze(0)  # Add batch dimension

                sample = sample.to(device)

                # Get model prediction (normalized)
                x_input = sample[:, :10, :, :]
                main_pred, _ = self.model(x_input)  # (1, 10, C) normalized

                # Denormalize the predictions
                main_pred_np = main_pred.cpu().numpy().squeeze(0)  # (10, C)
                if self.mean is not None:
                    # Denormalize: reverse (data - mean) / (4 * std)
                    # mean/std shape is (1, 1, C, F), get feature 0
                    mean_f0 = self.mean[0, 0, :, 0]  # (C,)
                    std_f0 = self.std[0, 0, :, 0]  # (C,)
                    main_pred_np = main_pred_np * (4 * std_f0) + mean_f0

                # Use original (unnormalized) input for first 10 timesteps
                input_feature0 = original_x[i, :10, :, 0]  # (10, C) original scale

                # Combine input and denormalized prediction
                full_pred = np.concatenate(
                    [input_feature0, main_pred_np], axis=0
                )  # (20, C)

                predictions.append(full_pred)

        return np.array(predictions)
