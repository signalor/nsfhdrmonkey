"""
Spatial-Temporal Transformer with Reversible Instance Normalization
for Neural Forecasting Competition

Key components:
1. RevIN - handles cross-session distribution shift
2. Convolutional Tokenizer - captures multi-scale local patterns
3. Factorized Spatial-Temporal Attention - efficient and interpretable
4. Learned Future Queries - direct multi-step prediction
5. Multi-task learning - predict all frequency bands as auxiliary task
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Component 1: Reversible Instance Normalization (RevIN)
# Paper: "Reversible Instance Normalization for Accurate Time-Series Forecasting"
# =============================================================================


class RevIN(nn.Module):
    """
    Reversible Instance Normalization for handling distribution shift.
    Normalizes input, stores statistics, and can denormalize output.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
        """
        Args:
            x: (batch, time, channels) or (batch, time, channels, features)
            mode: 'norm' to normalize, 'denorm' to denormalize
        """
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x: torch.Tensor):
        # Compute mean and std across time dimension
        dim2reduce = tuple(range(1, x.ndim - 1))  # all dims except batch and last
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


# =============================================================================
# Component 2: Convolutional Tokenizer
# Captures local temporal patterns at multiple scales before attention
# =============================================================================


class ConvTokenizer(nn.Module):
    """
    Multi-scale convolutional front-end that captures local temporal patterns.
    Uses depthwise separable convolutions for efficiency.
    """

    def __init__(self, in_channels: int, d_model: int, kernel_sizes: list = [3, 5, 7]):
        super().__init__()
        self.kernel_sizes = kernel_sizes

        # Multi-scale convolutions
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

        # Projection to combine scales
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, features)
        Returns:
            (batch, time, d_model)
        """
        # (batch, time, features) -> (batch, features, time)
        x = x.transpose(1, 2)

        # Apply multi-scale convolutions
        conv_outputs = [conv(x) for conv in self.convs]

        # Concatenate and transpose back
        x = torch.cat(conv_outputs, dim=1)  # (batch, d_model, time)
        x = x.transpose(1, 2)  # (batch, time, d_model)

        x = self.proj(x)
        x = self.norm(x)
        return x


# =============================================================================
# Component 3: Factorized Spatial-Temporal Attention
# Alternates between attending across channels and across time
# =============================================================================


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
        """
        Args:
            x: (batch, time, channels, d_model)
        """
        B, T, C, D = x.shape

        # Reshape to process all timesteps in parallel
        x_flat = x.reshape(B * T, C, D)

        # Self-attention across channels
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        attn_out = self.dropout(attn_out)

        # Residual and reshape
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
        """
        Args:
            x: (batch, time, channels, d_model)
            mask: optional causal mask
        """
        B, T, C, D = x.shape

        # Reshape to process all channels in parallel
        x_flat = x.permute(0, 2, 1, 3).reshape(B * C, T, D)

        # Self-attention across time
        attn_out, _ = self.attn(x_flat, x_flat, x_flat, attn_mask=mask)
        attn_out = self.dropout(attn_out)

        # Residual and reshape
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

        # Apply feed-forward to each position
        B, T, C, D = x.shape
        x = x.reshape(B * T * C, D)
        x = self.ff(x.unsqueeze(1)).squeeze(1)
        x = x.reshape(B, T, C, D)
        return x


# =============================================================================
# Component 4: Positional Encodings
# =============================================================================


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding for both time and space."""

    def __init__(self, max_time: int, max_channels: int, d_model: int):
        super().__init__()
        self.time_embed = nn.Parameter(torch.randn(1, max_time, 1, d_model) * 0.02)
        self.channel_embed = nn.Parameter(
            torch.randn(1, 1, max_channels, d_model) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, channels, d_model)
        """
        B, T, C, D = x.shape
        return x + self.time_embed[:, :T, :, :] + self.channel_embed[:, :, :C, :]


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time dimension."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer(
            "pe", pe.unsqueeze(0).unsqueeze(2)
        )  # (1, max_len, 1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :, :]


# =============================================================================
# Component 5: Future Query Decoder
# Uses learned queries to directly predict future timesteps
# =============================================================================


class FutureQueryDecoder(nn.Module):
    """
    Decoder that uses learned queries to predict future timesteps.
    Similar to DETR's object queries but for time series.
    """

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

        # Learned queries for future timesteps
        self.future_queries = nn.Parameter(
            torch.randn(1, n_future, n_channels, d_model) * 0.02
        )

        # Cross-attention layers
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
        """
        Args:
            queries: (batch, n_future, n_channels, d_model)
            encoder_output: (batch, time, n_channels, d_model)
        """
        B = encoder_output.shape[0]

        # Expand queries for batch
        x = queries.expand(B, -1, -1, -1)

        for layer in self.layers:
            x = layer(x, encoder_output)

        return self.norm(x)


class FutureQueryDecoderLayer(nn.Module):
    """Single decoder layer with self-attention and cross-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        # Self-attention among future queries
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention to encoder output
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward
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
        """
        Args:
            x: (batch, n_future, n_channels, d_model)
            encoder_output: (batch, time, n_channels, d_model)
        """
        B, T_fut, C, D = x.shape
        _, T_enc, _, _ = encoder_output.shape

        # Reshape for attention: combine future and channels
        x_flat = x.reshape(B, T_fut * C, D)
        enc_flat = encoder_output.reshape(B, T_enc * C, D)

        # Self-attention
        attn_out, _ = self.self_attn(x_flat, x_flat, x_flat)
        x_flat = self.norm1(x_flat + self.dropout(attn_out))

        # Cross-attention to encoder
        attn_out, _ = self.cross_attn(x_flat, enc_flat, enc_flat)
        x_flat = self.norm2(x_flat + self.dropout(attn_out))

        # Feed-forward
        x_flat = self.norm3(x_flat + self.ff(x_flat))

        return x_flat.reshape(B, T_fut, C, D)


# =============================================================================
# Component 6: Feature Embedding with Frequency Band Fusion
# =============================================================================


class FeatureEmbedding(nn.Module):
    """
    Embeds the 9 features (1 target + 8 frequency bands) with learnable fusion.
    Uses separate projections for target and frequency bands.
    """

    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()

        # Separate embedding for target feature
        self.target_proj = nn.Linear(1, d_model // 2)

        # Embedding for frequency bands
        self.freq_proj = nn.Linear(n_features - 1, d_model // 2)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, channels, features)
        Returns:
            (batch, time, channels, d_model)
        """
        target = x[..., 0:1]  # (B, T, C, 1)
        freq_bands = x[..., 1:]  # (B, T, C, 8)

        target_emb = self.target_proj(target)
        freq_emb = self.freq_proj(freq_bands)

        combined = torch.cat([target_emb, freq_emb], dim=-1)
        return self.fusion(combined)


# =============================================================================
# Main Model: SpatialTemporalForecaster
# =============================================================================


class SpatialTemporalForecaster(nn.Module):
    """
    Complete model for neural forecasting with:
    - RevIN for distribution shift handling
    - Multi-scale convolutional tokenizer
    - Factorized spatial-temporal attention
    - Learned future queries for direct multi-step prediction
    """

    def __init__(
        self,
        n_channels: int,
        n_features: int = 9,
        n_input_steps: int = 10,
        n_output_steps: int = 10,
        d_model: int = 256,
        n_heads: int = 8,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 2,
        d_ff: int = 1024,
        dropout: float = 0.1,
        use_revin: bool = True,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_features = n_features
        self.n_input_steps = n_input_steps
        self.n_output_steps = n_output_steps
        self.d_model = d_model
        self.use_revin = use_revin

        # RevIN for handling distribution shift
        if use_revin:
            self.revin = RevIN(n_channels)

        # Feature embedding
        self.feature_embed = FeatureEmbedding(n_features, d_model, dropout)

        # Positional encoding
        self.pos_encoding = LearnablePositionalEncoding(
            max_time=n_input_steps + n_output_steps,
            max_channels=n_channels,
            d_model=d_model,
        )

        # Encoder: Spatial-Temporal blocks
        self.encoder_layers = nn.ModuleList(
            [
                SpatialTemporalBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_encoder_layers)
            ]
        )

        # Decoder with learned future queries
        self.decoder = FutureQueryDecoder(
            n_future=n_output_steps,
            n_channels=n_channels,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_decoder_layers,
            dropout=dropout,
        )

        # Output projections
        # Main output: predict feature 0
        self.output_proj = nn.Linear(d_model, 1)

        # Auxiliary output: predict all features (multi-task learning)
        self.aux_output_proj = nn.Linear(d_model, n_features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, time, channels, features) - input sequence
        Returns:
            main_output: (batch, n_output_steps, channels) - predicted feature 0
            aux_output: (batch, n_output_steps, channels, features) - all features
        """
        B, T, C, F = x.shape

        # Apply RevIN normalization (on feature 0)
        if self.use_revin:
            # Normalize feature 0 across time
            x_target = x[..., 0]  # (B, T, C)
            x_target = self.revin(x_target, mode="norm")
            x = torch.cat([x_target.unsqueeze(-1), x[..., 1:]], dim=-1)

        # Feature embedding
        x = self.feature_embed(x)  # (B, T, C, d_model)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Encode
        for layer in self.encoder_layers:
            x = layer(x)

        # Decode with future queries
        future_queries = self.decoder.future_queries
        decoded = self.decoder(future_queries, x)  # (B, n_output, C, d_model)

        # Project to outputs
        main_output = self.output_proj(decoded).squeeze(-1)  # (B, n_output, C)
        aux_output = self.aux_output_proj(decoded)  # (B, n_output, C, F)

        # Apply RevIN denormalization
        if self.use_revin:
            main_output = self.revin(main_output, mode="denorm")

        return main_output, aux_output

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inference mode - returns only main prediction."""
        main_output, _ = self.forward(x)
        return main_output


# =============================================================================
# Loss Functions
# =============================================================================


class ForecastingLoss(nn.Module):
    """
    Combined loss with:
    - MSE on target feature
    - Auxiliary loss on all frequency bands
    - Optional temporal consistency regularization
    """

    def __init__(self, aux_weight: float = 0.3, consistency_weight: float = 0.1):
        super().__init__()
        self.aux_weight = aux_weight
        self.consistency_weight = consistency_weight
        self.mse = nn.MSELoss()

    def forward(
        self,
        main_pred: torch.Tensor,
        aux_pred: torch.Tensor,
        target: torch.Tensor,
        target_full: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            main_pred: (B, T, C) - predicted feature 0
            aux_pred: (B, T, C, F) - predicted all features
            target: (B, T, C) - ground truth feature 0
            target_full: (B, T, C, F) - ground truth all features (optional)
        """
        # Main loss
        main_loss = self.mse(main_pred, target)

        # Auxiliary loss
        aux_loss = torch.tensor(0.0, device=main_pred.device)
        if target_full is not None and self.aux_weight > 0:
            aux_loss = self.mse(aux_pred, target_full)

        # Temporal consistency: predictions shouldn't jump too much between timesteps
        consistency_loss = torch.tensor(0.0, device=main_pred.device)
        if self.consistency_weight > 0:
            pred_diff = main_pred[:, 1:, :] - main_pred[:, :-1, :]
            consistency_loss = torch.mean(pred_diff**2)

        total_loss = (
            main_loss
            + self.aux_weight * aux_loss
            + self.consistency_weight * consistency_loss
        )

        metrics = {
            "main_loss": main_loss.item(),
            "aux_loss": aux_loss.item(),
            "consistency_loss": consistency_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, metrics


# =============================================================================
# Data Augmentation
# =============================================================================


class NeuralDataAugmentation:
    """Data augmentation strategies for neural signals."""

    @staticmethod
    def channel_dropout(x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
        """Randomly zero out entire channels."""
        if not torch.is_grad_enabled():  # Only during training
            return x
        B, T, C, F = x.shape
        mask = torch.rand(B, 1, C, 1, device=x.device) > p
        return x * mask

    @staticmethod
    def time_noise(x: torch.Tensor, std: float = 0.05) -> torch.Tensor:
        """Add Gaussian noise."""
        if not torch.is_grad_enabled():
            return x
        noise = torch.randn_like(x) * std
        return x + noise

    @staticmethod
    def time_warp(x: torch.Tensor, sigma: float = 0.2) -> torch.Tensor:
        """Slight temporal warping via interpolation."""
        # This is more complex to implement efficiently, skipping for now
        return x


# =============================================================================
# Complete Model Wrapper for Competition
# =============================================================================


class NeuralForecaster(nn.Module):
    """
    Complete wrapper for the competition.
    Handles both Monkey A and Monkey B with configurable channel counts.
    """

    def __init__(self, n_channels: int, memory_efficient: bool = False):
        super().__init__()

        self.n_channels = n_channels

        # Model configuration based on memory mode
        if memory_efficient:
            # Reduced model for limited GPU memory
            d_model = 128
            n_heads = 4
            n_encoder_layers = 2
            n_decoder_layers = 1
            d_ff = 256
        else:
            # Full model
            d_model = 256
            n_heads = 8
            n_encoder_layers = 4
            n_decoder_layers = 2
            d_ff = 1024

        self.model = SpatialTemporalForecaster(
            n_channels=n_channels,
            n_features=9,
            n_input_steps=10,
            n_output_steps=10,
            d_model=d_model,
            n_heads=n_heads,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            d_ff=d_ff,
            dropout=0.15,
            use_revin=False,
        )

        self.augmentation = NeuralDataAugmentation()

    def forward(self, x: torch.Tensor, augment: bool = False) -> torch.Tensor:
        """
        Args:
            x: (batch, 20, channels, 9) - full sequence with input and target
        Returns:
            (batch, 20, channels) - prediction for feature 0
        """
        # Split input and target
        x_input = x[:, :10, :, :]  # First 10 timesteps

        # Apply augmentation during training
        if augment and self.training:
            x_input = self.augmentation.channel_dropout(x_input, p=0.1)
            x_input = self.augmentation.time_noise(x_input, std=0.03)

        # Get predictions
        main_pred, aux_pred = self.model(x_input)

        # Concatenate input and prediction for full sequence output
        # Use input feature 0 for first 10 steps, prediction for last 10
        input_feature0 = x[:, :10, :, 0]
        full_pred = torch.cat([input_feature0, main_pred], dim=1)

        return full_pred

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inference mode."""
        self.eval()
        with torch.no_grad():
            return self.forward(x, augment=False)


# =============================================================================
# Training Utilities
# =============================================================================


def create_model(
    n_channels: int, device: torch.device, memory_efficient: bool = False
) -> NeuralForecaster:
    """Factory function to create model."""
    model = NeuralForecaster(n_channels=n_channels, memory_efficient=memory_efficient)
    return model.to(device)


def train_step(
    model: NeuralForecaster,
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: ForecastingLoss,
    device: torch.device,
) -> dict:
    """Single training step."""
    model.train()
    batch = batch.to(device)

    optimizer.zero_grad()

    # Forward pass with augmentation
    x_input = batch[:, :10, :, :]
    x_input_aug = model.augmentation.channel_dropout(x_input, p=0.1)
    x_input_aug = model.augmentation.time_noise(x_input_aug, std=0.03)

    main_pred, aux_pred = model.model(x_input_aug)

    # Targets
    target = batch[:, 10:, :, 0]  # Feature 0 of future timesteps
    target_full = batch[:, 10:, :, :]  # All features

    # Compute loss
    loss, metrics = loss_fn(main_pred, aux_pred, target, target_full)

    # Backward pass
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    return metrics


def validate(
    model: NeuralForecaster,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Validation loop."""
    model.eval()
    total_mse = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            # Get predictions
            pred = model.predict(batch)
            target = batch[:, :, :, 0]  # Feature 0

            # Compute MSE on future timesteps only (use mean reduction)
            mse = F.mse_loss(pred[:, 10:, :], target[:, 10:, :], reduction="mean")
            total_mse += mse.item()
            n_batches += 1

    return total_mse / n_batches


def validate_competition(
    model: NeuralForecaster,
    test_loaders: dict,
    device: torch.device,
) -> dict:
    """
    Validation with competition metrics.

    Args:
        test_loaders: dict with keys like 'affi', 'affi_d2', 'beignet', 'beignet_d2', 'beignet_d3'
    Returns:
        dict with all MSE scores and Total MSR
    """
    model.eval()
    results = {}

    with torch.no_grad():
        for dataset_name, loader in test_loaders.items():
            total_mse = 0.0
            n_samples = 0

            for batch in loader:
                batch = batch.to(device)

                # Get predictions
                pred = model.predict(batch)
                target = batch[:, :, :, 0]  # Feature 0

                # Compute MSE on future timesteps only (10:20)
                mse = F.mse_loss(pred[:, 10:, :], target[:, 10:, :], reduction="sum")
                total_mse += mse.item()
                n_samples += (
                    batch.shape[0] * 10 * batch.shape[2]
                )  # batch * timesteps * channels

            # Store average MSE for this dataset
            results[f"MSE_{dataset_name}"] = total_mse / n_samples

    # Calculate Total MSR (sum of all MSEs)
    results["Total_MSR"] = sum(results.values())

    return results


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test Monkey A
    model = create_model("affi", device)
    x = torch.randn(4, 20, 239, 9).to(device)

    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
