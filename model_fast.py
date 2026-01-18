"""
Fast Neural Forecasting Model - Optimized for Competition Time Limits

This version prioritizes inference speed while maintaining accuracy:
- Simplified feature embedding (no frequency attention)
- Fewer encoder/decoder layers
- No autoregressive refinement
- Simpler positional encoding
- No multi-scale convolutions
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    """Simple Reversible Instance Normalization."""

    def __init__(self, n_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(1, 1, n_channels))
            self.bias = nn.Parameter(torch.zeros(1, 1, n_channels))

        self.mean = None
        self.std = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.std = (
                (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            )
            x = (x - self.mean) / self.std
            if self.affine:
                x = x * self.weight + self.bias
            return x
        else:  # denorm
            if self.affine:
                x = (x - self.bias) / (self.weight + self.eps)
            return x * self.std + self.mean


class SimpleFeatureEmbedding(nn.Module):
    """Fast feature embedding - simple linear projection."""

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, F) -> (B, T, C, d_model)
        return self.norm(self.proj(x))


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding - fast and effective."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(
            position * div_term[: d_model // 2] if d_model % 2 else div_term
        )
        self.register_buffer(
            "pe", pe.unsqueeze(0).unsqueeze(2)
        )  # (1, max_len, 1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, D)
        return x + self.pe[:, : x.size(1), :, :]


class FastSpatialTemporalBlock(nn.Module):
    """Simplified spatial-temporal attention block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Spatial attention
        self.spatial_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.spatial_norm = nn.LayerNorm(d_model)

        # Temporal attention
        self.temporal_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, D = x.shape

        # Spatial attention (across channels)
        x_s = x.reshape(B * T, C, D)
        attn_out, _ = self.spatial_attn(x_s, x_s, x_s)
        x = x + self.dropout(attn_out).reshape(B, T, C, D)
        x = self.spatial_norm(x)

        # Temporal attention (across time)
        x_t = x.permute(0, 2, 1, 3).reshape(B * C, T, D)
        attn_out, _ = self.temporal_attn(x_t, x_t, x_t)
        x = x + self.dropout(attn_out).reshape(B, C, T, D).permute(0, 2, 1, 3)
        x = self.temporal_norm(x)

        # FFN
        x = x + self.ffn(x)
        x = self.ffn_norm(x)

        return x


class FastDecoder(nn.Module):
    """Simplified decoder - no refinement, just cross-attention."""

    def __init__(
        self,
        n_future: int,
        n_channels: int,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.future_queries = nn.Parameter(
            torch.randn(1, n_future, n_channels, d_model) * 0.02
        )

        # Single decoder layer
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_out: torch.Tensor) -> torch.Tensor:
        B, T_enc, C, D = encoder_out.shape
        T_fut = self.future_queries.size(1)

        x = self.future_queries.expand(B, -1, -1, -1)

        # Flatten for attention
        x_flat = x.reshape(B, T_fut * C, D)
        enc_flat = encoder_out.reshape(B, T_enc * C, D)

        # Self attention
        attn_out, _ = self.self_attn(x_flat, x_flat, x_flat)
        x_flat = self.norm1(x_flat + self.dropout(attn_out))

        # Cross attention
        attn_out, _ = self.cross_attn(x_flat, enc_flat, enc_flat)
        x_flat = self.norm2(x_flat + self.dropout(attn_out))

        # FFN
        x_flat = self.norm3(x_flat + self.ffn(x_flat))

        return x_flat.reshape(B, T_fut, C, D)


class FastForecaster(nn.Module):
    """
    Fast neural forecasting model optimized for competition inference speed.
    """

    def __init__(
        self,
        n_channels: int,
        n_features: int = 9,
        n_input_steps: int = 10,
        n_output_steps: int = 10,
        d_model: int = 96,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        use_revin: bool = True,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_input_steps = n_input_steps
        self.n_output_steps = n_output_steps
        self.use_revin = use_revin

        # RevIN
        if use_revin:
            self.revin = RevIN(n_channels)

        # Feature embedding
        self.feature_embed = SimpleFeatureEmbedding(n_features, d_model)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(
            n_input_steps + n_output_steps, d_model
        )

        # Encoder
        self.encoder_layers = nn.ModuleList(
            [
                FastSpatialTemporalBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_encoder_layers)
            ]
        )

        # Decoder
        self.decoder = FastDecoder(
            n_output_steps, n_channels, d_model, n_heads, dropout
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 20, C, 9) - full input with all features
        Returns:
            (B, 20, C) - prediction for feature 0
        """
        B, T, C, F = x.shape

        # Extract input portion
        x_input = x[:, : self.n_input_steps, :, :]

        # Apply RevIN to target feature
        target = x_input[..., 0]
        if self.use_revin:
            target_norm = self.revin(target, mode="norm")
            x_input = torch.cat([target_norm.unsqueeze(-1), x_input[..., 1:]], dim=-1)

        # Feature embedding
        x_emb = self.feature_embed(x_input)

        # Positional encoding
        x_emb = self.pos_enc(x_emb)

        # Encode
        for layer in self.encoder_layers:
            x_emb = layer(x_emb)

        # Decode
        decoded = self.decoder(x_emb)

        # Output projection
        pred_future = self.output_proj(decoded).squeeze(-1)  # (B, n_output, C)

        # Denormalize
        if self.use_revin:
            pred_future = self.revin(pred_future, mode="denorm")

        # Combine with input
        input_target = x[:, : self.n_input_steps, :, 0]
        return torch.cat([input_target, pred_future], dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inference interface."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class NeuralForecaster(nn.Module):
    """Wrapper class matching competition interface."""

    def __init__(self, n_channels: int, memory_efficient: bool = True, **kwargs):
        super().__init__()

        # Fast configuration
        self.model = FastForecaster(
            n_channels=n_channels,
            n_features=9,
            n_input_steps=10,
            n_output_steps=10,
            d_model=96,  # Reduced from 128
            n_heads=4,
            n_encoder_layers=2,  # Reduced from 3-4
            d_ff=256,  # Reduced from 512
            dropout=0.1,
            use_revin=True,
        )

        self.n_channels = n_channels

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.predict(x)


def create_model(
    n_channels: int, device: torch.device, memory_efficient: bool = True, **kwargs
):
    """Factory function."""
    model = NeuralForecaster(n_channels=n_channels, memory_efficient=memory_efficient)
    return model.to(device)


# Loss function for training
class ForecastingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target, **kwargs):
        # Only compute loss on future steps (10-19)
        loss = self.mse(pred[:, 10:, :], target[:, 10:, :])
        return loss, {"mse": loss.item()}


# Alias for backward compatibility
EnhancedForecastingLoss = ForecastingLoss


def validate(model, val_loader, device):
    model.eval()
    total_mse = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model.predict(batch)
            target = batch[:, :, :, 0]
            mse = F.mse_loss(pred[:, 10:, :], target[:, 10:, :])
            total_mse += mse.item()
            n += 1
    return total_mse / max(n, 1)


if __name__ == "__main__":
    # Quick benchmark
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Test both monkey types
    for n_channels, name in [(239, "affi"), (89, "beignet")]:
        print(f"\nTesting {name} ({n_channels} channels)...")

        model = create_model(n_channels, device)
        model.eval()

        params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {params:,}")

        # Benchmark inference
        x = torch.randn(32, 20, n_channels, 9).to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model.predict(x)

        # Time it
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        n_iters = 50
        with torch.no_grad():
            for _ in range(n_iters):
                _ = model.predict(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.time() - start
        per_batch = elapsed / n_iters * 1000
        print(f"  Inference time: {per_batch:.1f} ms/batch (batch_size=32)")

        # Estimate total time for competition
        # Rough estimate: ~500 test samples total
        est_batches = 500 // 32 + 1
        est_total = per_batch * est_batches / 1000
        print(f"  Estimated total inference time: {est_total:.1f} seconds")
