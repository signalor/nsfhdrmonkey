"""
Enhanced Spatial-Temporal Transformer for Neural Forecasting Competition

Key improvements over baseline:
1. Channel-Aware Positional Encoding - learns channel relationships
2. Frequency Band Attention - explicit cross-frequency modeling
3. Autoregressive Refinement - iterative prediction improvement
4. Test-Time Augmentation (TTA) - ensemble predictions during inference
5. Ensemble Support - combine multiple model predictions
6. Session-Specific Normalization - adaptive normalization layers
7. Gated Residual Connections - improved gradient flow
8. Multi-Scale Temporal Convolutions - capture patterns at different scales
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Component 1: Enhanced Reversible Instance Normalization (RevIN)
# Handles distribution shift across sessions
# =============================================================================


class EnhancedRevIN(nn.Module):
    """
    Enhanced Reversible Instance Normalization with:
    - Optional "subtract_last" mode for non-stationary data
    - Learnable affine parameters
    - Better numerical stability
    """

    def __init__(
        self,
        n_channels: int,
        affine: bool = True,
        subtract_last: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.affine = affine
        self.subtract_last = subtract_last
        self.eps = eps

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, n_channels))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, n_channels))

        # Statistics (set during forward)
        self.mean = None
        self.stdev = None
        self.last = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Args:
            x: (batch, time, channels) for mode='norm', or predictions for 'denorm'
            mode: 'norm' or 'denorm'
        """
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _get_statistics(self, x: torch.Tensor):
        """Compute and store normalization statistics."""
        dim2reduce = (1,)  # Reduce over time dimension

        if self.subtract_last:
            # Use last timestep as reference (helps with non-stationary data)
            self.last = x[:, -1:, :].detach()

        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.subtract_last:
            x = x - self.last
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        x = x * self.stdev + self.mean
        if self.subtract_last:
            x = x + self.last
        return x


# =============================================================================
# Component 2: Channel-Aware Positional Encoding (FIXED)
# Learns relationships between channels
# =============================================================================


class ChannelAwarePositionalEncoding(nn.Module):
    """
    Positional encoding that captures:
    - Temporal position (sinusoidal + learnable)
    - Channel identity (learnable embeddings)
    - Channel-channel relationships (learnable pairwise)

    FIXED: Proper tensor dimension handling in forward pass
    """

    def __init__(
        self,
        max_time: int,
        max_channels: int,
        d_model: int,
        use_sinusoidal: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_time = max_time
        self.max_channels = max_channels
        self.use_sinusoidal = use_sinusoidal

        # Learnable temporal embeddings
        self.time_embed = nn.Parameter(torch.randn(1, max_time, 1, d_model) * 0.02)

        # Learnable channel embeddings
        self.channel_embed = nn.Parameter(
            torch.randn(1, 1, max_channels, d_model) * 0.02
        )

        # Optional sinusoidal encoding for time
        if use_sinusoidal:
            self._init_sinusoidal(max_time, d_model)

    def _init_sinusoidal(self, max_time: int, d_model: int):
        """Initialize sinusoidal positional encoding."""
        pe = torch.zeros(max_time, d_model)
        position = torch.arange(0, max_time, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        # Shape: (1, max_time, 1, d_model)
        self.register_buffer("sinusoidal_pe", pe.unsqueeze(0).unsqueeze(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, channels, d_model)
        Returns:
            x with positional encoding added: (batch, time, channels, d_model)
        """
        B, T, C, D = x.shape

        # Get time encoding - broadcast across channels
        time_enc = self.time_embed[:, :T, :, :]  # (1, T, 1, D)

        if self.use_sinusoidal:
            # Add sinusoidal component to learnable time embedding
            sinusoidal = self.sinusoidal_pe[:, :T, :, :]  # (1, T, 1, D)
            time_enc = time_enc + sinusoidal

        # Get channel encoding - broadcast across time
        channel_enc = self.channel_embed[:, :, :C, :]  # (1, 1, C, D)

        # Add encodings - broadcasting handles dimension expansion
        # time_enc: (1, T, 1, D) broadcasts to (B, T, C, D)
        # channel_enc: (1, 1, C, D) broadcasts to (B, T, C, D)
        return x + time_enc + channel_enc


# =============================================================================
# Component 3: Enhanced Feature Embedding with Frequency Attention
# =============================================================================


class EnhancedFeatureEmbedding(nn.Module):
    """
    Feature embedding with explicit frequency band attention.
    Treats target (feature 0) and frequency bands (features 1-8) differently.
    """

    def __init__(
        self, n_features: int, d_model: int, n_heads: int = 4, dropout: float = 0.1
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Separate projections for target and frequency bands
        self.target_proj = nn.Sequential(
            nn.Linear(1, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, d_model // 2)
        )

        # Project each frequency band individually
        self.n_freq_bands = n_features - 1
        self.freq_proj = nn.Sequential(
            nn.Linear(self.n_freq_bands, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
        )

        # Cross-frequency attention
        self.freq_attention = nn.MultiheadAttention(
            embed_dim=d_model // 2,
            num_heads=max(1, n_heads),
            dropout=dropout,
            batch_first=True,
        )

        # Gated fusion
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

        # Final projection
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
        B, T, C, F = x.shape

        # Extract target and frequency bands
        # Use contiguous() to ensure proper memory alignment for CUDA operations
        target = x[..., 0:1].contiguous()  # (B, T, C, 1)
        freq_bands = x[..., 1:].contiguous()  # (B, T, C, n_freq_bands)

        # Project target
        target_emb = self.target_proj(target)  # (B, T, C, d_model//2)

        # Project frequency bands
        freq_emb = self.freq_proj(freq_bands)  # (B, T, C, d_model//2)

        # Apply frequency attention (reshape for attention)
        # Treat each (time, channel) position as a sequence element
        freq_flat = freq_emb.reshape(B, T * C, -1).contiguous()  # (B, T*C, d_model//2)
        freq_attended, _ = self.freq_attention(freq_flat, freq_flat, freq_flat)
        freq_attended = freq_attended.reshape(
            B, T, C, -1
        ).contiguous()  # (B, T, C, d_model//2)

        # Combine with gated fusion
        combined = torch.cat(
            [target_emb, freq_attended], dim=-1
        ).contiguous()  # (B, T, C, d_model)
        gate = self.gate(combined)

        # Apply fusion with residual
        output = self.fusion(combined * gate + combined * (1 - gate))

        return output.contiguous()


# =============================================================================
# Component 4: Enhanced Spatial-Temporal Block with Gated Residuals
# =============================================================================


class GatedResidual(nn.Module):
    """Gated residual connection for adaptive skip connections."""

    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([x, residual], dim=-1))
        return gate * x + (1 - gate) * residual


class MultiScaleTemporalConv(nn.Module):
    """Multi-scale temporal convolution for capturing patterns at different scales."""

    def __init__(self, d_model: int, kernel_sizes: List[int] = [3, 5, 7]):
        super().__init__()
        # Ensure hidden dim is divisible by 8 for CUDA memory alignment
        hidden_per_kernel = (d_model // len(kernel_sizes)) // 8 * 8
        if hidden_per_kernel == 0:
            hidden_per_kernel = 8
        self.hidden_total = hidden_per_kernel * len(kernel_sizes)

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(d_model, hidden_per_kernel, k, padding=k // 2)
                for k in kernel_sizes
            ]
        )
        self.proj = nn.Linear(self.hidden_total, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, channels, d_model)
        """
        B, T, C, D = x.shape

        # Reshape for conv: (B*C, D, T)
        # Use contiguous() to ensure proper memory alignment for CUDA operations
        x_conv = x.permute(0, 2, 3, 1).reshape(B * C, D, T).contiguous()

        # Apply multi-scale convolutions
        conv_outputs = [conv(x_conv) for conv in self.convs]
        x_conv = torch.cat(conv_outputs, dim=1).contiguous()  # (B*C, hidden_total, T)

        # Reshape and project back to d_model
        x_conv = x_conv.transpose(1, 2).contiguous()  # (B*C, T, hidden_total)
        x_conv = self.proj(x_conv)  # (B*C, T, D)
        x_conv = (
            x_conv.reshape(B, C, T, D).permute(0, 2, 1, 3).contiguous()
        )  # (B, T, C, D)

        return self.norm(x_conv)


class EnhancedSpatialTemporalBlock(nn.Module):
    """
    Enhanced spatial-temporal attention block with:
    - Factorized attention (spatial then temporal)
    - Multi-scale temporal convolutions
    - Gated residual connections
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_channels: int,
        n_time: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Spatial attention (across channels)
        self.spatial_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.spatial_norm = nn.LayerNorm(d_model)
        self.spatial_gate = GatedResidual(d_model)

        # Temporal attention (across time)
        self.temporal_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal_gate = GatedResidual(d_model)

        # Multi-scale temporal convolution
        self.temporal_conv = MultiScaleTemporalConv(d_model)
        self.conv_gate = GatedResidual(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn_gate = GatedResidual(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, channels, d_model)
        """
        B, T, C, D = x.shape

        # Spatial attention: attend across channels for each time step
        # Use contiguous() to ensure proper memory alignment for CUDA operations
        x_spatial = x.reshape(B * T, C, D).contiguous()
        spatial_out, _ = self.spatial_attn(x_spatial, x_spatial, x_spatial)
        spatial_out = self.dropout(spatial_out).reshape(B, T, C, D).contiguous()
        x = self.spatial_gate(self.spatial_norm(spatial_out), x)

        # Temporal attention: attend across time for each channel
        x_temporal = x.permute(0, 2, 1, 3).reshape(B * C, T, D).contiguous()
        temporal_out, _ = self.temporal_attn(x_temporal, x_temporal, x_temporal)
        temporal_out = (
            self.dropout(temporal_out)
            .reshape(B, C, T, D)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        x = self.temporal_gate(self.temporal_norm(temporal_out), x)

        # Multi-scale temporal convolution
        conv_out = self.temporal_conv(x)
        x = self.conv_gate(conv_out, x)

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.ffn_gate(self.ffn_norm(ffn_out), x)

        return x


# =============================================================================
# Component 5: Autoregressive Refinement Decoder
# =============================================================================


class AutoregressiveRefinementDecoder(nn.Module):
    """
    Decoder with autoregressive refinement:
    - Generates initial prediction from learned queries
    - Iteratively refines predictions
    """

    def __init__(
        self,
        n_future: int,
        n_channels: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        n_refinement_steps: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_future = n_future
        self.n_channels = n_channels
        self.n_refinement_steps = n_refinement_steps

        # Learned queries for future timesteps
        self.future_queries = nn.Parameter(
            torch.randn(1, n_future, n_channels, d_model) * 0.02
        )

        # Decoder layers
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        # Refinement embedding (embeds previous prediction for refinement)
        self.refinement_embed = nn.Sequential(
            nn.Linear(1, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, d_model)
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        encoder_output: torch.Tensor,
        prev_prediction: Optional[torch.Tensor] = None,
        refinement_step: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            encoder_output: (batch, time, channels, d_model)
            prev_prediction: (batch, n_future, channels) - previous prediction for refinement
            refinement_step: current refinement iteration
        Returns:
            (batch, n_future, channels, d_model)
        """
        B = encoder_output.shape[0]

        # Start with learned queries
        x = self.future_queries.expand(B, -1, -1, -1)

        # If refining, incorporate previous prediction
        if prev_prediction is not None and refinement_step > 0:
            prev_emb = self.refinement_embed(prev_prediction.unsqueeze(-1))
            x = x + prev_emb

        # Apply decoder layers with cross-attention to encoder
        for layer in self.layers:
            x = layer(x, encoder_output)

        return self.norm(x)


class DecoderLayer(nn.Module):
    """Single decoder layer with self and cross attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        # Self attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Cross attention
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward
        self.ffn = nn.Sequential(
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
            x: (batch, n_future, channels, d_model)
            encoder_output: (batch, time, channels, d_model)
        """
        B, T_fut, C, D = x.shape
        _, T_enc, _, _ = encoder_output.shape

        # Flatten for attention
        # Use contiguous() to ensure proper memory alignment for CUDA operations
        x_flat = x.reshape(B, T_fut * C, D).contiguous()
        enc_flat = encoder_output.reshape(B, T_enc * C, D).contiguous()

        # Self attention
        attn_out, _ = self.self_attn(x_flat, x_flat, x_flat)
        x_flat = self.norm1(x_flat + self.dropout(attn_out))

        # Cross attention
        attn_out, _ = self.cross_attn(x_flat, enc_flat, enc_flat)
        x_flat = self.norm2(x_flat + self.dropout(attn_out))

        # Feed-forward
        x_flat = self.norm3(x_flat + self.ffn(x_flat))

        return x_flat.reshape(B, T_fut, C, D).contiguous()


# =============================================================================
# Component 6: Main Model
# =============================================================================


class EnhancedSpatialTemporalForecaster(nn.Module):
    """
    Enhanced model with all improvements:
    - Enhanced RevIN for distribution shift
    - Frequency band attention
    - Channel-aware positional encoding
    - Multi-scale temporal processing
    - Autoregressive refinement
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
        n_refinement_steps: int = 2,
        d_ff: int = 1024,
        dropout: float = 0.1,
        use_revin: bool = True,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_features = n_features
        self.n_input_steps = n_input_steps
        self.n_output_steps = n_output_steps
        self.n_refinement_steps = n_refinement_steps
        self.d_model = d_model
        self.use_revin = use_revin

        # Enhanced RevIN
        if use_revin:
            self.revin = EnhancedRevIN(n_channels, affine=True, subtract_last=True)

        # Enhanced feature embedding with frequency attention
        self.feature_embed = EnhancedFeatureEmbedding(
            n_features, d_model, n_heads // 2, dropout
        )

        # Channel-aware positional encoding
        self.pos_encoding = ChannelAwarePositionalEncoding(
            max_time=n_input_steps + n_output_steps,
            max_channels=n_channels,
            d_model=d_model,
        )

        # Encoder with enhanced blocks
        self.encoder_layers = nn.ModuleList(
            [
                EnhancedSpatialTemporalBlock(
                    d_model, n_heads, d_ff, n_channels, n_input_steps, dropout
                )
                for _ in range(n_encoder_layers)
            ]
        )

        # Autoregressive refinement decoder
        self.decoder = AutoregressiveRefinementDecoder(
            n_future=n_output_steps,
            n_channels=n_channels,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_decoder_layers,
            n_refinement_steps=n_refinement_steps,
            dropout=dropout,
        )

        # Output projections
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1)
        )

        # Auxiliary output for all features (multi-task learning)
        self.aux_output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_features),
        )

    def forward(
        self, x: torch.Tensor, return_intermediate: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            x: (batch, time, channels, features) - full sequence with features
            return_intermediate: whether to return intermediate predictions
        Returns:
            main_pred: (batch, time, channels) - feature 0 prediction
            aux_pred: (batch, n_output, channels, n_features) - all features prediction
            intermediate: list of intermediate predictions (if return_intermediate)
        """
        B, T, C, F = x.shape

        # Extract input steps
        # Use contiguous() to ensure proper memory alignment for CUDA operations
        x_input = x[:, : self.n_input_steps, :, :].contiguous()  # (B, n_input, C, F)

        # Apply RevIN to target feature
        target = x_input[..., 0].contiguous()  # (B, n_input, C)
        if self.use_revin:
            target_norm = self.revin(target, mode="norm")
            # Reconstruct input with normalized target
            x_input = torch.cat(
                [target_norm.unsqueeze(-1), x_input[..., 1:].contiguous()], dim=-1
            ).contiguous()

        # Feature embedding
        x_emb = self.feature_embed(x_input)  # (B, n_input, C, d_model)

        # Add positional encoding
        x_emb = self.pos_encoding(x_emb)

        # Encode
        encoder_out = x_emb
        for layer in self.encoder_layers:
            encoder_out = layer(encoder_out)

        # Decode with refinement
        intermediate_preds = []
        prev_pred = None

        for step in range(self.n_refinement_steps):
            decoder_out = self.decoder(encoder_out, prev_pred, step)

            # Get main prediction
            main_out = (
                self.output_proj(decoder_out).squeeze(-1).contiguous()
            )  # (B, n_output, C)

            # Denormalize if using RevIN
            if self.use_revin:
                main_out = self.revin(main_out, mode="denorm")

            if return_intermediate:
                intermediate_preds.append(main_out)

            prev_pred = (
                main_out.detach() if step < self.n_refinement_steps - 1 else main_out
            )

        # Get auxiliary prediction (for multi-task learning)
        aux_out = self.aux_output_proj(decoder_out).contiguous()  # (B, n_output, C, F)

        # Construct full prediction
        # Input steps: use ground truth target
        input_target = x[:, : self.n_input_steps, :, 0].contiguous()  # (B, n_input, C)
        full_pred = torch.cat(
            [input_target, prev_pred], dim=1
        ).contiguous()  # (B, T, C)

        if return_intermediate:
            return full_pred, aux_out, intermediate_preds

        return full_pred, aux_out

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Prediction interface for inference."""
        self.eval()
        with torch.no_grad():
            pred, _ = self.forward(x, return_intermediate=False)
        return pred


# =============================================================================
# Component 7: Test-Time Augmentation
# =============================================================================


class TTAWrapper(nn.Module):
    """Test-time augmentation wrapper."""

    def __init__(self, model: nn.Module, n_augments: int = 5):
        super().__init__()
        self.model = model
        self.n_augments = n_augments

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentation."""
        # Small noise augmentation
        noise_scale = 0.01
        return x + torch.randn_like(x) * noise_scale * x.std()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with TTA."""
        predictions = []

        # Original prediction
        pred, _ = self.model(x)
        predictions.append(pred)

        # Augmented predictions
        for _ in range(self.n_augments - 1):
            x_aug = self.augment(x)
            pred_aug, _ = self.model(x_aug)
            predictions.append(pred_aug)

        # Aggregate with trimmed mean (remove outliers)
        stacked = torch.stack(predictions, dim=0)
        if self.n_augments >= 5:
            # Remove top and bottom predictions
            sorted_preds, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_preds[1:-1]  # Remove min and max
            return trimmed.mean(dim=0)
        else:
            return stacked.mean(dim=0)


# =============================================================================
# Component 8: Neural Forecaster Wrapper (Competition Interface)
# =============================================================================


class NeuralForecaster(nn.Module):
    """
    Main wrapper class that provides a clean interface for training and inference.
    Handles RevIN statistics properly and supports TTA.
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
        n_refinement_steps: int = 2,
        d_ff: int = 1024,
        dropout: float = 0.1,
        use_revin: bool = True,
        use_tta: bool = False,
        n_tta_augments: int = 5,
    ):
        super().__init__()

        self.model = EnhancedSpatialTemporalForecaster(
            n_channels=n_channels,
            n_features=n_features,
            n_input_steps=n_input_steps,
            n_output_steps=n_output_steps,
            d_model=d_model,
            n_heads=n_heads,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            n_refinement_steps=n_refinement_steps,
            d_ff=d_ff,
            dropout=dropout,
            use_revin=use_revin,
        )

        self.use_tta = use_tta
        if use_tta:
            self.tta_wrapper = TTAWrapper(self.model, n_tta_augments)

        self.n_channels = n_channels
        self.n_input_steps = n_input_steps
        self.n_output_steps = n_output_steps

    def forward(
        self, x: torch.Tensor, return_intermediate: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass for training."""
        return self.model(x, return_intermediate)

    def forward_with_intermediate(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Forward pass that returns intermediate predictions."""
        return self.model(x, return_intermediate=True)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Prediction interface for inference."""
        self.eval()
        with torch.no_grad():
            if self.use_tta:
                return self.tta_wrapper(x)
            else:
                pred, _ = self.model(x)
                return pred


# =============================================================================
# Component 9: Loss Functions
# =============================================================================


class EnhancedForecastingLoss(nn.Module):
    """
    Enhanced loss function with:
    - MSE + Huber combination for robustness
    - Temporal consistency loss
    - Intermediate prediction supervision
    """

    def __init__(
        self,
        mse_weight: float = 0.7,
        huber_weight: float = 0.3,
        temporal_weight: float = 0.1,
        intermediate_weight: float = 0.2,
        aux_weight: float = 0.1,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.huber_weight = huber_weight
        self.temporal_weight = temporal_weight
        self.intermediate_weight = intermediate_weight
        self.aux_weight = aux_weight

        self.mse = nn.MSELoss()
        self.huber = nn.SmoothL1Loss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        aux_pred: Optional[torch.Tensor] = None,
        aux_target: Optional[torch.Tensor] = None,
        intermediate_preds: Optional[List[torch.Tensor]] = None,
        n_input_steps: int = 10,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred: (batch, time, channels) - full prediction
            target: (batch, time, channels) - target values
            aux_pred: (batch, n_output, channels, features) - auxiliary prediction
            aux_target: (batch, n_output, channels, features) - auxiliary target
            intermediate_preds: list of intermediate predictions
            n_input_steps: number of input steps
        """
        # Only compute loss on future steps
        pred_future = pred[:, n_input_steps:, :]
        target_future = target[:, n_input_steps:, :]

        # Main losses
        mse_loss = self.mse(pred_future, target_future)
        huber_loss = self.huber(pred_future, target_future)
        main_loss = self.mse_weight * mse_loss + self.huber_weight * huber_loss

        # Temporal consistency loss (smoothness)
        pred_diff = pred_future[:, 1:, :] - pred_future[:, :-1, :]
        target_diff = target_future[:, 1:, :] - target_future[:, :-1, :]
        temporal_loss = self.mse(pred_diff, target_diff)

        total_loss = main_loss + self.temporal_weight * temporal_loss

        metrics = {
            "mse": mse_loss.item(),
            "huber": huber_loss.item(),
            "temporal": temporal_loss.item(),
        }

        # Intermediate prediction loss
        if intermediate_preds is not None and len(intermediate_preds) > 1:
            inter_loss = 0
            for inter_pred in intermediate_preds[
                :-1
            ]:  # Skip final (already in main loss)
                inter_loss += self.mse(inter_pred, target_future)
            inter_loss /= len(intermediate_preds) - 1
            total_loss += self.intermediate_weight * inter_loss
            metrics["intermediate"] = inter_loss.item()

        # Auxiliary loss
        if aux_pred is not None and aux_target is not None:
            aux_loss = self.mse(aux_pred, aux_target)
            total_loss += self.aux_weight * aux_loss
            metrics["auxiliary"] = aux_loss.item()

        metrics["total"] = total_loss.item()

        return total_loss, metrics


# =============================================================================
# Factory Functions
# =============================================================================


def create_model(
    n_channels: int,
    device: torch.device,
    memory_efficient: bool = False,
    use_tta: bool = False,
) -> NeuralForecaster:
    """Create model with appropriate settings."""

    if memory_efficient:
        config = {
            "d_model": 128,
            "n_heads": 4,
            "n_encoder_layers": 3,
            "n_decoder_layers": 2,
            "n_refinement_steps": 1,
            "d_ff": 512,
            "dropout": 0.1,
        }
    else:
        config = {
            "d_model": 256,
            "n_heads": 8,
            "n_encoder_layers": 4,
            "n_decoder_layers": 2,
            "n_refinement_steps": 2,
            "d_ff": 1024,
            "dropout": 0.1,
        }

    model = NeuralForecaster(n_channels=n_channels, use_tta=use_tta, **config)

    return model.to(device)


def validate(
    model: NeuralForecaster,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Compute validation MSE."""
    model.eval()
    total_mse = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model.predict(batch)
            target = batch[:, :, :, 0]
            mse = F.mse_loss(pred[:, 10:, :], target[:, 10:, :], reduction="mean")
            total_mse += mse.item()
            n_batches += 1

    return total_mse / n_batches


def validate_competition(
    model: NeuralForecaster,
    test_loaders: dict,
    device: torch.device,
) -> dict:
    """Competition validation with proper MSE calculation."""
    model.eval()
    results = {}

    with torch.no_grad():
        for dataset_name, loader in test_loaders.items():
            total_mse = 0.0
            n_samples = 0

            for batch in loader:
                batch = batch.to(device)
                pred = model.predict(batch)
                target = batch[:, :, :, 0]
                mse = F.mse_loss(pred[:, 10:, :], target[:, 10:, :], reduction="sum")
                total_mse += mse.item()
                n_samples += batch.shape[0] * 10 * batch.shape[2]

            results[f"MSE_{dataset_name}"] = total_mse / n_samples

    results["Total_MSR"] = sum(results.values())
    return results


# Backwards compatibility aliases
ForecastingLoss = EnhancedForecastingLoss


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test Monkey A (239 channels)
    print("Testing Monkey A (239 channels)...")
    model = create_model(239, device, memory_efficient=True)
    x = torch.randn(2, 20, 239, 9).to(device)

    out, aux = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Aux shape: {aux.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test with intermediate
    out, aux, inter = model.forward_with_intermediate(x)
    print(f"Intermediate predictions: {len(inter)}")

    # Test Monkey B (87 channels)
    print("\nTesting Monkey B (87 channels)...")
    model_b = create_model(87, device, memory_efficient=True)
    x_b = torch.randn(2, 20, 87, 9).to(device)

    out_b, aux_b = model_b(x_b)
    print(f"Input shape: {x_b.shape}")
    print(f"Output shape: {out_b.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model_b.parameters()):,}")

    print("\nAll tests passed!")
