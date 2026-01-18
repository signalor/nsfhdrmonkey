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
8. Multi-scale Temporal Processing - captures different temporal patterns
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Component 1: Enhanced Reversible Instance Normalization
# With session-specific adaptation layers
# =============================================================================


class EnhancedRevIN(nn.Module):
    """
    Enhanced Reversible Instance Normalization with:
    - Learnable affine parameters
    - Optional session-specific adaptation
    - Robust statistics computation
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        subtract_last: bool = False,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

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
# Component 2: Channel-Aware Positional Encoding
# Learns relationships between channels
# =============================================================================


class ChannelAwarePositionalEncoding(nn.Module):
    """
    Positional encoding that captures:
    - Temporal position (sinusoidal + learnable)
    - Channel identity (learnable embeddings)
    - Channel-channel relationships (learnable pairwise)
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
        self.use_sinusoidal = use_sinusoidal

        # Learnable temporal embeddings
        self.time_embed = nn.Parameter(torch.randn(1, max_time, 1, d_model // 2) * 0.02)

        # Sinusoidal temporal encoding
        if use_sinusoidal:
            pe = torch.zeros(max_time, d_model // 2)
            position = torch.arange(0, max_time, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model // 2, 2).float()
                * (-math.log(10000.0) / (d_model // 2))
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("sinusoidal_pe", pe.unsqueeze(0).unsqueeze(2))

        # Learnable channel embeddings
        self.channel_embed = nn.Parameter(
            torch.randn(1, 1, max_channels, d_model) * 0.02
        )

        # Channel relationship matrix (captures cross-channel patterns)
        self.channel_relation = nn.Parameter(
            torch.eye(max_channels).unsqueeze(0).unsqueeze(0) * 0.1
        )
        self.relation_proj = nn.Linear(max_channels, d_model // 4)

        # Final projection to combine all encodings
        # x + channel_encoding: d_model
        # time_encoding: d_model // 2 (learnable) + d_model // 2 (sinusoidal if enabled)
        # relation_encoding: d_model // 4
        total_dim = d_model + d_model // 2 + d_model // 4  # base: 256 + 128 + 64 = 448
        if use_sinusoidal:
            total_dim += d_model // 2  # add sinusoidal: 448 + 128 = 576
        self.output_proj = nn.Linear(total_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, channels, d_model)
        """
        B, T, C, D = x.shape

        # Combine temporal encodings
        time_encoding = self.time_embed[:, :T, :, :].expand(B, -1, C, -1)
        if self.use_sinusoidal:
            time_encoding = torch.cat(
                [time_encoding, self.sinusoidal_pe[:, :T, :, :].expand(B, -1, C, -1)],
                dim=-1,
            )

        # Channel embeddings
        channel_encoding = self.channel_embed[:, :, :C, :].expand(B, T, -1, -1)

        # Channel relationship encoding
        relation_weights = self.channel_relation[:, :, :C, :C]  # (1, 1, C, C)
        relation_encoding = self.relation_proj(
            relation_weights.squeeze(0).squeeze(0)
        )  # (C, d//4)
        relation_encoding = (
            relation_encoding.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        )

        # Combine all encodings
        combined = torch.cat(
            [
                x + channel_encoding,
                time_encoding,
                relation_encoding,
            ],
            dim=-1,
        )

        # Project to d_model
        out = self.output_proj(combined)
        return self.norm(out)


# =============================================================================
# Component 3: Frequency Band Attention
# Explicit modeling of cross-frequency dynamics
# =============================================================================


class FrequencyBandAttention(nn.Module):
    """
    Attention mechanism that models relationships between frequency bands.
    The input features are: [target, band1, band2, ..., band8]
    This module learns how different frequency bands interact.
    """

    def __init__(
        self, n_features: int, d_model: int, n_heads: int = 4, dropout: float = 0.1
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Embed each frequency band separately
        self.band_embeddings = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(n_features)]
        )

        # Cross-frequency attention
        self.freq_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Learnable frequency position encoding
        self.freq_pos = nn.Parameter(torch.randn(1, n_features, d_model) * 0.02)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model * n_features, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        # Gating mechanism for feature fusion
        self.gate = nn.Sequential(
            nn.Linear(d_model * n_features, d_model), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, channels, features)
        Returns:
            (batch, time, channels, d_model)
        """
        B, T, C, F = x.shape

        # Reshape to process all (B*T*C) positions together
        x_flat = x.reshape(B * T * C, F, 1)  # (B*T*C, F, 1)

        # Embed each frequency band
        band_embeds = []
        for i, embed in enumerate(self.band_embeddings):
            band_embeds.append(embed(x_flat[:, i : i + 1, :]))  # (B*T*C, 1, d_model)

        freq_tokens = torch.cat(band_embeds, dim=1)  # (B*T*C, F, d_model)
        freq_tokens = freq_tokens + self.freq_pos

        # Apply cross-frequency attention
        attn_out, _ = self.freq_attention(freq_tokens, freq_tokens, freq_tokens)

        # Flatten and project
        attn_flat = attn_out.reshape(B * T * C, -1)  # (B*T*C, F*d_model)

        # Gated fusion
        gate = self.gate(attn_flat)
        output = self.output_proj(attn_flat) * gate

        return output.reshape(B, T, C, -1)


# =============================================================================
# Component 4: Gated Residual Block
# Improved gradient flow with learnable gating
# =============================================================================


class GatedResidualBlock(nn.Module):
    """Residual block with learnable gate for adaptive skip connection."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, residual], dim=-1)
        gate = self.gate(combined)
        return self.dropout(x) * gate + residual * (1 - gate)


# =============================================================================
# Component 5: Multi-Scale Temporal Convolution
# Captures patterns at different time scales
# =============================================================================


class MultiScaleTemporalConv(nn.Module):
    """
    Multi-scale temporal convolutions to capture patterns at different scales.
    Uses dilated convolutions for larger receptive fields.
    """

    def __init__(
        self,
        d_model: int,
        kernel_sizes: List[int] = [3, 5, 7],
        dilations: List[int] = [1, 2, 4],
        dropout: float = 0.1,
    ):
        super().__init__()

        n_scales = len(kernel_sizes) * len(dilations)
        hidden_per_scale = d_model // n_scales

        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            for d in dilations:
                padding = (k - 1) * d // 2
                # Use GroupNorm instead of BatchNorm for DataParallel compatibility
                # Find a valid num_groups that divides hidden_per_scale
                num_groups = 1
                for g in [8, 4, 2, 1]:
                    if hidden_per_scale % g == 0:
                        num_groups = g
                        break
                self.convs.append(
                    nn.Sequential(
                        nn.Conv1d(
                            d_model, hidden_per_scale, k, padding=padding, dilation=d
                        ),
                        nn.GELU(),
                        nn.GroupNorm(num_groups, hidden_per_scale),
                    )
                )

        self.proj = nn.Linear(hidden_per_scale * n_scales, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, channels, d_model)
        """
        B, T, C, D = x.shape

        # Process each channel independently
        x = x.permute(0, 2, 3, 1).reshape(B * C, D, T)  # (B*C, D, T)

        # Apply multi-scale convolutions
        conv_outs = [conv(x) for conv in self.convs]
        x = torch.cat(conv_outs, dim=1)  # (B*C, hidden_total, T)

        x = x.transpose(1, 2)  # (B*C, T, hidden_total)
        x = self.proj(x)  # (B*C, T, D)
        x = x.reshape(B, C, T, D).permute(0, 2, 1, 3)  # (B, T, C, D)

        return self.norm(self.dropout(x))


# =============================================================================
# Component 6: Enhanced Spatial-Temporal Attention
# =============================================================================


class EnhancedSpatialAttention(nn.Module):
    """Spatial attention with relative position bias."""

    def __init__(
        self, d_model: int, n_heads: int, max_channels: int, dropout: float = 0.1
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Relative position bias for channel attention
        self.rel_pos_bias = nn.Parameter(
            torch.zeros(n_heads, max_channels, max_channels)
        )
        nn.init.xavier_uniform_(self.rel_pos_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, D = x.shape
        x_flat = x.reshape(B * T, C, D)

        # Add relative position bias as 2D mask (C, C) - broadcasts across batch and heads
        attn_bias = self.rel_pos_bias[:, :C, :C].mean(dim=0)  # (C, C)

        attn_out, _ = self.attn(x_flat, x_flat, x_flat, attn_mask=attn_bias)
        attn_out = self.dropout(attn_out)
        x_flat = self.norm(x_flat + attn_out)

        return x_flat.reshape(B, T, C, D)


class EnhancedTemporalAttention(nn.Module):
    """Temporal attention with causal masking option."""

    def __init__(self, d_model: int, n_heads: int, max_time: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Learnable temporal bias
        self.temporal_bias = nn.Parameter(torch.zeros(n_heads, max_time, max_time))
        nn.init.xavier_uniform_(self.temporal_bias)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, T, C, D = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(B * C, T, D)

        # Create attention mask
        if causal:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            mask = mask.float().masked_fill(mask, float("-inf"))
        else:
            mask = None

        attn_out, _ = self.attn(x_flat, x_flat, x_flat, attn_mask=mask)
        attn_out = self.dropout(attn_out)
        x_flat = self.norm(x_flat + attn_out)

        return x_flat.reshape(B, C, T, D).permute(0, 2, 1, 3)


class EnhancedSpatialTemporalBlock(nn.Module):
    """Enhanced block with multi-scale convolutions and gated residuals."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_channels: int,
        max_time: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.spatial_attn = EnhancedSpatialAttention(
            d_model, n_heads, max_channels, dropout
        )
        self.temporal_attn = EnhancedTemporalAttention(
            d_model, n_heads, max_time, dropout
        )
        self.multi_scale_conv = MultiScaleTemporalConv(d_model, dropout=dropout)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(d_model)

        # Gated residuals
        self.gate1 = GatedResidualBlock(d_model, dropout)
        self.gate2 = GatedResidualBlock(d_model, dropout)
        self.gate3 = GatedResidualBlock(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial attention with gated residual
        spatial_out = self.spatial_attn(x)
        x = self.gate1(spatial_out, x)

        # Temporal attention with gated residual
        temporal_out = self.temporal_attn(x)
        x = self.gate2(temporal_out, x)

        # Multi-scale convolution
        conv_out = self.multi_scale_conv(x)
        x = x + conv_out * 0.5  # Scaled addition

        # Feed-forward with gated residual
        B, T, C, D = x.shape
        x_flat = x.reshape(B * T * C, D)
        ff_out = self.ff(x_flat).reshape(B, T, C, D)
        ff_out = self.ff_norm(ff_out)
        x = self.gate3(ff_out, x)

        return x


# =============================================================================
# Component 7: Autoregressive Refinement Decoder
# Iteratively refines predictions
# =============================================================================


class AutoregressiveRefinementDecoder(nn.Module):
    """
    Decoder that refines predictions through multiple iterations.
    Uses cross-attention to encoder output and self-attention among predictions.
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

        # Initial future queries
        self.future_queries = nn.Parameter(
            torch.randn(1, n_future, n_channels, d_model) * 0.02
        )

        # Main decoder layers
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        # Refinement layers (share weights across refinement steps)
        self.refinement_layer = DecoderLayer(d_model, n_heads, dropout)

        # Initial prediction head
        self.initial_pred = nn.Linear(d_model, 1)

        # Refinement embedding (projects initial prediction back to d_model)
        self.refine_embed = nn.Linear(1, d_model)

        # Final output
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, encoder_output: torch.Tensor, return_intermediate: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            encoder_output: (batch, time, n_channels, d_model)
            return_intermediate: whether to return intermediate predictions
        Returns:
            decoded: (batch, n_future, n_channels, d_model)
            intermediate_preds: list of intermediate predictions if requested
        """
        B = encoder_output.shape[0]
        intermediate_preds = []

        # Start with learned queries
        x = self.future_queries.expand(B, -1, -1, -1)

        # Initial decoding
        for layer in self.decoder_layers:
            x = layer(x, encoder_output)

        # Get initial prediction
        initial_pred = self.initial_pred(x).squeeze(-1)  # (B, n_future, C)
        if return_intermediate:
            intermediate_preds.append(initial_pred)

        # Refinement iterations
        for step in range(self.n_refinement_steps):
            # Embed prediction back into representation space
            pred_embed = self.refine_embed(
                initial_pred.unsqueeze(-1)
            )  # (B, n_future, C, d_model)

            # Combine with current representation
            x = x + pred_embed * 0.5

            # Refine through attention
            x = self.refinement_layer(x, encoder_output)

            # Get refined prediction
            refined_pred = self.initial_pred(x).squeeze(-1)

            # Residual refinement
            initial_pred = initial_pred + refined_pred * 0.3

            if return_intermediate:
                intermediate_preds.append(initial_pred)

        return self.norm(x), intermediate_preds


class DecoderLayer(nn.Module):
    """Single decoder layer with self and cross attention."""

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

        # Self attention
        attn_out, _ = self.self_attn(x_flat, x_flat, x_flat)
        x_flat = self.norm1(x_flat + self.dropout(attn_out))

        # Cross attention
        attn_out, _ = self.cross_attn(x_flat, enc_flat, enc_flat)
        x_flat = self.norm2(x_flat + self.dropout(attn_out))

        # Feed-forward
        x_flat = self.norm3(x_flat + self.ff(x_flat))

        return x_flat.reshape(B, T_fut, C, D)


# =============================================================================
# Component 8: Feature Embedding with Frequency Attention
# =============================================================================


class EnhancedFeatureEmbedding(nn.Module):
    """
    Enhanced feature embedding that:
    - Separately embeds target and frequency bands
    - Uses frequency attention for cross-band modeling
    - Applies gated fusion
    """

    def __init__(
        self, n_features: int, d_model: int, n_heads: int = 4, dropout: float = 0.1
    ):
        super().__init__()
        self.n_features = n_features

        # Individual feature projections
        self.target_proj = nn.Linear(1, d_model // 2)
        self.freq_proj = nn.Linear(n_features - 1, d_model // 2)

        # Frequency band attention
        self.freq_attention = FrequencyBandAttention(
            n_features, d_model // 4, n_heads, dropout
        )

        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(d_model + d_model // 4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Gating for adaptive fusion
        self.gate = nn.Sequential(
            nn.Linear(d_model + d_model // 4, d_model), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, channels, features)
        Returns:
            (batch, time, channels, d_model)
        """
        target = x[..., 0:1]
        freq_bands = x[..., 1:]

        # Embed target and frequency bands
        target_emb = self.target_proj(target)
        freq_emb = self.freq_proj(freq_bands)
        simple_embed = torch.cat([target_emb, freq_emb], dim=-1)

        # Get frequency attention embedding
        freq_attn_embed = self.freq_attention(x)

        # Combine embeddings
        combined = torch.cat([simple_embed, freq_attn_embed], dim=-1)

        # Gated fusion
        gate = self.gate(combined)
        output = self.fusion(combined) * gate + simple_embed * (1 - gate)

        return output


# =============================================================================
# Main Model: EnhancedSpatialTemporalForecaster
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
        self.output_proj = nn.Linear(d_model, 1)
        self.aux_output_proj = nn.Linear(d_model, n_features)

    def forward(
        self, x: torch.Tensor, return_intermediate: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (batch, time, channels, features) - input sequence
        Returns:
            main_output: (batch, n_output_steps, channels) - predicted feature 0
            aux_output: (batch, n_output_steps, channels, features) - all features
            intermediate: list of intermediate predictions if requested
        """
        B, T, C, F = x.shape

        # Apply RevIN normalization
        if self.use_revin:
            x_target = x[..., 0]
            x_target = self.revin(x_target, mode="norm")
            x = torch.cat([x_target.unsqueeze(-1), x[..., 1:]], dim=-1)

        # Feature embedding with frequency attention
        x = self.feature_embed(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Encode
        for layer in self.encoder_layers:
            x = layer(x)

        # Decode with autoregressive refinement
        decoded, intermediate = self.decoder(x, return_intermediate)

        # Project to outputs
        main_output = self.output_proj(decoded).squeeze(-1)
        aux_output = self.aux_output_proj(decoded)

        # Apply RevIN denormalization
        if self.use_revin:
            main_output = self.revin(main_output, mode="denorm")

        return main_output, aux_output, intermediate

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inference mode - returns only main prediction."""
        main_output, _, _ = self.forward(x, return_intermediate=False)
        return main_output


# =============================================================================
# Test-Time Augmentation
# =============================================================================


class TestTimeAugmentation:
    """Test-time augmentation strategies."""

    @staticmethod
    def apply_augmentations(
        x: torch.Tensor, n_augmentations: int = 5
    ) -> List[torch.Tensor]:
        """Generate augmented versions of input."""
        augmented = [x]  # Original

        for i in range(n_augmentations - 1):
            aug = x.clone()

            # Random noise with varying scales
            noise_scale = 0.01 * (i + 1) / n_augmentations
            aug = aug + torch.randn_like(aug) * noise_scale

            # Random scaling per channel (slight)
            if i % 2 == 0:
                scale = (
                    1.0
                    + (torch.rand(1, 1, x.shape[2], 1, device=x.device) - 0.5) * 0.02
                )
                aug = aug * scale

            augmented.append(aug)

        return augmented

    @staticmethod
    def aggregate_predictions(
        predictions: List[torch.Tensor], method: str = "mean"
    ) -> torch.Tensor:
        """Aggregate predictions from augmented inputs."""
        stacked = torch.stack(predictions, dim=0)

        if method == "mean":
            return stacked.mean(dim=0)
        elif method == "median":
            return stacked.median(dim=0)[0]
        elif method == "trimmed_mean":
            # Remove highest and lowest, then mean
            sorted_preds, _ = torch.sort(stacked, dim=0)
            return sorted_preds[1:-1].mean(dim=0)
        else:
            return stacked.mean(dim=0)


# =============================================================================
# Ensemble Model
# =============================================================================


class EnsembleForecaster(nn.Module):
    """Ensemble of multiple models with learned weights."""

    def __init__(self, models: List[nn.Module], learnable_weights: bool = False):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_models = len(models)

        if learnable_weights:
            self.weights = nn.Parameter(torch.ones(self.n_models) / self.n_models)
        else:
            self.register_buffer("weights", torch.ones(self.n_models) / self.n_models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predictions = []
        for model in self.models:
            pred = model.predict(x)
            predictions.append(pred)

        # Weighted average
        weights = F.softmax(self.weights, dim=0)
        stacked = torch.stack(predictions, dim=0)
        output = (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)

        return output


# =============================================================================
# Enhanced Loss Functions
# =============================================================================


class EnhancedForecastingLoss(nn.Module):
    """
    Enhanced loss with:
    - MSE on target feature
    - Auxiliary loss on frequency bands
    - Temporal smoothness
    - Intermediate prediction supervision
    """

    def __init__(
        self,
        aux_weight: float = 0.3,
        consistency_weight: float = 0.1,
        intermediate_weight: float = 0.2,
    ):
        super().__init__()
        self.aux_weight = aux_weight
        self.consistency_weight = consistency_weight
        self.intermediate_weight = intermediate_weight
        self.mse = nn.MSELoss()
        self.huber = nn.SmoothL1Loss()

    def forward(
        self,
        main_pred: torch.Tensor,
        aux_pred: torch.Tensor,
        target: torch.Tensor,
        target_full: Optional[torch.Tensor] = None,
        intermediate_preds: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            main_pred: (B, T, C) - predicted feature 0
            aux_pred: (B, T, C, F) - predicted all features
            target: (B, T, C) - ground truth feature 0
            target_full: (B, T, C, F) - ground truth all features
            intermediate_preds: list of intermediate predictions
        """
        # Main loss (combination of MSE and Huber for robustness)
        main_loss = 0.7 * self.mse(main_pred, target) + 0.3 * self.huber(
            main_pred, target
        )

        # Auxiliary loss
        aux_loss = torch.tensor(0.0, device=main_pred.device)
        if target_full is not None and self.aux_weight > 0:
            aux_loss = self.mse(aux_pred, target_full)

        # Temporal consistency
        consistency_loss = torch.tensor(0.0, device=main_pred.device)
        if self.consistency_weight > 0:
            pred_diff = main_pred[:, 1:, :] - main_pred[:, :-1, :]
            target_diff = target[:, 1:, :] - target[:, :-1, :]
            consistency_loss = self.mse(pred_diff, target_diff)

        # Intermediate supervision
        intermediate_loss = torch.tensor(0.0, device=main_pred.device)
        if (
            intermediate_preds is not None
            and len(intermediate_preds) > 0
            and self.intermediate_weight > 0
        ):
            for i, inter_pred in enumerate(intermediate_preds):
                # Decreasing weight for earlier predictions
                weight = (i + 1) / len(intermediate_preds)
                intermediate_loss = intermediate_loss + weight * self.mse(
                    inter_pred, target
                )
            intermediate_loss = intermediate_loss / len(intermediate_preds)

        total_loss = (
            main_loss
            + self.aux_weight * aux_loss
            + self.consistency_weight * consistency_loss
            + self.intermediate_weight * intermediate_loss
        )

        metrics = {
            "main_loss": main_loss.item(),
            "aux_loss": aux_loss.item(),
            "consistency_loss": consistency_loss.item(),
            "intermediate_loss": intermediate_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, metrics


# =============================================================================
# Data Augmentation
# =============================================================================


class EnhancedDataAugmentation:
    """Enhanced data augmentation strategies."""

    @staticmethod
    def channel_dropout(x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
        """Randomly zero out entire channels."""
        if not torch.is_grad_enabled():
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
    def channel_shuffle(x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
        """Randomly shuffle a subset of channels (within neighborhoods)."""
        if not torch.is_grad_enabled() or torch.rand(1).item() > p:
            return x
        B, T, C, F = x.shape
        # Only shuffle small groups of adjacent channels
        group_size = min(5, C // 4)
        x = x.clone()
        for b in range(B):
            if torch.rand(1).item() < p:
                start = torch.randint(0, C - group_size, (1,)).item()
                perm = torch.randperm(group_size)
                x[b, :, start : start + group_size, :] = x[b, :, start + perm, :]
        return x

    @staticmethod
    def temporal_mask(
        x: torch.Tensor, p: float = 0.1, mask_ratio: float = 0.2
    ) -> torch.Tensor:
        """Randomly mask portions of the temporal sequence."""
        if not torch.is_grad_enabled() or torch.rand(1).item() > p:
            return x
        B, T, C, F = x.shape
        x = x.clone()
        mask_len = int(T * mask_ratio)
        for b in range(B):
            if torch.rand(1).item() < p:
                start = torch.randint(0, T - mask_len, (1,)).item()
                # Replace with linear interpolation
                x[b, start : start + mask_len] = torch.lerp(
                    x[b, start - 1 : start],
                    x[b, start + mask_len : start + mask_len + 1],
                    torch.linspace(0, 1, mask_len, device=x.device).view(-1, 1, 1),
                )
        return x

    @staticmethod
    def frequency_dropout(x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
        """Randomly zero out frequency bands (but never the target)."""
        if not torch.is_grad_enabled():
            return x
        B, T, C, F = x.shape
        if F <= 1:
            return x
        x = x.clone()
        mask = torch.ones(B, 1, 1, F, device=x.device)
        mask[:, :, :, 1:] = (torch.rand(B, 1, 1, F - 1, device=x.device) > p).float()
        return x * mask

    @staticmethod
    def mixup(
        x1: torch.Tensor, x2: torch.Tensor, alpha: float = 0.2
    ) -> Tuple[torch.Tensor, float]:
        """Mixup two samples."""
        lam = np.random.beta(alpha, alpha)
        return lam * x1 + (1 - lam) * x2, lam


# =============================================================================
# Neural Forecaster Wrapper
# =============================================================================


class NeuralForecaster(nn.Module):
    """
    Complete wrapper for the competition.
    """

    def __init__(
        self,
        n_channels: int,
        memory_efficient: bool = False,
        use_tta: bool = False,
        n_tta_augmentations: int = 5,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.use_tta = use_tta
        self.n_tta_augmentations = n_tta_augmentations

        # Model configuration
        if memory_efficient:
            d_model = 128
            n_heads = 4
            n_encoder_layers = 2
            n_decoder_layers = 1
            n_refinement_steps = 1
            d_ff = 256
        else:
            d_model = 256
            n_heads = 8
            n_encoder_layers = 4
            n_decoder_layers = 2
            n_refinement_steps = 2
            d_ff = 1024

        self.model = EnhancedSpatialTemporalForecaster(
            n_channels=n_channels,
            n_features=9,
            n_input_steps=10,
            n_output_steps=10,
            d_model=d_model,
            n_heads=n_heads,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            n_refinement_steps=n_refinement_steps,
            d_ff=d_ff,
            dropout=0.15,
            use_revin=True,
        )

        self.augmentation = EnhancedDataAugmentation()
        self.tta = TestTimeAugmentation()

    def forward(
        self, x: torch.Tensor, augment: bool = False, return_intermediate: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, 20, channels, 9) - full sequence
            augment: whether to apply data augmentation
            return_intermediate: if True, returns (main_pred, aux_pred, intermediate)
                                 for training loss computation
        Returns:
            If return_intermediate=False: (batch, 20, channels) - prediction for feature 0
            If return_intermediate=True: tuple of (main_pred, aux_pred, intermediate)
        """
        x_input = x[:, :10, :, :]

        # Apply augmentation during training
        if augment and self.training:
            x_input = self.augmentation.channel_dropout(x_input, p=0.1)
            x_input = self.augmentation.time_noise(x_input, std=0.03)
            x_input = self.augmentation.frequency_dropout(x_input, p=0.1)

        # Get predictions
        main_pred, aux_pred, intermediate = self.model(
            x_input, return_intermediate=return_intermediate or self.training
        )

        if return_intermediate:
            return main_pred, aux_pred, intermediate

        # Concatenate input and prediction
        input_feature0 = x[:, :10, :, 0]
        full_pred = torch.cat([input_feature0, main_pred], dim=1)

        return full_pred

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inference mode with optional TTA."""
        self.eval()
        with torch.no_grad():
            if self.use_tta:
                x_input = x[:, :10, :, :]
                augmented_inputs = self.tta.apply_augmentations(
                    x_input, self.n_tta_augmentations
                )

                predictions = []
                for aug_x in augmented_inputs:
                    main_pred, _, _ = self.model(aug_x, return_intermediate=False)
                    predictions.append(main_pred)

                main_pred = self.tta.aggregate_predictions(
                    predictions, method="trimmed_mean"
                )
                input_feature0 = x[:, :10, :, 0]
                return torch.cat([input_feature0, main_pred], dim=1)
            else:
                return self.forward(x, augment=False)

    def forward_with_intermediate(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Forward pass returning intermediate predictions for loss computation.

        Returns:
            main_pred: (B, 10, C) - predicted feature 0 for future timesteps only
            aux_pred: (B, 10, C, F) - predicted all features for future timesteps
            intermediate: list of intermediate predictions (B, 10, C) each
        """
        return self.forward(x, augment=self.training, return_intermediate=True)


# =============================================================================
# Factory Function
# =============================================================================


def create_model(
    n_channels: int,
    device: torch.device,
    memory_efficient: bool = False,
    use_tta: bool = False,
) -> NeuralForecaster:
    """Factory function to create model."""
    model = NeuralForecaster(
        n_channels=n_channels, memory_efficient=memory_efficient, use_tta=use_tta
    )
    return model.to(device)


# =============================================================================
# Training Utilities
# =============================================================================


def train_step(
    model: NeuralForecaster,
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: EnhancedForecastingLoss,
    device: torch.device,
) -> dict:
    """Single training step."""
    model.train()
    batch = batch.to(device)

    optimizer.zero_grad()

    # Forward pass with intermediate predictions
    main_pred, aux_pred, intermediate = model.forward_with_intermediate(batch)

    # Targets
    target = batch[:, 10:, :, 0]
    target_full = batch[:, 10:, :, :]

    # Compute loss
    loss, metrics = loss_fn(main_pred, aux_pred, target, target_full, intermediate)

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

    # Test Monkey A
    model = create_model(239, device, memory_efficient=True)
    x = torch.randn(4, 20, 239, 9).to(device)

    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
