from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Lightweight CNN configuration for OCEC."""

    base_channels: int = 32
    num_blocks: int = 4
    dropout: float = 0.3
    arch_variant: str = "baseline"
    expansion: int = 4
    se_reduction: int = 8
    head_variant: str = "auto"
    token_mixer_grid: tuple[int, int] = (2, 3)
    token_mixer_layers: int = 2


class _SepConvBlock(nn.Module):
    """Depthwise separable convolution block with optional residual."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.use_residual:
            x = x + identity
        x = F.relu(x, inplace=True)
        return x


class _SqueezeExcite(nn.Module):
    """Squeeze-and-excitation attention module."""

    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        reduced = max(1, channels // max(1, reduction))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.fc(scale)
        return x * scale


class _InvertedResidualSEBlock(nn.Module):
    """MobileNetV2-style inverted residual block with squeeze-and-excite."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 4,
        se_reduction: int = 8,
    ) -> None:
        super().__init__()
        if expansion < 1:
            raise ValueError("Expansion factor must be >= 1.")
        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expansion

        layers = []
        if expansion != 1:
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.SiLU(inplace=True),
                ]
            )
        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
            ]
        )
        if se_reduction > 0:
            layers.append(_SqueezeExcite(hidden_dim, se_reduction))
        layers.extend(
            [
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.block = nn.Sequential(*layers)
        self.out_act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.block(x)
        if self.use_residual:
            out = out + identity
        out = self.out_act(out)
        return out


class _LayerNorm2d(nn.Module):
    """LayerNorm operating on channels-last tensors but returning channels-first."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class _ConvNeXtBlock(nn.Module):
    """ConvNeXt-style block with depthwise conv and channel MLP."""

    def __init__(self, dim: int, layer_scale_init: float = 1e-6) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = _LayerNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim * 4, dim)
        if layer_scale_init > 0:
            self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
        else:
            self.gamma = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = x + shortcut
        return x


class _ConvNeXtDownsample(nn.Module):
    """Downsample layer used between ConvNeXt stages."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("ConvNeXt downsample stride must be 1 or 2.")
        if stride == 1 and in_channels == out_channels:
            self.op = None
        else:
            kernel = 2 if stride == 2 else 1
            self.op = nn.Sequential(
                _LayerNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.op is None:
            return x
        return self.op(x)


class _MLPMixerBlock(nn.Module):
    """Lightweight MLP-Mixer block operating on a fixed token grid."""

    def __init__(
        self,
        num_tokens: int,
        channels: int,
        dropout: float = 0.0,
        token_expansion: int = 2,
        channel_expansion: int = 2,
    ) -> None:
        super().__init__()
        token_hidden = max(num_tokens, num_tokens * token_expansion)
        channel_hidden = max(channels, channels * channel_expansion)
        self.norm_tokens = nn.LayerNorm(channels)
        self.token_fc1 = nn.Linear(num_tokens, token_hidden)
        self.token_act = nn.GELU()
        self.token_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.token_fc2 = nn.Linear(token_hidden, num_tokens)
        self.norm_channels = nn.LayerNorm(channels)
        self.channel_fc1 = nn.Linear(channels, channel_hidden)
        self.channel_act = nn.GELU()
        self.channel_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.channel_fc2 = nn.Linear(channel_hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, tokens, channels)
        residual = x
        y = self.norm_tokens(x)
        y = y.transpose(1, 2)  # (batch, channels, tokens)
        y = self.token_fc1(y)
        y = self.token_act(y)
        y = self.token_dropout(y)
        y = self.token_fc2(y)
        y = self.token_dropout(y)
        y = y.transpose(1, 2)
        x = residual + y

        y = self.norm_channels(x)
        y = self.channel_fc1(y)
        y = self.channel_act(y)
        y = self.channel_dropout(y)
        y = self.channel_fc2(y)
        y = self.channel_dropout(y)
        return x + y


def _select_attention_heads(channels: int) -> int:
    for candidate in range(min(8, channels), 0, -1):
        if channels % candidate == 0:
            return candidate
    return 1


class _TokenMixerHead(nn.Module):
    """Applies Transformer or MLP-Mixer across pooled spatial tokens before classification."""

    def __init__(
        self,
        channels: int,
        dropout: float,
        mixer_type: str,
        grid: tuple[int, int],
        layers: int,
    ) -> None:
        super().__init__()
        if mixer_type not in {"transformer", "mlp_mixer"}:
            raise ValueError(f"Unsupported token mixer type: {mixer_type}")
        self.grid = grid
        self.channels = channels
        self.num_tokens = grid[0] * grid[1]
        self.mixer_type = mixer_type
        self.layers = max(1, int(layers))
        dropout = float(max(0.0, dropout))

        if mixer_type == "transformer":
            heads = _select_attention_heads(channels)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=channels,
                nhead=heads,
                dim_feedforward=max(128, channels * 2),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.mixer = nn.TransformerEncoder(encoder_layer, num_layers=self.layers)
        else:
            blocks = [
                _MLPMixerBlock(self.num_tokens, channels, dropout=dropout)
                for _ in range(self.layers)
            ]
            self.mixer = nn.Sequential(*blocks)

        self.final_norm = nn.LayerNorm(channels)
        self.final_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, height, width)
        h, w = self.grid
        height, width = x.shape[-2], x.shape[-1]
        if height % h != 0 or width % w != 0:
            raise RuntimeError(
                f"Feature map {height}x{width} is not divisible by requested token grid {h}x{w}. "
                "Adjust --token_mixer_grid or image/architecture settings."
            )
        kernel_h = height // h
        kernel_w = width // w
        tokens = F.avg_pool2d(x, kernel_size=(kernel_h, kernel_w), stride=(kernel_h, kernel_w))
        tokens = tokens.flatten(2).transpose(1, 2)  # (batch, tokens, channels)
        tokens = self.mixer(tokens)
        pooled = tokens.mean(dim=1)
        pooled = self.final_norm(pooled)
        pooled = self.final_dropout(pooled)
        logits = self.fc(pooled)
        return logits


class VSDLM(nn.Module):
    """Compact mouth state classifier that outputs logits."""

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        base = self.config.base_channels
        num_blocks = max(1, self.config.num_blocks)
        variant = (self.config.arch_variant or "baseline").lower()
        head_variant = getattr(self.config, "head_variant", "auto")
        if head_variant:
            head_variant = head_variant.lower()
        if head_variant == "auto":
            if variant == "inverted_se":
                self._head_variant = "avgmax_mlp"
            elif variant == "convnext":
                self._head_variant = "transformer"
            else:
                self._head_variant = "avg"
        else:
            self._head_variant = head_variant
        self._token_grid = self._ensure_token_grid(getattr(self.config, "token_mixer_grid", (3, 2)))
        self._token_layers = int(max(1, getattr(self.config, "token_mixer_layers", 2)))

        if variant == "baseline":
            stem_layers = [
                nn.Conv2d(3, base, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base),
                nn.ReLU(inplace=True),
            ]
        elif variant == "inverted_se":
            stem_layers = [
                nn.Conv2d(3, base, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(base),
                nn.SiLU(inplace=True),
                nn.Conv2d(base, base, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base),
                nn.SiLU(inplace=True),
            ]
        elif variant == "convnext":
            stem_layers = [
                nn.Conv2d(3, base, kernel_size=4, stride=4, padding=0, bias=True),
                _LayerNorm2d(base),
            ]
        else:
            raise ValueError(f"Unsupported architecture variant: {self.config.arch_variant}")
        self._variant = variant
        self.stem = nn.Sequential(*stem_layers)

        channels = base
        blocks = []
        for idx in range(num_blocks):
            stride = 2 if idx % 2 == 0 and idx > 0 else 1
            next_channels = channels * (2 if stride == 2 else 1)
            blocks.append(self._make_block(channels, next_channels, stride))
            channels = next_channels

        self.features = nn.Sequential(*blocks)
        if self._head_variant in {"avg", "avgmax_mlp"}:
            head_in_features = self._head_input_dim(channels)
            self.head = self._build_head(head_in_features)
        elif self._head_variant in {"transformer", "mlp_mixer"}:
            self.head = _TokenMixerHead(
                channels,
                dropout=self.config.dropout,
                mixer_type=self._head_variant,
                grid=self._token_grid,
                layers=self._token_layers,
            )
        else:
            raise ValueError(f"Unsupported head variant: {self._head_variant}")
        self._feature_channels = channels

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        if self._head_variant in {"transformer", "mlp_mixer"}:
            logits = self.head(x)
        else:
            x = self._pool_features(x)
            logits = self.head(x)
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    def _make_block(self, in_channels: int, out_channels: int, stride: int) -> nn.Module:
        if self._variant == "baseline":
            return _SepConvBlock(in_channels, out_channels, stride=stride)
        if self._variant == "inverted_se":
            expansion = int(max(1, getattr(self.config, "expansion", 4)))
            se_reduction = int(max(0, getattr(self.config, "se_reduction", 8)))
            return _InvertedResidualSEBlock(
                in_channels,
                out_channels,
                stride=stride,
                expansion=expansion,
                se_reduction=se_reduction,
            )
        if self._variant == "convnext":
            modules = []
            if stride != 1 or in_channels != out_channels:
                modules.append(_ConvNeXtDownsample(in_channels, out_channels, stride))
            modules.append(_ConvNeXtBlock(out_channels))
            return nn.Sequential(*modules)
        raise ValueError(f"Unsupported architecture variant: {self._variant}")

    def _head_input_dim(self, feature_channels: int) -> int:
        if self._head_variant == "avg":
            return feature_channels
        if self._head_variant == "avgmax_mlp":
            return feature_channels * 2
        raise ValueError(f"Unsupported head variant: {self._head_variant}")

    def _build_head(self, in_features: int) -> nn.Module:
        dropout = float(max(0.0, self.config.dropout))
        if self._head_variant == "avg":
            return nn.Sequential(
                nn.BatchNorm1d(in_features),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(in_features, 1),
            )
        if self._head_variant == "avgmax_mlp":
            hidden = max(16, in_features // 2)
            layers = [
                nn.BatchNorm1d(in_features),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(in_features, hidden),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden, 1))
            return nn.Sequential(*layers)
        raise ValueError(f"Unsupported head variant: {self._head_variant}")

    def _pool_features(self, x: torch.Tensor) -> torch.Tensor:
        if self._head_variant == "avg":
            return x.mean(dim=(-2, -1))
        if self._head_variant == "avgmax_mlp":
            avg = x.mean(dim=(-2, -1))
            maxv = torch.amax(x, dim=(-2, -1))
            return torch.cat([avg, maxv], dim=1)
        raise ValueError(f"Pooling not defined for head variant: {self._head_variant}")

    def _ensure_token_grid(self, grid: Any) -> tuple[int, int]:
        if not isinstance(grid, (tuple, list)) or len(grid) != 2:
            return (3, 2)
        h, w = int(grid[0]), int(grid[1])
        if h <= 0 or w <= 0:
            return (3, 2)
        return (h, w)
