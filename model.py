"""TLOB — dual-attention transformer for LOB mid-price trend prediction.

My own reimplementation from the paper (Berti & Kasneci 2025,
arXiv 2502.15757), sections 5.1–5.2 and Appendix A. Written from the
paper's description, NOT copied from the official repo — the repo
(``reference-TLOB/``) is only a reference for the data pipeline and
the DeepLOB baseline. If an interviewer asks how any piece works, the
answer is in the docstrings and comments here, because I wrote it.

Architecture (paper §5.2, Fig. 2; Appendix A hyperparameters):

* Input: the last ``T`` LOB snapshots, ``F`` features each
  (FI-2010: F = 40 LOB-only, 144 with the 104 handcrafted features).
  Projected to ``d_model`` (paper's FI-2010 choice: d_model = F, i.e.
  40 / 144), plus sinusoidal positional encoding.
* Each TLOB block, in order:
    1. Bilinear Normalization (§5.2, Tran et al. 2021) — normalizes
       using batch-specific statistics along both axes, so it adapts
       when market conditions shift instead of relying on fixed
       training-set statistics.
    2. Temporal self-attention — attention over the T snapshots.
    3. Spatial (feature) self-attention — attention over the F LOB
       features, treating each feature's T-length series as its token
       (axial attention, same idea as Axial-LOB). This is the
       "dual attention".
    4. An MLPLOB block (§5.1) instead of a plain Transformer FFN:
       feature-mixing MLP (per time step) then temporal-mixing MLP
       (per feature), GeLU + LayerNorm + residual, eqs. 8–9.
* 4 blocks (ablation §7.5: TLOB has 4 temporal + 4 spatial attention
  layers), 1 head (Appendix A: no difference between 1/2/4 heads, so
  heads fixed to 1), Adam, lr 1e-4, sequence size 128 in the paper.
  Here the sequence length is a constructor argument (default 100,
  the DeepLOB window, so the comparison uses the same windows).
* Head: the representation of the FINAL time step (the paper: the
  network "outputs the directional trend for the final time step")
  through fully connected layers down to 3 logits (down/flat/up).

Design choices the paper does NOT pin down (mine, documented):
* Pre-norm residuals (LayerNorm before each sub-layer) for training
  stability; the paper does not specify pre- vs post-norm.
* MLP expansion factor 4 in the mixing MLPs (MLP-Mixer convention);
  the paper gives no factor.
* BilinearNorm implemented as sequential z-scores (time axis, then
  feature axis) with a learnable per-feature affine transform. The
  paper cites Tran et al. 2021 for the layer; this is my
  implementation of the described idea, not a transcription of the
  repo's code.
* Spatial attention uses d_model = T (the sequence length) after
  transposing, i.e. each feature attends as a T-dimensional token.
  This is forced by the geometry: attending over F features needs an
  embedding per feature, and the only per-feature vector available is
  its time series.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class BilinearNorm(nn.Module):
    """Two-axis z-score normalization with a learnable affine transform.

    For a batch ``x`` of shape [B, T, F]: first z-score along the time
    axis (per feature, using statistics pooled over the BATCH and time,
    dims (0, 1)), then z-score along the feature axis (per time step,
    pooled over the BATCH and features, dims (0, 2)). Finally
    ``gamma * x + beta`` with learnable per-feature ``gamma``/``beta``.
    Because the statistics come from the current batch, the layer adapts
    to distribution shift at inference time — the paper's stated reason
    for using it instead of fixed z-scores (prices and sizes have very
    different magnitudes and drift over time). Note this also means
    inference is batch-dependent (no running statistics are kept, same
    as the BiNCTABL reference implementation); a single sample still
    works because each statistic pools over T or F as well as B.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Time axis: mean/std over (batch, time), per feature.
        mu_t = x.mean(dim=(0, 1), keepdim=True)
        sd_t = x.std(dim=(0, 1), keepdim=True, unbiased=False)
        x = (x - mu_t) / (sd_t + self.eps)
        # Feature axis: mean/std over (batch, features), per time step.
        mu_f = x.mean(dim=(0, 2), keepdim=True)
        sd_f = x.std(dim=(0, 2), keepdim=True, unbiased=False)
        x = (x - mu_f) / (sd_f + self.eps)
        return x * self.gamma + self.beta


class SinusoidalPositionalEncoding(nn.Module):
    """Standard Vaswani et al. (2017) sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TemporalAttention(nn.Module):
    """Self-attention over the time axis (snapshots).

    Input/output [B, T, d_model]. ``key_padding_mask`` is a [B, T]
    bool tensor (True = ignore this snapshot); used by the masking
    test to prove masked snapshots cannot influence the output.
    """

    def __init__(self, d_model: int, nhead: int = 1, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm(x)
        out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask,
                           need_weights=False)
        return x + self.dropout(out)


class SpatialAttention(nn.Module):
    """Self-attention over the feature axis (LOB levels / features).

    Input/output [B, T, d_model]. Transposes to [B, F, T] so each
    feature becomes a token whose embedding is its T-length series,
    attends over the F tokens, transposes back. ``key_padding_mask``
    is a [B, F] bool tensor (True = ignore this feature).
    """

    def __init__(self, seq_len: int, nhead: int = 1, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            seq_len, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, F] -> [B, F, T]; LayerNorm over the last (time) dim.
        xt = x.transpose(1, 2)
        h = self.norm(xt)
        out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask,
                           need_weights=False)
        return x + self.dropout(out.transpose(1, 2))


class _MixingMLP(nn.Module):
    """One mixing MLP of the MLPLOB block (paper §5.1, eqs. 8–9).

    Two fully connected layers with GeLU, residual, LayerNorm, then a
    final GeLU — following the equation's structure
    ``σ(LayerNorm(σ(X W1) W2 + X))``. ``dim`` is the axis being mixed
    (features for feature-mixing, time steps for temporal-mixing).
    """

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, expansion * dim)
        self.fc2 = nn.Linear(expansion * dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.gelu(self.fc1(x))
        h = self.fc2(h)
        h = self.norm(h + x)
        return torch.nn.functional.gelu(self.dropout(h))


class MLPLOBBlock(nn.Module):
    """Feature-mixing then temporal-mixing MLPs (paper §5.1).

    Feature-mixing acts per time step (mixes the feature axis);
    temporal-mixing acts per feature (mixes the time axis). Applied
    after the two attentions in each TLOB block, replacing the usual
    Transformer feed-forward network.
    """

    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.feature_mix = _MixingMLP(d_model, dropout=dropout)
        self.temporal_mix = _MixingMLP(seq_len, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature mixing: [B, T, F] row by row (per time step).
        x = self.feature_mix(x)
        # Temporal mixing: transpose to [B, F, T], mix per feature.
        x = self.temporal_mix(x.transpose(1, 2)).transpose(1, 2)
        return x


class TLOBBlock(nn.Module):
    """One TLOB block: BilinearNorm -> temporal attn -> spatial attn
    -> MLPLOB block (paper §5.2, Fig. 2 caption)."""

    def __init__(self, seq_len: int, d_model: int, nhead: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.norm = BilinearNorm(d_model)
        self.temporal = TemporalAttention(d_model, nhead, dropout)
        self.spatial = SpatialAttention(seq_len, nhead, dropout)
        self.mlp = MLPLOBBlock(seq_len, d_model, dropout)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.norm(x)
        x = self.temporal(x, key_padding_mask=key_padding_mask)
        x = self.spatial(x)
        return self.mlp(x)


class TLOB(nn.Module):
    """Dual-attention transformer for LOB trend prediction.

    Input ``[B, T, F]`` -> output ``[B, 3]`` logits (down/flat/up).
    ``d_model`` defaults to ``in_features`` (the paper's FI-2010
    choice: hidden dim 40 with LOB-only features, 144 with the
    handcrafted features). ``num_blocks`` = 4 and ``nhead`` = 1 follow
    Appendix A / §7.5.
    """

    def __init__(self, in_features: int, seq_len: int = 100,
                 d_model: int | None = None, num_blocks: int = 4,
                 nhead: int = 1, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model or in_features
        self.input_proj = nn.Linear(in_features, self.d_model)
        self.pos = SinusoidalPositionalEncoding(self.d_model)
        self.blocks = nn.ModuleList(
            [TLOBBlock(seq_len, self.d_model, nhead, dropout)
             for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, F] -> [B, T, d_model]
        x = self.input_proj(x)
        x = self.pos(x)
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        # Final time step's representation -> classification head.
        return self.head(x[:, -1, :])


if __name__ == "__main__":
    torch.manual_seed(0)
    model = TLOB(in_features=40, seq_len=100)
    x = torch.randn(8, 100, 40)
    print("params:", sum(p.numel() for p in model.parameters()))
    print("out:", model(x).shape)
    print("logits[0]:", model(x)[0].detach().numpy().round(3))
