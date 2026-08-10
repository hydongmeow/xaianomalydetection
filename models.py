"""
models.py
=========
Graph-temporal architectures for multivariate ICS anomaly detection
(SWaT / WADI).

Two detectors, one shared scaffold
----------------------------------
    GraphTemporalAttnDetector      temporal_arch in {tcn, gru, rnn, lstm}
    GraphTransformerAttnDetector   temporal_arch in {informer, autoformer,
                                                     vanilla}

Both inherit `_GraphTemporalBase`, so the graph branch, sensor gating,
temporal-attention pooling, residual shortcut and prediction head are
identical across every architecture.  Only the temporal encoder changes,
which is what an architecture comparison should isolate.

Common forward contract
-----------------------
    forward(x, adjacency) -> pred
        x         : (B, W, d)  sliding window of sensor readings
        adjacency : (d, d)     prior graph (from pre_pipeline)
        pred      : (B, d)     predicted next timestep

Every temporal encoder maps (B, W, H) -> (B, W, H); sequence length is
preserved so temporal-attention pooling and the Level-3 XAI overlay work
identically for all architectures.
"""

from typing import Optional, Tuple, Union

import math
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "GraphAttentionLayer",
    "GraphTemporalAttnDetector",
    "GraphTransformerAttnDetector",
    "build_detector",
    "build_temporal_encoder",
    "supports_temporal_attention",
    "count_parameters",
    "get_model_size_mb",
    "measure_latency_cpu",
    "RECURRENT_ARCHS",
    "TRANSFORMER_ARCHS",
]

RECURRENT_ARCHS = ("tcn", "gru", "rnn", "lstm")
TRANSFORMER_ARCHS = ("informer", "autoformer", "vanilla")


# =============================================================================
# TCN building blocks
# =============================================================================

class _Chomp1d(nn.Module):
    """Remove the last `chomp_size` elements along the time axis."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size <= 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class _TCNBlock(nn.Module):
    """
    Causal dilated convolution block with residual connection.

    GroupNorm(1, C) is used instead of BatchNorm1d: it behaves identically in
    train and eval mode and is stable at any batch size.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dilation: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation            # causal left padding
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=pad, dilation=dilation)
        self.chomp = _Chomp1d(pad)                    # drop the right-side leak
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop(self.act(self.norm(self.chomp(self.conv(x)))))
        return out + self.residual(x)


# =============================================================================
# Temporal encoders — all map (B, W, H) -> (B, W, H)
# =============================================================================

class TemporalEncoderTCN(nn.Module):
    """Dilated temporal convolutional network. Dilation doubles per block."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 depth: int = 3, dropout: float = 0.2):
        super().__init__()
        blocks, c_in = [], input_dim
        for i in range(depth):
            blocks.append(_TCNBlock(c_in, hidden_dim, kernel_size=3,
                                    dilation=2 ** i, dropout=dropout))
            c_in = hidden_dim
        self.net = nn.Sequential(*blocks)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x.transpose(1, 2)).transpose(1, 2)   # (B, W, hidden)
        return self.proj(x)


class _RecurrentEncoder(nn.Module):
    """Shared wrapper for GRU / RNN / LSTM encoders."""

    _CELL = None

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_layers: int = 2, dropout: float = 0.2, **cell_kwargs):
        super().__init__()
        self.rnn = self._CELL(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            # PyTorch applies dropout *between* layers only; passing a
            # non-zero value with num_layers=1 warns and does nothing.
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=False,
            **cell_kwargs,
        )
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)              # (B, W, hidden)
        return self.proj(out)


class TemporalEncoderGRU(_RecurrentEncoder):
    _CELL = nn.GRU


class TemporalEncoderLSTM(_RecurrentEncoder):
    _CELL = nn.LSTM


class TemporalEncoderRNN(_RecurrentEncoder):
    _CELL = nn.RNN

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("nonlinearity", "tanh")
        super().__init__(*args, **kwargs)


# =============================================================================
# Transformer-family attention
# =============================================================================

class _VanillaAttention(nn.Module):
    """Standard scaled dot-product multi-head self-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _shape(self, t: torch.Tensor) -> torch.Tensor:
        B, L, _ = t.shape
        return t.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        q = self._shape(self.q(x))
        k = self._shape(self.k(x))
        v = self._shape(self.v(x))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = self.drop(torch.softmax(scores, dim=-1))
        ctx = (attn @ v).transpose(1, 2).reshape(B, L, -1)
        return self.out(ctx)


class _ProbSparseAttention(nn.Module):
    """
    ProbSparse self-attention (Informer, Zhou et al. 2021), simplified.

    Only the `u = c * ln(L)` queries with the highest sparsity measure

        M(q, K) = max_k (q k^T / sqrt(d)) - mean_k (q k^T / sqrt(d))

    attend over the full key set; the remaining queries are assigned the mean
    of V.  Cost drops from O(L^2) to O(L log L).

    No distilling layer is used, so the sequence length is preserved and the
    temporal-attention pooling head still sees all W timesteps.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 factor: int = 5):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.factor = factor
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _shape(self, t: torch.Tensor) -> torch.Tensor:
        B, L, _ = t.shape
        return t.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        H, D = self.n_heads, self.head_dim
        q = self._shape(self.q(x))
        k = self._shape(self.k(x))
        v = self._shape(self.v(x))

        n_sample = min(L, max(1, int(self.factor * math.ceil(math.log(L + 1)))))
        u = n_sample

        # Sparsity measurement against a random subset of keys
        idx = torch.randint(L, (n_sample,), device=x.device)
        k_sample = k[:, :, idx, :]                              # (B,H,n_sample,D)
        scores_sample = (q @ k_sample.transpose(-2, -1)) / math.sqrt(D)
        m = scores_sample.max(dim=-1).values - scores_sample.mean(dim=-1)
        top_idx = m.topk(u, dim=-1).indices                     # (B,H,u)

        # Full attention for the top-u queries only
        gather_q = top_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        q_top = torch.gather(q, 2, gather_q)                    # (B,H,u,D)
        scores = (q_top @ k.transpose(-2, -1)) / math.sqrt(D)   # (B,H,u,L)
        attn = self.drop(torch.softmax(scores, dim=-1))
        ctx_top = attn @ v                                      # (B,H,u,D)

        # Remaining queries fall back to the mean of V
        ctx = v.mean(dim=2, keepdim=True).expand(B, H, L, D).clone()
        ctx = ctx.scatter(2, gather_q, ctx_top)

        ctx = ctx.transpose(1, 2).reshape(B, L, H * D)
        return self.out(ctx)


class _SeriesDecomp(nn.Module):
    """Moving-average series decomposition: x -> (seasonal, trend)."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L, C). Replicate-pad so the trend keeps the original length.
        k = min(self.kernel_size, x.size(1) if x.size(1) % 2 == 1 else x.size(1) - 1)
        k = max(k, 1)
        pad = k // 2
        xt = x.transpose(1, 2)                                  # (B, C, L)
        if pad > 0:
            xt = F.pad(xt, (pad, pad), mode="replicate")
        trend = F.avg_pool1d(xt, kernel_size=k, stride=1).transpose(1, 2)
        return x - trend, trend


class _AutoCorrelation(nn.Module):
    """
    Auto-Correlation (Autoformer, Wu et al. 2021), simplified.

    Correlation is computed in the frequency domain via FFT; the top-k delays
    are selected and the value sequence is aggregated over those time lags
    instead of over point-wise attention weights.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 factor: int = 1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.factor = factor
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _shape(self, t: torch.Tensor) -> torch.Tensor:
        B, L, _ = t.shape
        return t.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        H, D = self.n_heads, self.head_dim
        q = self._shape(self.q(x))
        k = self._shape(self.k(x))
        v = self._shape(self.v(x))

        # corr[tau] = IFFT( FFT(q) * conj(FFT(k)) )
        fq = torch.fft.rfft(q, dim=2)
        fk = torch.fft.rfft(k, dim=2)
        corr = torch.fft.irfft(fq * torch.conj(fk), n=L, dim=2)   # (B,H,L,D)
        corr = corr.mean(dim=-1)                                  # (B,H,L)

        top_k = min(L, max(1, int(self.factor * math.ceil(math.log(L + 1)))))
        weights, delays = torch.topk(corr, top_k, dim=-1)         # (B,H,top_k)
        weights = self.drop(torch.softmax(weights, dim=-1))

        # Time-delay aggregation
        v_cat = torch.cat([v, v], dim=2)                          # (B,H,2L,D)
        arange = torch.arange(L, device=x.device).view(1, 1, L, 1)
        agg = torch.zeros_like(v)
        for i in range(top_k):
            tau = delays[..., i].view(B, H, 1, 1)
            gather_idx = (arange + tau).expand(B, H, L, D)
            rolled = torch.gather(v_cat, 2, gather_idx)           # (B,H,L,D)
            agg = agg + rolled * weights[..., i].view(B, H, 1, 1)

        agg = agg.transpose(1, 2).reshape(B, L, H * D)
        return self.out(agg)


class _TransformerEncoderLayer(nn.Module):
    """
    One encoder block.  `attn_type` selects the attention operator:
      vanilla     -> pre-norm block with standard self-attention
      informer    -> pre-norm block with ProbSparse self-attention
      autoformer  -> Auto-Correlation with series decomposition replacing
                     LayerNorm (the trend component is discarded, matching
                     the original encoder design)
    """

    def __init__(self, d_model: int, n_heads: int, ff_dim: int,
                 dropout: float, attn_type: str, moving_avg: int = 25):
        super().__init__()
        self.attn_type = attn_type
        if attn_type == "informer":
            self.attn = _ProbSparseAttention(d_model, n_heads, dropout)
        elif attn_type == "autoformer":
            self.attn = _AutoCorrelation(d_model, n_heads, dropout)
        else:
            self.attn = _VanillaAttention(d_model, n_heads, dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.drop = nn.Dropout(dropout)

        if attn_type == "autoformer":
            self.decomp1 = _SeriesDecomp(moving_avg)
            self.decomp2 = _SeriesDecomp(moving_avg)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attn_type == "autoformer":
            x, _ = self.decomp1(x + self.drop(self.attn(x)))
            x, _ = self.decomp2(x + self.drop(self.ff(x)))
            return x
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class TemporalEncoderTransformer(nn.Module):
    """Stack of transformer encoder layers with sinusoidal position encoding."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_layers: int = 2, n_heads: int = 4, dropout: float = 0.2,
                 attn_type: str = "informer", ff_mult: int = 2,
                 max_len: int = 5000):
        super().__init__()
        self.attn_type = attn_type
        self.in_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim else nn.Identity()
        )
        self.register_buffer(
            "pos_enc", self._sinusoidal(max_len, hidden_dim), persistent=False
        )
        self.layers = nn.ModuleList([
            _TransformerEncoderLayer(hidden_dim, n_heads,
                                     hidden_dim * ff_mult, dropout, attn_type)
            for _ in range(n_layers)
        ])
        self.norm = (
            nn.Identity() if attn_type == "autoformer" else nn.LayerNorm(hidden_dim)
        )
        self.proj = nn.Linear(hidden_dim, output_dim)

    @staticmethod
    def _sinusoidal(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        return pe.unsqueeze(0)                                   # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = x + self.pos_enc[:, : x.size(1), :]
        for layer in self.layers:
            x = layer(x)
        return self.proj(self.norm(x))


# =============================================================================
# Encoder factory
# =============================================================================

def build_temporal_encoder(
    temporal_arch: str,
    hidden_size: int,
    dropout: float = 0.2,
    tcn_depth: int = 3,
    rnn_layers: int = 2,
    n_heads: int = 4,
    n_transformer_layers: int = 2,
) -> nn.Module:
    """
    Build a temporal encoder mapping (B, W, hidden_size) -> (B, W, hidden_size).

    temporal_arch : 'tcn' | 'gru' | 'rnn' | 'lstm'
                    | 'informer' | 'autoformer' | 'vanilla'
    """
    arch = temporal_arch.lower()
    common = dict(input_dim=hidden_size, hidden_dim=hidden_size,
                  output_dim=hidden_size, dropout=dropout)

    if arch == "tcn":
        return TemporalEncoderTCN(depth=tcn_depth, **common)
    if arch == "gru":
        return TemporalEncoderGRU(n_layers=rnn_layers, **common)
    if arch == "lstm":
        return TemporalEncoderLSTM(n_layers=rnn_layers, **common)
    if arch == "rnn":
        return TemporalEncoderRNN(n_layers=rnn_layers, **common)
    if arch in TRANSFORMER_ARCHS:
        return TemporalEncoderTransformer(
            n_layers=n_transformer_layers, n_heads=n_heads,
            attn_type=arch, **common,
        )
    raise ValueError(
        f"Unknown temporal_arch '{temporal_arch}'. Choose from "
        f"{RECURRENT_ARCHS + TRANSFORMER_ARCHS}."
    )


# =============================================================================
# Graph attention
# =============================================================================

class GraphAttentionLayer(nn.Module):
    """
    Multi-head graph attention over a dense adjacency matrix.

    Input : h (N, in_features), adj (N, N)
    Output: (N, out_features)

    A row of `adj` that is entirely zero would make softmax return NaN (every
    logit is -inf).  Such rows are zeroed instead, so an adjacency without
    self-loops cannot poison training.
    """

    def __init__(self, in_features: int, out_features: int,
                 n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        assert out_features % n_heads == 0, (
            f"out_features ({out_features}) must be divisible by "
            f"n_heads ({n_heads})"
        )
        self.n_heads = n_heads
        self.head_dim = out_features // n_heads
        self.dropout = nn.Dropout(dropout)

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a_src = nn.Parameter(torch.empty(n_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(n_heads, self.head_dim))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

    def attention(self, h: torch.Tensor, adj: torch.Tensor
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (alpha, Wh): alpha (N, N, n_heads), Wh (N, n_heads, head_dim)."""
        N = h.size(0)
        Wh = self.W(h).view(N, self.n_heads, self.head_dim)

        src = (Wh * self.a_src).sum(-1)                 # (N, n_heads)
        dst = (Wh * self.a_dst).sum(-1)                 # (N, n_heads)
        e = F.leaky_relu(src.unsqueeze(1) + dst.unsqueeze(0), 0.2)  # (N,N,heads)

        e = e.masked_fill((adj == 0).unsqueeze(-1), float("-inf"))
        alpha = torch.softmax(e, dim=1)
        alpha = torch.nan_to_num(alpha, nan=0.0)        # isolated nodes
        return self.dropout(alpha), Wh

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        alpha, Wh = self.attention(h, adj)
        out = torch.einsum("ijh,jhd->ihd", alpha, Wh)
        return F.elu(out.reshape(h.size(0), -1))


# =============================================================================
# Shared detector scaffold
# =============================================================================

class _GraphTemporalBase(nn.Module):
    """
    Shared graph-temporal scaffold.  Subclasses only choose the temporal
    encoder; every other component is defined here.

    Pipeline
    --------
      1. GAT over the prior sensor graph            -> node embeddings (d, H)
      2. Per-sensor gate  g = tanh(W h) in (-1, 1)  -> x * (1 + alpha * g)
      3. Input projection d -> H
      4. Temporal encoder (B, W, H) -> (B, W, H)
      5. Pooling: learned temporal attention, or last timestep
      6. Residual shortcut from the mean-pooled gated input
      7. LayerNorm MLP head -> (B, d)

    GAT gradient flow
    -----------------
    The graph branch is recomputed on every training batch so `node_emb` and
    the GAT layers receive gradients.  It is cached only in eval mode, where
    the weights are frozen and the adjacency is constant, which is where the
    saving actually matters.  Caching a *detached* tensor across training
    batches would freeze the whole graph branch at its initialisation;
    caching a live one raises "backward through the graph a second time".
    Recomputing per batch avoids both.
    """

    def __init__(
        self,
        n_features: int,
        window_size: int,
        hidden_size: int = 64,
        n_gat_layers: int = 1,
        n_gat_heads: int = 4,
        dropout: float = 0.2,
        use_temporal_attention: bool = True,
        gate_strength: float = 0.1,
        attn_hidden: int = 0,       # 0 -> hidden_size // 2
        mlp_hidden: int = 0,        # 0 -> hidden_size * 2
        use_anomaly_head: bool = True,   # accepted for API compatibility
    ):
        super().__init__()
        assert hidden_size % n_gat_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by "
            f"n_gat_heads ({n_gat_heads})"
        )

        self.n_features = n_features
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.use_temporal_attention = use_temporal_attention
        self.gate_strength = gate_strength
        # `use_anomaly_head` is kept so existing call sites keep working, but no
        # classifier head is built: every training path in train.py and
        # ablation_swat.py is reconstruction-based, and an unused head would
        # inflate the parameter counts reported by the ablation study.
        self.use_anomaly_head = use_anomaly_head

        _attn_hidden = attn_hidden if attn_hidden > 0 else max(hidden_size // 2, 1)
        _mlp_hidden = mlp_hidden if mlp_hidden > 0 else hidden_size * 2

        # 1. Graph branch
        self.node_emb = nn.Embedding(n_features, hidden_size)
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_size, hidden_size, n_gat_heads, dropout)
            for _ in range(n_gat_layers)
        ])

        # 2. Sensor gate
        self.gate_proj = nn.Linear(hidden_size, 1)

        # 3. Input projection
        self.input_proj = nn.Linear(n_features, hidden_size)

        # 4. Temporal encoder — provided by the subclass
        self.temporal_encoder = self._build_encoder()

        # 5. Temporal attention pooling
        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden_size, _attn_hidden),
            nn.Tanh(),
            nn.Linear(_attn_hidden, 1),
        ) if use_temporal_attention else None

        # 6. Residual shortcut
        self.shortcut_proj = nn.Linear(hidden_size, hidden_size)

        # 7. Prediction head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, _mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(_mlp_hidden),
            nn.Linear(_mlp_hidden, n_features),
        )

        self._gat_cache_key: Optional[tuple] = None
        self._gat_cache_val: Optional[torch.Tensor] = None
        self._init_weights()

    # ── implemented by subclasses ────────────────────────────────────────────
    def _build_encoder(self) -> nn.Module:
        raise NotImplementedError

    # ── initialisation ───────────────────────────────────────────────────────
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GRU, nn.LSTM, nn.RNN)):
                for name, p in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(p)
                    elif "bias" in name:
                        nn.init.zeros_(p)

    # ── graph branch ─────────────────────────────────────────────────────────
    def invalidate_gat_cache(self) -> None:
        """Drop the cached GAT output. Call at the start of each epoch."""
        self._gat_cache_key = None
        self._gat_cache_val = None

    def _gat_encode(self, adjacency: torch.Tensor) -> torch.Tensor:
        """Run the GAT stack on the static sensor graph -> (d, hidden_size)."""
        key = (adjacency.data_ptr(), tuple(adjacency.shape))
        if (not self.training
                and self._gat_cache_key == key
                and self._gat_cache_val is not None):
            return self._gat_cache_val

        node_ids = torch.arange(self.n_features, device=adjacency.device)
        h = self.node_emb(node_ids)
        for layer in self.gat_layers:
            h = layer(h, adjacency)

        if not self.training:
            self._gat_cache_key = key
            self._gat_cache_val = h.detach()
        return h

    def sensor_gates(self, adjacency: torch.Tensor) -> torch.Tensor:
        """Per-sensor gate values in (-1, 1). Used by Level-2 XAI."""
        return torch.tanh(self.gate_proj(self._gat_encode(adjacency)).squeeze(-1))

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        x         : (B, W, d)
        adjacency : (d, d)
        returns   : pred (B, d), or (pred, attn) when return_attention=True.
                    attn is (B, W) with temporal attention enabled, else None.
        """
        B, W, d = x.shape
        if d != self.n_features:
            raise ValueError(f"Expected d={self.n_features} channels, got {d}")
        if adjacency.device != x.device:
            adjacency = adjacency.to(x.device)
        if adjacency.dtype != x.dtype:
            adjacency = adjacency.to(x.dtype)

        node_h = self._gat_encode(adjacency)                     # (d, H)
        gate = torch.tanh(self.gate_proj(node_h).squeeze(-1))    # (d,)
        x_gated = x * (1.0 + self.gate_strength * gate.view(1, 1, d))

        h = self.input_proj(x_gated)                             # (B, W, H)
        shortcut = h.mean(dim=1)                                 # (B, H)

        h_enc = self.temporal_encoder(h)                         # (B, W, H)

        if self.use_temporal_attention:
            weights = torch.softmax(self.temporal_attn(h_enc), dim=1)  # (B,W,1)
            context = (weights * h_enc).sum(dim=1)               # (B, H)
            attn = weights.squeeze(-1)
        else:
            context = h_enc[:, -1, :]
            attn = None

        context = context + self.shortcut_proj(shortcut)
        pred = self.head(context)                                # (B, d)
        return (pred, attn) if return_attention else pred


# =============================================================================
# Public detectors
# =============================================================================

class GraphTemporalAttnDetector(_GraphTemporalBase):
    """
    Graph-temporal detector with a convolutional or recurrent encoder.

    temporal_arch : 'tcn' | 'gru' | 'rnn' | 'lstm'
    """

    def __init__(
        self,
        n_features: int,
        window_size: int,
        hidden_size: int = 64,
        temporal_arch: str = "tcn",
        n_gat_layers: int = 2,
        n_gat_heads: int = 4,
        dropout: float = 0.2,
        use_temporal_attention: bool = True,
        use_anomaly_head: bool = True,
        tcn_depth: int = 3,
        gru_layers: int = 2,        # also used by 'rnn' and 'lstm'
        attn_hidden: int = 0,
        mlp_hidden: int = 0,
        gate_strength: float = 0.1,
    ):
        arch = temporal_arch.lower()
        if arch not in RECURRENT_ARCHS:
            raise ValueError(
                f"GraphTemporalAttnDetector supports {RECURRENT_ARCHS}, got "
                f"'{temporal_arch}'. Use GraphTransformerAttnDetector for "
                f"{TRANSFORMER_ARCHS}."
            )
        self.temporal_arch = arch
        self._enc_kwargs = dict(
            temporal_arch=arch, hidden_size=hidden_size, dropout=dropout,
            tcn_depth=tcn_depth, rnn_layers=gru_layers,
        )
        super().__init__(
            n_features=n_features, window_size=window_size,
            hidden_size=hidden_size, n_gat_layers=n_gat_layers,
            n_gat_heads=n_gat_heads, dropout=dropout,
            use_temporal_attention=use_temporal_attention,
            gate_strength=gate_strength, attn_hidden=attn_hidden,
            mlp_hidden=mlp_hidden, use_anomaly_head=use_anomaly_head,
        )

    def _build_encoder(self) -> nn.Module:
        return build_temporal_encoder(**self._enc_kwargs)


class GraphTransformerAttnDetector(_GraphTemporalBase):
    """
    Graph-temporal detector with a transformer-family encoder.

    temporal_arch : 'informer'   — ProbSparse self-attention
                    'autoformer' — Auto-Correlation + series decomposition
                    'vanilla'    — standard self-attention (reference baseline)

    Same constructor contract and same forward signature as
    GraphTemporalAttnDetector, so both can be swept from one loop.

    Pooling
    -------
    `use_temporal_attention` defaults to False here and True for the recurrent
    detector, and passing True is coerced back to False.

    These encoders already mix information across timesteps with their own
    attention — ProbSparse self-attention and Auto-Correlation both weight the
    whole sequence — so the last position already carries a sequence-level
    summary.  Stacking the pooling head on top applies a second softmax over
    the same axis, which flattens the representation the encoder just built
    rather than adding capacity.  TCN and RNN encoders are the opposite case:
    their final hidden state is dominated by recent timesteps, so the pooling
    head is what lets earlier timesteps contribute at all.
    """

    def __init__(
        self,
        n_features: int,
        window_size: int,
        hidden_size: int = 64,
        temporal_arch: str = "informer",
        n_gat_layers: int = 2,
        n_gat_heads: int = 4,
        dropout: float = 0.2,
        use_temporal_attention: bool = False,
        use_anomaly_head: bool = True,
        n_transformer_layers: int = 2,
        n_attn_heads: int = 0,      # 0 -> reuse n_gat_heads
        attn_hidden: int = 0,
        mlp_hidden: int = 0,
        gate_strength: float = 0.1,
    ):
        arch = temporal_arch.lower()
        if arch not in TRANSFORMER_ARCHS:
            raise ValueError(
                f"GraphTransformerAttnDetector supports {TRANSFORMER_ARCHS}, "
                f"got '{temporal_arch}'. Use GraphTemporalAttnDetector for "
                f"{RECURRENT_ARCHS}."
            )
        if use_temporal_attention:
            warnings.warn(
                f"use_temporal_attention=True ignored for '{arch}': the "
                f"encoder already attends across timesteps, so a pooling "
                f"softmax on top of it would compress the sequence "
                f"representation rather than extend it. Using last-step "
                f"pooling.",
                UserWarning,
            )
            use_temporal_attention = False

        self.temporal_arch = arch
        self._enc_kwargs = dict(
            temporal_arch=arch, hidden_size=hidden_size, dropout=dropout,
            n_heads=n_attn_heads if n_attn_heads > 0 else n_gat_heads,
            n_transformer_layers=n_transformer_layers,
        )
        super().__init__(
            n_features=n_features, window_size=window_size,
            hidden_size=hidden_size, n_gat_layers=n_gat_layers,
            n_gat_heads=n_gat_heads, dropout=dropout,
            use_temporal_attention=use_temporal_attention,
            gate_strength=gate_strength, attn_hidden=attn_hidden,
            mlp_hidden=mlp_hidden, use_anomaly_head=use_anomaly_head,
        )

    def _build_encoder(self) -> nn.Module:
        return build_temporal_encoder(**self._enc_kwargs)


def supports_temporal_attention(temporal_arch: str) -> bool:
    """
    Whether the pooling head is meaningful for this encoder.

    False for the transformer family, which pools across timesteps internally.
    Sweeps use this to avoid enumerating a pooling dimension that the model
    would only coerce back to False, producing duplicate configurations.
    """
    return temporal_arch.lower() in RECURRENT_ARCHS


def build_detector(temporal_arch: str, **kwargs) -> _GraphTemporalBase:
    """Dispatch to whichever detector class owns `temporal_arch`."""
    arch = temporal_arch.lower()
    if arch in RECURRENT_ARCHS:
        return GraphTemporalAttnDetector(temporal_arch=arch, **kwargs)
    if arch in TRANSFORMER_ARCHS:
        return GraphTransformerAttnDetector(temporal_arch=arch, **kwargs)
    raise ValueError(
        f"Unknown temporal_arch '{temporal_arch}'. Choose from "
        f"{RECURRENT_ARCHS + TRANSFORMER_ARCHS}."
    )


# =============================================================================
# Model statistics
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """Model size in MB, assuming float32 parameters and buffers."""
    n = (sum(p.numel() for p in model.parameters())
         + sum(b.numel() for b in model.buffers()))
    return (n * 4) / (1024 ** 2)


def measure_latency_cpu(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    adjacency: Optional[torch.Tensor] = None,
    n_runs: int = 10,
    warmup: int = 3,
    restore_device: bool = True,
) -> float:
    """
    Mean single-batch CPU inference latency in ms.

    The GAT cache is cleared before and after: a cached CUDA tensor from an
    earlier eval pass would otherwise be reused against CPU inputs.

    restore_device : return the model to the device it started on.  Without
        this the caller silently ends up with a CPU model, which is why the
        original ablation had to run the latency test last.
    """
    was_training = model.training
    origin = next(model.parameters()).device

    model.eval()
    if hasattr(model, "invalidate_gat_cache"):
        model.invalidate_gat_cache()
    model.to("cpu")

    d = input_shape[-1]
    x = torch.randn(input_shape)
    adj = (torch.eye(d) if adjacency is None
           else adjacency.detach().to("cpu").float())

    with torch.no_grad():
        for _ in range(warmup):
            model(x, adj)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(x, adj)
            times.append((time.perf_counter() - t0) * 1000.0)

    if restore_device:
        model.to(origin)
    if hasattr(model, "invalidate_gat_cache"):
        model.invalidate_gat_cache()
    model.train(was_training)
    return float(np.mean(times))