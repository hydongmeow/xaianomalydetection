"""
xai_functions.py
================
Four-level explainability for GraphTemporalAttnDetector and
GraphTransformerAttnDetector.

Level 1  Feature relevance      mRMR relevance + prior coupling graph
Level 2  Spatial graph (GAT)    per-layer attention, node norms, sensor gates
Level 3  Temporal attention     timestep weights + signal overlay
Level 4  Loss components        per-sensor MSE, per-rule physics residuals,
                                graph regularisation, score anatomy

Alignment with models.py
------------------------
* Attention weights come from `model(x, adj, return_attention=True)` instead
  of a forward hook, so they are correct for every temporal encoder and for
  both detector classes.
* GAT internals come from `GraphAttentionLayer.attention()`, so the plotted
  weights are produced by the same code the forward pass uses.
* `use_temporal_attention=False` (last-step pooling) is handled explicitly
  rather than crashing on a `None` module.

Requires: matplotlib, seaborn, numpy, pandas, torch.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")                       # headless / server safe
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F

_PALETTE = {
    "bg":      "#ffffff",
    "panel":   "#f9fbff",
    "accent1": "#4fa3e0",   # blue  — relevance / attention
    "accent2": "#e07c4f",   # amber — anomaly / violation
    "accent3": "#4fe07a",   # green — normal / low residual
    "accent4": "#9b59b6",   # purple — temporal
    "accent5": "#27ae60",   # green — graph
    "text":    "#1a1d27",
    "subtext": "#5d6679",
    "grid":    "#d9dde5",
}

def _style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(_PALETTE["panel"])
    ax.tick_params(colors=_PALETTE["subtext"], labelsize=8)
    ax.xaxis.label.set_color(_PALETTE["subtext"])
    ax.yaxis.label.set_color(_PALETTE["subtext"])
    for spine in ax.spines.values():
        spine.set_edgecolor(_PALETTE["grid"])
    if title:
        ax.set_title(title, color=_PALETTE["text"], fontsize=10, pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)


def _save_or_show(fig, path: Optional[str]):
    fig.patch.set_facecolor(_PALETTE["bg"])
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=_PALETTE["bg"])
        print(f"  Figure saved -> {path}")
    return fig


def _name(feature_cols: List[str], idx: int) -> str:
    return feature_cols[idx] if idx < len(feature_cols) else f"F{idx}"


# =============================================================================
# LEVEL 1 — Feature relevance
# =============================================================================

def explain_feature_relevance(
    relevance: pd.Series,
    prior_adjacency: pd.DataFrame,
    feature_cols: List[str],
    top_k: int = 25,
    prediction_error_per_feature: Optional[np.ndarray] = None,
    save_path: Optional[str] = "xai_level1_relevance.png",
) -> plt.Figure:
    """
    Feature relevance and sensor coupling from pre_pipeline.

    relevance : pre_pipeline()["relevance"] — mRMR predictive-coupling scores.
    prior_adjacency : pre_pipeline()["prior_adjacency"] — (k, k) DataFrame.
    prediction_error_per_feature : optional per-feature reconstruction MSE for
        the sample under study; bar color then encodes how much each
        high-relevance sensor contributed to the anomaly score.
    """
    rel = relevance.reindex(feature_cols).fillna(0.0)
    top_feats = rel.nlargest(min(top_k, len(rel)))

    print("\n" + "=" * 70)
    print("LEVEL 1 — Feature relevance (mRMR + coupling graph)")
    print("=" * 70)
    print(f"  Selected features : {len(feature_cols)}")
    if prior_adjacency is not None:
        n_edges = int(prior_adjacency.values.sum()) - len(feature_cols)
        print(f"  Coupling edges (excl. self-loops): {n_edges}")
    print(f"\n  Top {min(10, len(top_feats))} features by relevance:")
    for i, (feat, val) in enumerate(top_feats.head(10).items(), 1):
        err_str = ""
        if prediction_error_per_feature is not None and feat in feature_cols:
            err_str = (f"  recon_err="
                       f"{prediction_error_per_feature[feature_cols.index(feat)]:.4f}")
        print(f"    {i:>2}. {feat:<22} relevance={val:.4f}{err_str}")
    print("=" * 70)

    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.15)

    # Left: relevance bars
    ax_bar = fig.add_subplot(gs[0])
    if prediction_error_per_feature is not None:
        emax = prediction_error_per_feature.max() + 1e-8
        colors = []
        for feat in top_feats.index:
            idx = feature_cols.index(feat) if feat in feature_cols else -1
            err = prediction_error_per_feature[idx] if idx >= 0 else 0.0
            t = float(min(err / emax, 1.0))          # green -> amber
            r = int(0x4f + t * (0xe0 - 0x4f))
            g = int(0xe0 + t * (0x7c - 0xe0))
            b = int(0x7a + t * (0x4f - 0x7a))
            colors.append(f"#{r:02x}{g:02x}{b:02x}")
        subtitle = "color = reconstruction error"
    else:
        colors = [_PALETTE["accent1"]] * len(top_feats)
        subtitle = "color = uniform"

    y_pos = np.arange(len(top_feats))
    ax_bar.barh(y_pos, top_feats.values, color=colors, height=0.7,
                edgecolor="none")
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(top_feats.index, fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.axvline(top_feats.values.mean(), color=_PALETTE["subtext"],
                   linestyle="--", linewidth=0.8, label="mean")
    ax_bar.legend(fontsize=7, facecolor=_PALETTE["panel"],
                  labelcolor=_PALETTE["subtext"])
    _style_ax(ax_bar,
              title=f"Top-{len(top_feats)} feature relevance (mRMR)\n{subtitle}",
              xlabel="mRMR relevance")

    # Right: coupling sub-matrix
    ax_heat = fig.add_subplot(gs[1])
    available = ([f for f in top_feats.index if f in prior_adjacency.columns]
                 if prior_adjacency is not None else [])
    if len(available) >= 2:
        sub_adj = prior_adjacency.loc[available, available].values.astype(float)
        np.fill_diagonal(sub_adj, 0)                 # hide self-loops
        sns.heatmap(sub_adj, ax=ax_heat, cmap="Blues",
                    xticklabels=available, yticklabels=available,
                    linewidths=0.3, linecolor=_PALETTE["bg"],
                    cbar_kws={"shrink": 0.7}, annot=False)
        ax_heat.tick_params(labelsize=6, rotation=45)
        _style_ax(ax_heat,
                  title="Prior coupling graph (top features)\n"
                        "edge = significant directed coupling")
    else:
        ax_heat.text(0.5, 0.5, "Adjacency not available\nfor selected features",
                     ha="center", va="center", color=_PALETTE["subtext"],
                     transform=ax_heat.transAxes)
        _style_ax(ax_heat, title="Prior coupling graph")

    return _save_or_show(fig, save_path)


# =============================================================================
# LEVEL 2 — Spatial graph (GAT)
# =============================================================================


def explain_gat_spatial(
        model,
        adjacency: torch.Tensor,
        feature_cols: List[str],
        top_k: int = 20,
        save_path: Optional[str] = "xai_level2_gat.png",
) -> plt.Figure:
    """
    What the GAT stack learns about the sensor graph.

    Top row  : per-layer head-averaged attention matrix, one square heatmap
               per GAT layer.
    Bottom   : per-sensor gate values across all sensors.

    Attention is read from `GraphAttentionLayer.attention()` — the same
    function the forward pass calls — so the plots cannot drift from the model.

    Sensors are ranked by |gate| rather than by embedding norm. The norm is a
    scale that the downstream LayerNorm largely removes, so it carried little
    information; the gate is the quantity that actually reaches the input as
    `x * (1 + gate_strength * gate)`, which makes it the meaningful ordering
    and ties the heatmap selection to the strip below it.
    """
    model.eval()
    adjacency = adjacency.to(next(model.parameters()).device).float()

    with torch.no_grad():
        node_ids = torch.arange(model.n_features, device=adjacency.device)
        h = model.node_emb(node_ids)

        last_attn = None
        for layer in model.gat_layers:
            alpha, _ = layer.attention(h, adjacency)  # (N, N, heads)
            last_attn = alpha.mean(-1).cpu().numpy()  # overwrite -> keeps the LAST
            h = layer(h, adjacency)

        gate = torch.tanh(model.gate_proj(h).squeeze(-1)).cpu().numpy()

    if last_attn is None:
        raise RuntimeError("model.gat_layers is empty — nothing to plot")

    degree = (adjacency != 0).sum(dim=1).cpu().numpy() - 1  # excl. self-loop
    rank = np.argsort(np.abs(gate))[::-1]

    n_gat = len(model.gat_layers)
    print("\n" + "=" * 70)
    print("LEVEL 2 — Spatial graph (GAT) internals")
    print("=" * 70)
    print(f"  GAT layers: {n_gat}   "
          f"graph edges: {int((adjacency != 0).sum().item())}")
    print("\n  Sensors with the strongest gate (largest effect on the input):")
    for pos, idx in enumerate(rank[:10], 1):
        direction = "amplify" if gate[idx] >= 0 else "suppress"
        print(f"    {pos:>2}. {_name(feature_cols, idx):<22} "
              f"gate={gate[idx]:+.4f} ({direction})  "
              f"neighbours={int(degree[idx])}")
    print("=" * 70)

    k = min(top_k, model.n_features)
    sub_idx = rank[:k]
    sub_labs = [_name(feature_cols, i) for i in sub_idx]

    fig = plt.figure(figsize=(7.0, 8.0), layout="constrained")
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3.6, 1])

    # --- panel 1: attention of the LAST GAT layer ---
    ax = fig.add_subplot(gs[0, 0])
    sub_mat = last_attn[np.ix_(sub_idx, sub_idx)]
    sns.heatmap(sub_mat, ax=ax, cmap="YlOrRd",
                xticklabels=sub_labs, yticklabels=sub_labs,
                linewidths=0.2, linecolor=_PALETTE["bg"],
                square=True, cbar_kws={"shrink": 0.6},
                vmin=0, vmax=max(sub_mat.max(), 1e-8))
    ax.tick_params(labelsize=6)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    _style_ax(ax, title=f"GAT layer {n_gat} (last) — attention weights\n"
                        f"(top-{k} sensors by |gate|)")

    # --- panel 2: gate values ---
    ax_gate = fig.add_subplot(gs[1, 0])
    x = np.arange(model.n_features)
    ax_gate.bar(x, gate, width=0.8, edgecolor="none",
                color=[_PALETTE["accent1"] if g >= 0 else _PALETTE["accent2"]
                       for g in gate])
    ax_gate.axhline(0, color=_PALETTE["subtext"], linewidth=0.8, linestyle="--")
    step = max(1, len(x) // 25)
    ax_gate.set_xticks(x[::step])
    ax_gate.set_xticklabels([_name(feature_cols, i) for i in x[::step]],
                            rotation=90, ha="right", fontsize=6)
    ax_gate.margins(x=0.01)
    _style_ax(ax_gate,
              title="Sensor gate values  (blue = amplify, amber = suppress)",
              ylabel="gate in (-1, 1)")

    return _save_or_show(fig, save_path)

# =============================================================================
# LEVEL 3 — Temporal attention
# =============================================================================

def explain_temporal_attention(
    model,
    x_window: torch.Tensor,
    adjacency: torch.Tensor,
    feature_cols: List[str],
    sample_label: int,
    pred: Optional[torch.Tensor] = None,
    recon_error_per_feature: Optional[np.ndarray] = None,
    save_path: Optional[str] = "xai_level3_temporal_attn.png",
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Which timesteps the model attended to, overlaid on the raw signals of the
    sensors with the highest reconstruction error.

    When the model was built with `use_temporal_attention=False`, pooling is
    "take the last step"; the effective weight vector is then a one-hot on the
    final timestep and the plot is annotated accordingly.

    Returns (figure, attention_weights (W,)).
    """
    model.eval()
    B, W, d = x_window.shape
    assert B == 1, "Pass a single sample (batch=1)"
    adjacency = adjacency.to(x_window.device).float()

    with torch.no_grad():
        out = model(x_window, adjacency, return_attention=True)
        pred_new, attn = out
    if pred is None:
        pred = pred_new

    pooling = "learned attention"
    if attn is None:
        # Last-step pooling: the effective weighting is a one-hot.
        attn_w = np.zeros(W, dtype=np.float64)
        attn_w[-1] = 1.0
        pooling = "last-step pooling (no learned attention)"
    else:
        attn_w = attn[0].cpu().numpy()

    peak_t = int(np.argmax(attn_w))
    entropy = float(-np.sum(attn_w * np.log(attn_w + 1e-12)))
    max_entropy = float(np.log(W))

    print("\n" + "=" * 70)
    print("LEVEL 3 — Temporal attention")
    print("=" * 70)
    print(f"  Sample label   : {'ANOMALY' if sample_label else 'NORMAL'}")
    print(f"  Pooling        : {pooling}")
    print(f"  Window length  : {W} timesteps")
    print(f"  Peak attention : timestep {peak_t}/{W - 1} "
          f"(weight={attn_w[peak_t]:.4f})")
    print(f"  Top-5 timesteps: {list(np.argsort(attn_w)[::-1][:5])}")
    print(f"  Entropy        : {entropy:.3f} / {max_entropy:.3f}  "
          f"({'diffuse — many timesteps matter' if entropy > 0.8 * max_entropy else 'focused — few timesteps dominate'})")
    print("=" * 70)

    x_np = x_window[0].cpu().numpy()
    top_sensors = (np.argsort(recon_error_per_feature)[::-1][:4]
                   if recon_error_per_feature is not None
                   else np.arange(min(4, d)))

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, hspace=0.3, wspace=0.15)
    t_axis = np.arange(W)

    ax_attn = fig.add_subplot(gs[0, :])
    ax_attn.bar(t_axis, attn_w, width=1.0, edgecolor="none", color=[
        _PALETTE["accent2"] if w >= attn_w[peak_t] * 0.8
        else (_PALETTE["accent1"] if w >= attn_w.mean() else _PALETTE["grid"])
        for w in attn_w
    ])
    ax_attn.axvline(peak_t, color=_PALETTE["accent2"], linewidth=1.2,
                    linestyle="--", label=f"peak t={peak_t}")
    ax_attn.axhline(1.0 / W, color=_PALETTE["subtext"], linewidth=0.8,
                    linestyle=":", label="uniform (1/W)")
    ax_attn.legend(fontsize=8, facecolor=_PALETTE["panel"],
                   labelcolor=_PALETTE["subtext"])
    _style_ax(ax_attn,
              title=f"Temporal weights — {pooling}  "
                    f"[label={'ANOMALY' if sample_label else 'NORMAL'}]",
              xlabel="Timestep", ylabel="Weight")

    for plot_i, sensor_idx in enumerate(top_sensors):
        ax = fig.add_subplot(gs[1 + plot_i // 2, plot_i % 2])
        fname = _name(feature_cols, int(sensor_idx))

        denom = attn_w.max() + 1e-8
        for t in range(W - 1):
            ax.axvspan(t, t + 1, alpha=float(attn_w[t] / denom) * 0.4,
                       color=_PALETTE["accent2"], linewidth=0)

        ax.plot(t_axis, x_np[:, sensor_idx], color=_PALETTE["accent1"],
                linewidth=1.2)
        ax.axvline(peak_t, color=_PALETTE["accent2"], linewidth=0.8,
                   linestyle="--")

        title = fname
        if recon_error_per_feature is not None:
            title += f"  (recon_err={recon_error_per_feature[sensor_idx]:.4f})"
        _style_ax(ax, title=title, xlabel="Timestep", ylabel="Value (scaled)")

    return _save_or_show(fig, save_path), attn_w


# =============================================================================
# LEVEL 4 — Loss component breakdown
# =============================================================================

def explain_loss_components(
    model,
    loss_fn,
    x_window: torch.Tensor,
    y_true: torch.Tensor,
    adjacency: torch.Tensor,
    feature_cols: List[str],
    save_path: Optional[str] = "xai_level4_loss.png",
) -> Tuple[plt.Figure, Dict]:
    """
    Where the anomaly score comes from, for one sample:

      (a) per-sensor reconstruction MSE
      (b) per-rule physics residual
      (c) stacked contribution of every active loss component

    `adjacency` is forwarded to the loss so the graph regularisation term is
    evaluated, not silently skipped — with adjacency=None the graph component
    would read as a constant zero and the score anatomy would be wrong.
    """
    model.eval()
    adjacency = adjacency.to(x_window.device).float()

    with torch.no_grad():
        pred = model(x_window, adjacency)                    # (1, d)
        x_prev = x_window[:, -1, :]                          # (1, d)
        per_feat_mse = F.mse_loss(pred, y_true, reduction="none"
                                  ).squeeze(0).cpu().numpy()

        physics_breakdown: Dict[str, float] = {}
        if loss_fn is not None and getattr(loss_fn, "residual_calc", None) is not None:
            state = pred if getattr(loss_fn, "physics_on_prediction", False) else None
            residuals = loss_fn.residual_calc.compute_all_residuals(
                x_t=x_prev, x_next=pred if state is not None else y_true,
                state=state,
            )
            rules = getattr(loss_fn.residual_calc.rule_set, "rules", [])
            for key, val in residuals.items():
                i = int(key.split("_")[-1])
                label = (f"{rules[i].rule_type.value}:"
                         f"{'/'.join(rules[i].features)}"
                         if i < len(rules) else key)
                physics_breakdown[label] = float(val.mean().item())

        if loss_fn is not None:
            _, components = loss_fn(pred, y_true, x_prev=x_prev,
                                    adjacency=adjacency, return_components=True)
        else:
            components = {"mse": float(F.mse_loss(pred, y_true).item())}

    breakdown = {
        "per_feature_mse": per_feat_mse,
        "physics_residuals": physics_breakdown,
        "loss_components": components,
        "prediction": pred.cpu().numpy()[0],
        "ground_truth": y_true.cpu().numpy()[0],
    }

    print("\n" + "=" * 70)
    print("LEVEL 4 — Loss component breakdown")
    print("=" * 70)
    if loss_fn is not None:
        print(f"  Active terms: mse"
              f"{' + physics' if getattr(loss_fn, 'use_physics', False) else ''}"
              f"{' + temporal' if getattr(loss_fn, 'use_temporal', False) else ''}"
              f"{' + graph' if getattr(loss_fn, 'use_graph', False) else ''}"
              f"   (ablation_mode='{getattr(loss_fn, 'ablation_mode', '?')}')")
    print("  Loss components:")
    for k, v in components.items():
        if isinstance(v, float) and not k.startswith("temp_"):
            print(f"    {k:<25} = {v:.6f}")

    if physics_breakdown:
        print(f"\n  Physics residuals ({len(physics_breakdown)} rules):")
        for key, val in sorted(physics_breakdown.items(),
                               key=lambda kv: kv[1], reverse=True)[:10]:
            print(f"    {key:<32} = {val:.6f}")

    print("\n  Top-5 sensors by reconstruction error:")
    for rank, idx in enumerate(np.argsort(per_feat_mse)[::-1][:5], 1):
        print(f"    {rank}. {_name(feature_cols, int(idx)):<22} "
              f"MSE={per_feat_mse[idx]:.6f}")
    print("=" * 70)

    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 2, hspace=0.25, wspace=0.15)

    # Panel A: per-sensor reconstruction error
    ax_mse = fig.add_subplot(gs[0, :])
    x_pos = np.arange(len(per_feat_mse))
    err_norm = per_feat_mse / (per_feat_mse.max() + 1e-8)
    ax_mse.bar(x_pos, per_feat_mse, width=0.8, edgecolor="none", color=[
        _PALETTE["accent2"] if e >= 0.7
        else (_PALETTE["accent1"] if e >= 0.3 else _PALETTE["accent3"])
        for e in err_norm
    ])
    step = max(1, len(x_pos) // 20)
    ax_mse.set_xticks(x_pos[::step])
    ax_mse.set_xticklabels([_name(feature_cols, int(i)) for i in x_pos[::step]],
                           rotation=45, ha="right", fontsize=7)
    ax_mse.axhline(per_feat_mse.mean(), color=_PALETTE["subtext"],
                   linestyle="--", linewidth=0.9, label="mean MSE")
    ax_mse.legend(fontsize=8, facecolor=_PALETTE["panel"],
                  labelcolor=_PALETTE["subtext"])
    _style_ax(ax_mse,
              title="Per-sensor reconstruction error (MSE)\n"
                    "amber = high contribution, blue = moderate, green = normal",
              ylabel="MSE")
    n_sensors = len(x_pos)
    label_size = 8 if n_sensors <= 30 else (7 if n_sensors <= 60 else 6)
    ax_mse.set_xticks(x_pos)
    ax_mse.set_xticklabels([_name(feature_cols, int(i)) for i in x_pos],
                           rotation=90, ha="center", fontsize=6)
    ax_mse.tick_params(axis="x", labelsize=label_size,
                       colors=_PALETTE["subtext"])
    ax_mse.set_xlim(-0.6, n_sensors - 0.4)

    # Panel B: physics residuals
    ax_phy = fig.add_subplot(gs[1, 0])
    if physics_breakdown:
        keys = sorted(physics_breakdown, key=physics_breakdown.get, reverse=True)
        vals = [physics_breakdown[k] for k in keys]
        y_ph = np.arange(len(keys))
        ax_phy.barh(y_ph, vals, height=0.7, edgecolor="none", color=[
            _PALETTE["accent2"] if v > np.mean(vals) else _PALETTE["accent1"]
            for v in vals
        ])
        ax_phy.set_yticks(y_ph)
        ax_phy.set_yticklabels(keys, fontsize=7)
        ax_phy.invert_yaxis()
        ax_phy.axvline(float(np.mean(vals)), color=_PALETTE["subtext"],
                       linestyle="--", linewidth=0.8, label="mean")
        ax_phy.legend(fontsize=7, facecolor=_PALETTE["panel"],
                      labelcolor=_PALETTE["subtext"])
        _style_ax(ax_phy,
                  title="Per-rule physics residuals\n(amber = above average)",
                  xlabel="Residual magnitude")
    else:
        ax_phy.text(0.5, 0.5, "No physics rules\n(MSE-only mode)",
                    ha="center", va="center", color=_PALETTE["subtext"],
                    fontsize=10, transform=ax_phy.transAxes)
        _style_ax(ax_phy, title="Physics residuals")

    # Panel C: score anatomy
    ax_stack = fig.add_subplot(gs[1, 1])
    comp_keys = [k for k in ("mse", "physics", "temporal", "graph")
                 if isinstance(components.get(k), float)]
    comp_vals = [abs(components[k]) for k in comp_keys]
    total = sum(comp_vals) + 1e-8
    colors = {"mse": _PALETTE["accent1"], "physics": _PALETTE["accent2"],
              "temporal": _PALETTE["accent4"], "graph": _PALETTE["accent5"]}

    bottom = 0.0
    for key, val in zip(comp_keys, comp_vals):
        frac = val / total
        ax_stack.bar(0, frac, bottom=bottom, width=0.5,
                     color=colors.get(key, _PALETTE["subtext"]),
                     edgecolor=_PALETTE["bg"], linewidth=0.8,
                     label=f"{key}  ({val:.4f})")
        ax_stack.text(0.28, bottom + frac / 2, f"{key}\n{frac * 100:.1f}%",
                      ha="left", va="center", color=_PALETTE["text"],
                      fontsize=8)
        bottom += frac

    ax_stack.set_xlim(-0.4, 1.2)
    ax_stack.set_ylim(0, 1.05)
    ax_stack.set_xticks([])
    ax_stack.legend(fontsize=7, loc="upper right",
                    facecolor=_PALETTE["panel"],
                    labelcolor=_PALETTE["subtext"])
    _style_ax(ax_stack,
              title="Anomaly score anatomy\n(fraction of weighted total loss)",
              ylabel="Fraction")

    return _save_or_show(fig, save_path), breakdown


# =============================================================================
# All four levels in one call
# =============================================================================

def explain_sample(
    model,
    loss_fn,
    x_window: torch.Tensor,
    y_true: torch.Tensor,
    adjacency: torch.Tensor,
    feature_cols: List[str],
    sample_label: int,
    relevance: pd.Series,
    prior_adjacency_df: pd.DataFrame,
    save_dir: str = ".",
    prefix: str = "xai",
) -> Dict:
    """
    Run all four XAI levels on one sample and save every figure.

    model     : a trained detector in eval mode
    loss_fn   : CompositeLoss, or None for MSE-only runs
    x_window  : (1, W, d)
    y_true    : (1, d)
    adjacency : (d, d) — the same matrix used during training
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    adjacency = adjacency.to(x_window.device).float()

    print("\n" + "#" * 70)
    print("  XAI EXPLANATION REPORT")
    print(f"  Sample label: {'ANOMALY' if sample_label else 'NORMAL'}")
    print(f"  Model: {type(model).__name__} "
          f"(temporal_arch={getattr(model, 'temporal_arch', '?')}, "
          f"temporal_attention={model.use_temporal_attention})")
    print("#" * 70)

    with torch.no_grad():
        pred = model(x_window, adjacency)
    per_feat_mse = F.mse_loss(pred, y_true, reduction="none"
                              ).squeeze(0).cpu().numpy()

    fig1 = explain_feature_relevance(
        relevance=relevance, prior_adjacency=prior_adjacency_df,
        feature_cols=feature_cols, top_k=min(20, len(feature_cols)),
        prediction_error_per_feature=per_feat_mse,
        save_path=f"{save_dir}/{prefix}_level1_relevance.png",
    )
    fig2 = explain_gat_spatial(
        model=model, adjacency=adjacency, feature_cols=feature_cols,
        top_k=min(20, len(feature_cols)),
        save_path=f"{save_dir}/{prefix}_level2_gat.png",
    )
    fig3, attn_weights = explain_temporal_attention(
        model=model, x_window=x_window, adjacency=adjacency,
        feature_cols=feature_cols, sample_label=sample_label, pred=pred,
        recon_error_per_feature=per_feat_mse,
        save_path=f"{save_dir}/{prefix}_level3_temporal.png",
    )
    fig4, breakdown = explain_loss_components(
        model=model, loss_fn=loss_fn, x_window=x_window, y_true=y_true,
        adjacency=adjacency, feature_cols=feature_cols,
        save_path=f"{save_dir}/{prefix}_level4_loss.png",
    )

    print("\n" + "#" * 70)
    print(f"  XAI COMPLETE — figures saved to: {save_dir}")
    print("#" * 70 + "\n")
    plt.close("all")

    return {
        "per_feature_mse": per_feat_mse,
        "attn_weights": attn_weights,
        "loss_breakdown": breakdown,
        "prediction": pred.cpu().numpy()[0],
        "ground_truth": y_true.cpu().numpy()[0],
        "figures": [fig1, fig2, fig3, fig4],
    }
