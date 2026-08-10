"""
train.py
========
Trainer, architecture sweep (Part 1) and loss-component ablation (Part 2).

Evaluation protocols
--------------------
Every evaluation entry point exists in two forms:

    compute_anomaly_metrics_absolute      point-wise
    compute_anomaly_metrics_absolute_adj  point-adjusted

    run_hyperparameter_sweep              point-wise
    run_hyperparameter_sweep_adj          point-adjusted

    run_ablation_study                    point-wise
    run_ablation_study_adj                point-adjusted

The `_adj` variants are thin wrappers that set `point_adjust=True` on the
same implementation, so the two protocols cannot drift apart: any change to
the core is picked up by both.

Point adjustment (Xu et al. 2018) marks an entire ground-truth anomaly
segment as detected when at least one point inside it is flagged.  It is the
protocol most SWaT/WADI papers report, and it inflates recall substantially
compared with the point-wise numbers, so the two must never be mixed in one
table.
"""

import itertools
import time
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader

from models import (
    build_detector, count_parameters, get_model_size_mb, measure_latency_cpu,
    supports_temporal_attention,
)
from piloss import CompositeLoss, PhysicsResidualCalculator

warnings.filterwarnings("ignore")


# =============================================================================
# Optimisation helpers
# =============================================================================

def make_optimizer(
        model: nn.Module,
        loss_fn: Optional[nn.Module],
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
) -> torch.optim.Optimizer:
    """
    Adam over the model parameters AND any learnable uncertainty parameters
    (KendallGal log_sigma, GradNorm weights).  Building the optimizer outside
    the trainer usually misses `loss_fn.uncertainty.parameters()`, which
    silently freezes the multi-task weights.
    """
    params = list(model.parameters())
    if loss_fn is not None and getattr(loss_fn, "uncertainty", None) is not None:
        params += list(loss_fn.uncertainty.parameters())
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


def _as_adjacency_tensor(
        prior_adjacency, n_features: int
) -> torch.Tensor:
    """Accept a DataFrame, ndarray, tensor or None; return a float (d, d)."""
    if prior_adjacency is None:
        warnings.warn(
            "prior_adjacency is None — falling back to the identity matrix. "
            "The GAT then sees self-loops only and the graph regularisation "
            "term is identically zero.",
            RuntimeWarning,
        )
        return torch.eye(n_features, dtype=torch.float32)
    if isinstance(prior_adjacency, pd.DataFrame):
        prior_adjacency = prior_adjacency.values
    if isinstance(prior_adjacency, np.ndarray):
        prior_adjacency = torch.from_numpy(prior_adjacency)
    return prior_adjacency.to(torch.float32)


# =============================================================================
# Evaluation
# =============================================================================

def apply_point_adjust(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Point-adjust protocol (Xu et al. 2018).

    If any point inside a contiguous ground-truth anomaly segment is flagged,
    the whole segment counts as detected.  Points outside anomaly segments are
    left untouched, so false positives are unaffected.
    """
    y_pred = np.asarray(y_pred).astype(int).copy()
    y_true = np.asarray(y_true).astype(int)

    edges = np.diff(np.concatenate(([0], y_true, [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    for s, e in zip(starts, ends):
        if y_pred[s:e].any():
            y_pred[s:e] = 1
    return y_pred


def compute_anomaly_metrics(
        test_result: Dict,
        thresholds: Sequence[float] = (0.1, 0.3, 0.5, 1.0),
        score_type: str = "fused_scores",
        point_adjust: bool = False,
        verbose: bool = True,
) -> Dict:
    """
    Detection metrics for a list of ABSOLUTE score thresholds.

    A sample is flagged when its score exceeds the threshold; there is no
    mean + k*std rescaling, so thresholds are directly comparable across runs.

    point_adjust : bool
        False — point-wise metrics.
        True  — a detected anomaly segment counts as fully detected
                (`apply_point_adjust`).  ROC-AUC is left point-wise either
                way, since it is computed from continuous scores, not from
                thresholded predictions.

    Returns {threshold: {precision, recall, f1, specificity, roc_auc,
                         tp, fp, fn, tn, y_pred_binary}}.
    """
    scores = np.asarray(test_result[score_type])
    labels = np.asarray(test_result["labels"]).astype(int)
    protocol = "point-adjusted" if point_adjust else "point-wise"

    try:
        roc_auc = roc_auc_score(labels, scores)
    except Exception:
        roc_auc = float("nan")

    if verbose:
        print(f"\n{'=' * 78}")
        print(f"Anomaly detection — absolute thresholds  "
              f"(score: {score_type}, {protocol})")
        print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]  "
              f"mean={scores.mean():.4f}  std={scores.std():.4f}  "
              f"anomaly_rate={labels.mean():.4f}")
        print(f"{'=' * 78}")
        print(f"{'Threshold':>10} | {'Precision':>9} | {'Recall':>7} | "
              f"{'F1':>6} | {'TP':>6} | {'FP':>6} | {'FN':>6} | {'TN':>6}")
        print(f"{'-' * 78}")

    metrics_by_threshold = {}
    for thr in thresholds:
        y_pred = (scores > thr).astype(int)
        if point_adjust:
            y_pred = apply_point_adjust(y_pred, labels)

        tp = int(np.sum((y_pred == 1) & (labels == 1)))
        fp = int(np.sum((y_pred == 1) & (labels == 0)))
        fn = int(np.sum((y_pred == 0) & (labels == 1)))
        tn = int(np.sum((y_pred == 0) & (labels == 0)))

        metrics_by_threshold[thr] = {
            "threshold": thr,
            "precision": precision_score(labels, y_pred, zero_division=0),
            "recall": recall_score(labels, y_pred, zero_division=0),
            "f1": f1_score(labels, y_pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            "roc_auc": roc_auc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "point_adjust": point_adjust,
            "y_pred_binary": y_pred,
        }

        if verbose:
            m = metrics_by_threshold[thr]
            print(f"{thr:>10.3f} | {m['precision']:>9.4f} | {m['recall']:>7.4f} | "
                  f"{m['f1']:>6.4f} | {tp:>6d} | {fp:>6d} | {fn:>6d} | {tn:>6d}")

    if verbose:
        print(f"{'=' * 78}\n")
    return metrics_by_threshold


def compute_anomaly_metrics_absolute(test_result, thresholds=(0.1, 0.3, 0.5, 1.0),
                                     score_type="fused_scores", verbose=True) -> Dict:
    """Point-wise metrics (see compute_anomaly_metrics)."""
    return compute_anomaly_metrics(test_result, thresholds, score_type,
                                   point_adjust=False, verbose=verbose)


def compute_anomaly_metrics_absolute_adj(test_result, thresholds=(0.1, 0.3, 0.5, 1.0),
                                         score_type="fused_scores", verbose=True) -> Dict:
    """Point-adjusted metrics (see compute_anomaly_metrics)."""
    return compute_anomaly_metrics(test_result, thresholds, score_type,
                                   point_adjust=True, verbose=verbose)


# =============================================================================
# Trainer
# =============================================================================

class AnomalyDetectionTrainer:
    """
    Trainer for the graph-temporal detectors.

    Two loss modes:
      loss_fn=None            pure MSE reconstruction (architecture sweep).
      loss_fn=CompositeLoss   physics-informed composite loss (ablation).

    The prior adjacency is passed to BOTH the model (GAT / sensor gating) and
    the loss (graph regularisation) on every batch.
    """

    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            device: torch.device,
            prior_adjacency: torch.Tensor,
            feature_names: List[str],
            loss_fn: Optional[CompositeLoss] = None,
            grad_clip: float = 1.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.prior_adjacency = _as_adjacency_tensor(
            prior_adjacency, len(feature_names)).to(device)
        self.feature_names = feature_names
        self.grad_clip = grad_clip
        self.train_history: List[Dict] = []
        self.val_history: List[Dict] = []

        # Move the loss to the device BEFORE the optimizer is used: learnable
        # uncertainty parameters (log_sigma) must live where the model lives,
        # or the first backward pass raises a device mismatch.
        self.loss_fn = loss_fn.to(device) if loss_fn is not None else None

        self.optimizer = optimizer
        self.best_model_state = None
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer, mode="min", factor=0.5,
            patience=3, min_lr=1e-6,
        )

    _TRACKED = ("total", "mse", "physics", "temporal", "graph")

    def _compute_loss(
            self, y_pred: torch.Tensor, y_true: torch.Tensor,
            x_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """Dispatch to MSE or CompositeLoss. Components always contain 'mse'."""
        if self.loss_fn is None:
            loss = F.mse_loss(y_pred, y_true)
            return loss, {"mse": loss.item()}

        return self.loss_fn(
            y_pred, y_true,
            x_prev=x_prev,
            adjacency=self.prior_adjacency,
            return_components=True,
        )

    def _run_epoch(self, loader: DataLoader, train: bool) -> Dict:
        self.model.train(train)
        if train and hasattr(self.model, "invalidate_gat_cache"):
            # Refresh the eval-mode GAT cache: last epoch's weights are stale.
            self.model.invalidate_gat_cache()

        accum: Dict[str, List[float]] = {k: [] for k in self._TRACKED}

        with torch.set_grad_enabled(train):
            for x_win, y_true, _labels in loader:
                x_win = x_win.to(self.device)
                y_true = y_true.to(self.device)

                y_pred = self.model(x_win, self.prior_adjacency)
                loss, components = self._compute_loss(
                    y_pred, y_true, x_prev=x_win[:, -1, :]
                )

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.grad_clip)
                    self.optimizer.step()

                accum["total"].append(loss.item())
                for k in self._TRACKED[1:]:
                    accum[k].append(components.get(k, 0.0))

        metrics = {k: float(np.mean(v)) if v else 0.0 for k, v in accum.items()}
        metrics["loss"] = metrics.pop("total")
        (self.train_history if train else self.val_history).append(metrics)
        return metrics

    def train_epoch(self, train_loader: DataLoader) -> Dict:
        return self._run_epoch(train_loader, train=True)

    def validate(self, val_loader: DataLoader) -> Dict:
        metrics = self._run_epoch(val_loader, train=False)
        self.scheduler.step(metrics["loss"])
        return metrics

    def fit(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            n_epochs: int = 50,
            early_stopping_patience: int = 10,
            verbose: bool = True,
            restore_best: bool = True,
    ) -> Dict:
        best_val_loss = np.inf
        patience_counter = 0

        for epoch in range(n_epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch + 1:>3}/{n_epochs} | "
                      f"Train: {train_metrics['loss']:.5f} | "
                      f"Val: {val_metrics['loss']:.5f}")

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                self.best_model_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    break

        # Evaluate the model that actually generalised, not the last epoch's.
        if restore_best and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.model.to(self.device)
            if hasattr(self.model, "invalidate_gat_cache"):
                self.model.invalidate_gat_cache()

        return {
            "train_history": self.train_history,
            "val_history": self.val_history,
            "best_val_loss": best_val_loss,
        }

    def test(self, test_loader: DataLoader) -> Dict:
        """
        Score the test set.

        recon_errors   per-sample MSE of the predicted next timestep.
        physics_scores max rule residual on the OBSERVED transition; zero in
                       MSE-only mode. Observations are used deliberately: at
                       detection time a physical constraint violated by the
                       real data is the signal of interest.
        fused_scores   0.5 * recon + 0.5 * physics.
        """
        self.model.eval()
        if hasattr(self.model, "invalidate_gat_cache"):
            self.model.invalidate_gat_cache()

        preds, trues, labels_all, recon_all, physics_all = [], [], [], [], []
        has_physics = (
            self.loss_fn is not None
            and getattr(self.loss_fn, "residual_calc", None) is not None
            and len(self.loss_fn.residual_calc.rule_set.rules) > 0
        )

        with torch.no_grad():
            for x_win, y_true, labels in test_loader:
                x_win = x_win.to(self.device)
                y_true = y_true.to(self.device)

                y_pred = self.model(x_win, self.prior_adjacency)
                recon_error = F.mse_loss(
                    y_pred, y_true, reduction="none").mean(dim=1)

                if has_physics:
                    residuals = self.loss_fn.residual_calc.compute_all_residuals(
                        x_win[:, -1, :], y_true
                    )
                    physics_score = (
                        torch.stack(list(residuals.values())).max(dim=0)[0]
                        if residuals
                        else torch.zeros(y_true.shape[0], device=self.device)
                    )
                else:
                    physics_score = torch.zeros(y_true.shape[0], device=self.device)

                preds.append(y_pred.cpu().numpy())
                trues.append(y_true.cpu().numpy())
                labels_all.append(labels.numpy())
                recon_all.append(recon_error.cpu().numpy())
                physics_all.append(physics_score.cpu().numpy())

        recon_errors = np.concatenate(recon_all)
        physics_scores = np.concatenate(physics_all)
        return {
            "predictions": np.concatenate(preds),
            "ground_truth": np.concatenate(trues),
            "labels": np.concatenate(labels_all),
            "recon_errors": recon_errors,
            "physics_scores": physics_scores,
            "fused_scores": 0.5 * recon_errors + 0.5 * physics_scores,
        }


# =============================================================================
# PART 1 — architecture sweep
# =============================================================================

def run_hyperparameter_sweep(
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        prior_adjacency,                    # tensor / ndarray / DataFrame
        feature_cols: List[str],
        window_size: int,
        # ── sweep grid ────────────────────────────────────────────────────────
        hidden_sizes: Sequence[int] = (32, 64, 96, 128),
        temporal_archs: Sequence[str] = ("tcn", "gru"),
        n_gat_layers_list: Sequence[int] = (1, 2, 3),
        n_gat_heads: int = 4,
        dropout_rates: Sequence[float] = (0.2, 0.3, 0.4),
        attention_strategies: Sequence[bool] = (True,),
        # ── evaluation ────────────────────────────────────────────────────────
        thresholds: Sequence[float] = (0.1, 0.3, 0.5, 1.0),
        score_type: str = "recon_errors",   # MSE-only -> physics is all zero
        point_adjust: bool = False,
        # ── training ──────────────────────────────────────────────────────────
        n_epochs: int = 30,
        lr: float = 1e-3,
        early_stopping_patience: int = 5,
        # ── cost profiling ────────────────────────────────────────────────────
        measure_latency: bool = False,
        latency_runs: int = 10,
        # ── misc ──────────────────────────────────────────────────────────────
        device: str = "cpu",
        verbose_epochs: bool = False,
        save_results_csv: Optional[str] = "arch_sweep_results.csv",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Architecture sweep with pure MSE loss (loss_fn=None), so no physics rules
    or composite-loss kwargs are involved and architectures are compared on
    reconstruction quality alone.

    `temporal_archs` may mix recurrent and transformer names — 'tcn', 'gru',
    'rnn', 'lstm', 'informer', 'autoformer', 'vanilla' — since build_detector
    dispatches to the right class.

    attention_strategies : which pooling heads to try.
        True  = learned temporal attention, False = last-timestep pooling.
        Pass (True, False) to ablate the pooling head itself; see
        run_temporal_ablation for the preset that does this.

        Transformer archs ('informer', 'autoformer', 'vanilla') attend across
        timesteps inside the encoder, so they always use last-step pooling.
        For those the flag is normalised to False and the duplicate grid entry
        is dropped, which keeps a mixed sweep from training the same
        transformer twice under two labels.

    measure_latency : add n_params, model_size_mb and CPU latency_ms columns.
        Off by default because it costs a CPU round-trip per config.

    Returns
    -------
    results_df  : one row per (config x threshold)
    best_config : the config with the highest mean F1 across thresholds,
                  ready to pass to run_ablation_study() as `arch_config`.

    `window_size` is baked into the loaders; wrap this call in a loop over
    window sizes and concatenate if you want to sweep that too.
    """
    device_obj = torch.device(device)
    n_features = len(feature_cols)
    prior_adjacency = _as_adjacency_tensor(prior_adjacency, n_features)
    protocol = "point-adjusted" if point_adjust else "point-wise"

    param_grid = []
    seen = set()
    for combo in itertools.product(hidden_sizes, temporal_archs,
                                   n_gat_layers_list, dropout_rates,
                                   attention_strategies):
        hidden_size, temporal_arch, n_gat_layers, dropout, use_attn = combo
        # Transformer encoders pool across timesteps internally and coerce
        # use_temporal_attention to False.  Enumerating both values would train
        # the same network twice, so the pooling flag is normalised here and
        # the duplicate dropped.
        if not supports_temporal_attention(temporal_arch):
            use_attn = False
        key = (hidden_size, temporal_arch, n_gat_layers, dropout, use_attn)
        if key not in seen:
            seen.add(key)
            param_grid.append(key)
    total_runs = len(param_grid)

    print(f"\n{'#' * 78}")
    print(f"  ARCHITECTURE SWEEP (MSE only) — {total_runs} configs")
    print(f"  window_size={window_size}  n_features={n_features}  "
          f"metrics={protocol}")
    print(f"  Thresholds: {list(thresholds)}")
    print(f"{'#' * 78}\n")

    all_results = []
    for run_idx, (hidden_size, temporal_arch, n_gat_layers, dropout,
                  use_attn) in enumerate(param_grid, 1):

        config = {
            "run": run_idx, "window_size": window_size,
            "hidden_size": hidden_size, "temporal_arch": temporal_arch,
            "n_gat_layers": n_gat_layers, "n_gat_heads": n_gat_heads,
            "dropout": dropout, "use_temporal_attention": use_attn,
        }
        print(f"[{run_idx:>3}/{total_runs}] window={window_size:>3}  "
              f"hidden={hidden_size:>3}  arch={temporal_arch:<10}  "
              f"gat_layers={n_gat_layers}  dropout={dropout}  "
              f"attn={int(use_attn)}")

        # GAT requires hidden_size % n_gat_heads == 0
        if hidden_size % n_gat_heads != 0:
            print(f"       skipped: hidden_size={hidden_size} not divisible "
                  f"by n_gat_heads={n_gat_heads}\n")
            for thr in thresholds:
                all_results.append({
                    **config, "threshold": thr, "skipped": True,
                    **{k: float("nan") for k in
                       ("final_train_loss", "final_val_loss", "train_time_sec",
                        "latency_ms", "model_size_mb",
                        "precision", "recall", "f1", "specificity", "roc_auc")},
                    "n_params": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0,
                })
            continue

        model = build_detector(
            temporal_arch=temporal_arch,
            n_features=n_features, window_size=window_size,
            hidden_size=hidden_size, n_gat_layers=n_gat_layers,
            n_gat_heads=n_gat_heads, dropout=dropout,
            use_temporal_attention=use_attn,
        )
        n_params = count_parameters(model)
        model_size = get_model_size_mb(model)

        trainer = AnomalyDetectionTrainer(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=lr),
            device=device_obj,
            prior_adjacency=prior_adjacency,
            feature_names=feature_cols,
            loss_fn=None,                       # MSE-only mode
        )

        t0 = time.time()
        fit_result = trainer.fit(
            train_loader=train_loader, val_loader=val_loader,
            n_epochs=n_epochs,
            early_stopping_patience=early_stopping_patience,
            verbose=verbose_epochs,
        )
        elapsed = time.time() - t0

        final_train_loss = fit_result["train_history"][-1]["loss"]
        final_val_loss = fit_result["best_val_loss"]
        print(f"       params={n_params:>9,}  size={model_size:>6.2f}MB  "
              f"train_loss={final_train_loss:.5f}  "
              f"val_loss={final_val_loss:.5f}  time={elapsed:.1f}s")

        metrics_by_thr = compute_anomaly_metrics(
            test_result=trainer.test(test_loader),
            thresholds=thresholds, score_type=score_type,
            point_adjust=point_adjust, verbose=False,
        )

        # After scoring: measure_latency_cpu restores the device, but keeping
        # it after evaluation avoids any CPU/GPU round-trip mid-experiment.
        latency_ms = float("nan")
        if measure_latency:
            latency_ms = measure_latency_cpu(
                trainer.model, input_shape=(1, window_size, n_features),
                adjacency=prior_adjacency, n_runs=latency_runs, warmup=3,
            )
            print(f"       CPU latency: {latency_ms:.2f} ms")

        _print_threshold_table(metrics_by_thr)

        for thr, m in metrics_by_thr.items():
            all_results.append({
                **config, "threshold": thr, "skipped": False,
                "n_params": n_params, "model_size_mb": model_size,
                "latency_ms": latency_ms,
                "final_train_loss": final_train_loss,
                "final_val_loss": final_val_loss,
                "train_time_sec": elapsed,
                **{k: m[k] for k in ("precision", "recall", "f1",
                                     "specificity", "roc_auc",
                                     "tp", "fp", "fn", "tn")},
            })

    results_df = pd.DataFrame(all_results)
    results_df["point_adjust"] = point_adjust
    if save_results_csv:
        results_df.to_csv(save_results_csv, index=False)
        print(f"Results saved -> {save_results_csv}")

    best_config = _select_best_config(results_df, protocol)
    return results_df, best_config


def run_hyperparameter_sweep_adj(*args, **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """
    Architecture sweep scored with the POINT-ADJUSTED protocol.

    Identical to run_hyperparameter_sweep in every other respect — same core
    implementation, so the two stay aligned automatically.
    """
    kwargs.setdefault("save_results_csv", "arch_sweep_results_adj.csv")
    kwargs["point_adjust"] = True
    return run_hyperparameter_sweep(*args, **kwargs)


def run_temporal_ablation(
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        prior_adjacency,
        feature_cols: List[str],
        window_size: int,
        temporal_archs: Sequence[str] = ("tcn", "gru", "rnn", "lstm"),
        attention_strategies: Sequence[bool] = (True, False),
        hidden_size: int = 64,
        n_gat_layers: int = 1,
        n_gat_heads: int = 4,
        dropout: float = 0.2,
        thresholds: Sequence[float] = (0.003, 0.005, 0.01, 0.02, 0.03, 0.05),
        **kwargs,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Temporal-encoder ablation: (temporal_arch x pooling strategy) at a fixed
    architecture size, with cost profiling switched on.

    This is a preset over run_hyperparameter_sweep — the architecture is held
    constant so the encoder and the pooling head are the only things varying,
    and n_params / model_size_mb / latency_ms come along for the
    accuracy-versus-cost table.  Because it is the same core, any fix to the
    sweep or to the metrics applies here too.

    Accepts every run_hyperparameter_sweep keyword (n_epochs, lr, device,
    point_adjust, save_results_csv, ...).
    """
    kwargs.setdefault("measure_latency", True)
    kwargs.setdefault("save_results_csv", "temporal_ablation_results.csv")
    return run_hyperparameter_sweep(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        prior_adjacency=prior_adjacency,
        feature_cols=feature_cols,
        window_size=window_size,
        hidden_sizes=[hidden_size],
        temporal_archs=temporal_archs,
        n_gat_layers_list=[n_gat_layers],
        n_gat_heads=n_gat_heads,
        dropout_rates=[dropout],
        attention_strategies=attention_strategies,
        thresholds=thresholds,
        **kwargs,
    )


def run_temporal_ablation_adj(*args, **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """Temporal-encoder ablation scored with the POINT-ADJUSTED protocol."""
    kwargs.setdefault("save_results_csv", "temporal_ablation_results_adj.csv")
    kwargs["point_adjust"] = True
    return run_temporal_ablation(*args, **kwargs)


def summarise_temporal_ablation(results_df: pd.DataFrame) -> pd.DataFrame:
    """Best threshold per (architecture, pooling), sorted by F1."""
    best = (results_df[~results_df["skipped"].fillna(False).astype(bool)]
            .sort_values("f1", ascending=False)
            .groupby(["temporal_arch", "use_temporal_attention"], as_index=False)
            .first()
            .sort_values("f1", ascending=False))

    protocol = ("point-adjusted" if bool(results_df["point_adjust"].iloc[0])
                else "point-wise")
    print(f"\n{'=' * 78}")
    print(f"  TEMPORAL ABLATION — best F1 per architecture ({protocol})")
    print(f"{'=' * 78}")
    for _, r in best.iterrows():
        latency = ("  latency=n/a" if pd.isna(r.latency_ms)
                   else f"  latency={r.latency_ms:6.2f}ms")
        print(f"  {r.temporal_arch:<11} attn={str(r.use_temporal_attention):<5} "
              f"F1={r.f1:.4f} @ th={r.threshold:<7.4f} "
              f"P={r.precision:.4f} R={r.recall:.4f} "
              f"params={r.n_params:>9,}{latency}")
    print(f"{'=' * 78}\n")
    return best


def _print_threshold_table(metrics_by_thr: Dict) -> None:
    print(f"       {'Threshold':>10}  {'Precision':>9}  {'Recall':>7}  {'F1':>6}")
    for thr, m in metrics_by_thr.items():
        print(f"       {thr:>10.4f}  {m['precision']:>9.4f}  "
              f"{m['recall']:>7.4f}  {m['f1']:>6.4f}")
    print()


def _select_best_config(results_df: pd.DataFrame, protocol: str) -> Dict:
    """Highest mean F1 across thresholds, over non-skipped configs."""
    valid = results_df[~results_df["skipped"].fillna(False).astype(bool)]
    if valid.empty:
        raise RuntimeError(
            "Every configuration was skipped — check that at least one "
            "hidden_size is divisible by n_gat_heads."
        )
    group_keys = ["window_size", "hidden_size", "temporal_arch",
                  "n_gat_layers", "n_gat_heads", "dropout"]
    if "use_temporal_attention" in valid.columns:
        group_keys.append("use_temporal_attention")
    mean_f1 = valid.groupby(group_keys)["f1"].mean()
    best_idx = mean_f1.idxmax()
    best_config = dict(zip(mean_f1.index.names, best_idx))
    best_config["mean_f1"] = float(mean_f1[best_idx])

    print(f"\n{'=' * 78}")
    print(f"  BEST ARCHITECTURE (mean F1 across thresholds, {protocol})")
    print(f"{'=' * 78}")
    for k, v in best_config.items():
        print(f"  {k:<20} = {v}")
    print(f"{'=' * 78}\n")
    return best_config


# =============================================================================
# PART 2 — loss-component ablation
# =============================================================================

# (label, ablation_mode, uncertainty_weighting, score_type)
ABLATION_CONFIGS: List[Tuple[str, str, str, str]] = [
    ("MSE only",              "mse_only",     "none",     "recon_errors"),
    ("MSE + Physics",         "mse+physics",  "none",     "fused_scores"),
    ("MSE + Temporal",        "mse+temporal", "none",     "recon_errors"),
    ("MSE + Graph",           "mse+graph",    "none",     "recon_errors"),
    ("Full (no uncertainty)", "full",         "none",     "fused_scores"),
    ("Full + Kendall",        "full",         "kendall",  "fused_scores"),
    # ("Full + GradNorm",     "full",         "gradnorm", "fused_scores"),
    # ("Full + DWA",          "full",         "dwa",      "fused_scores"),
]


def run_ablation_study(
        arch_config: Dict,                  # best_config from Part 1
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        prior_adjacency,
        feature_cols: List[str],
        rule_set,                           # PhysicsRuleSet over feature_cols
        # ── evaluation ────────────────────────────────────────────────────────
        thresholds: Sequence[float] = (0.1, 0.3, 0.5, 1.0),
        point_adjust: bool = False,
        # ── training ──────────────────────────────────────────────────────────
        n_epochs: int = 50,
        lr: float = 1e-3,
        early_stopping_patience: int = 10,
        physics_weight: float = 0.1,
        temporal_weight: float = 0.05,
        graph_weight: float = 0.05,
        use_huber: bool = False,
        physics_on_prediction: bool = True,
        # ── misc ──────────────────────────────────────────────────────────────
        ablation_configs: Optional[List[Tuple[str, str, str, str]]] = None,
        device: str = "cpu",
        verbose_epochs: bool = False,
        save_results_csv: Optional[str] = "ablation_results.csv",
) -> pd.DataFrame:
    """
    Train the Part-1 architecture from scratch under each loss configuration.

    Loss-component notes
    --------------------
    use_huber : default False.
        The "MSE only" arm runs through `loss_fn=None`, which uses F.mse_loss.
        With use_huber=True the composite arms would use smooth_l1 instead, so
        the arms would differ in their RECONSTRUCTION term as well as in the
        auxiliary term under test — the comparison would no longer isolate the
        component being ablated.  Keep False unless every arm is switched.

    physics_on_prediction : default True.
        Evaluates the physics rules on the model output so the physics term
        actually has a gradient.  With False the term is a constant and
        "MSE + Physics" reduces to "MSE only".

    graph_weight : used by the "mse+graph" and "full" arms.  The graph term is
        activated by the ablation_mode STRING; the weight only sets its
        magnitude, so a stray positive weight can never desynchronise n_tasks.
        `prior_adjacency` is forwarded to the loss on every batch, which is
        what makes `CompositeLoss._graph_regularization` fire.

    Returns one row per (ablation config x threshold).
    """
    ablation_configs = ablation_configs or ABLATION_CONFIGS
    device_obj = torch.device(device)
    n_features = len(feature_cols)
    prior_adjacency = _as_adjacency_tensor(prior_adjacency, n_features)
    protocol = "point-adjusted" if point_adjust else "point-wise"

    arch = {
        "window_size": int(arch_config["window_size"]),
        "hidden_size": int(arch_config["hidden_size"]),
        "temporal_arch": str(arch_config["temporal_arch"]),
        "n_gat_layers": int(arch_config["n_gat_layers"]),
        "n_gat_heads": int(arch_config["n_gat_heads"]),
        "dropout": float(arch_config["dropout"]),
        # Preserved from the sweep so Part 2 ablates the LOSS on exactly the
        # architecture Part 1 selected, pooling head included.
        "use_temporal_attention": bool(
            arch_config.get("use_temporal_attention", True)),
    }

    print(f"\n{'#' * 78}")
    print(f"  PART 2 — ABLATION STUDY — {len(ablation_configs)} loss configs")
    print(f"  Architecture: " + "  ".join(f"{k}={v}" for k, v in arch.items()))
    print(f"  Thresholds: {list(thresholds)}   metrics={protocol}")
    print(f"  physics_w={physics_weight}  temporal_w={temporal_weight}  "
          f"graph_w={graph_weight}  huber={use_huber}  "
          f"physics_on_prediction={physics_on_prediction}")
    print(f"{'#' * 78}\n")

    all_results = []
    for run_idx, (label, ablation_mode, uncertainty, score_type) in \
            enumerate(ablation_configs, 1):

        print(f"[{run_idx:>2}/{len(ablation_configs)}] {label:<30} "
              f"(mode={ablation_mode}, uw={uncertainty})")

        # Fresh weights for every configuration
        model = build_detector(
            temporal_arch=arch["temporal_arch"],
            n_features=n_features, window_size=arch["window_size"],
            hidden_size=arch["hidden_size"],
            n_gat_layers=arch["n_gat_layers"],
            n_gat_heads=arch["n_gat_heads"], dropout=arch["dropout"],
            use_temporal_attention=arch["use_temporal_attention"],
        )

        if ablation_mode == "mse_only":
            # F.mse_loss directly — identical reconstruction term to the
            # composite arms as long as use_huber is False.
            loss_fn = None
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            loss_fn = CompositeLoss(
                residual_calculator=PhysicsResidualCalculator(
                    rule_set=rule_set, feature_names=feature_cols,
                    residual_stds={},
                ),
                feature_names=feature_cols,
                uncertainty_weighting=uncertainty,
                ablation_mode=ablation_mode,
                physics_weight=physics_weight,
                temporal_weight=temporal_weight,
                graph_weight=graph_weight,
                use_huber=use_huber,
                physics_on_prediction=physics_on_prediction,
            ).to(device_obj)
            # Move to device BEFORE building the optimizer, or it holds stale
            # references to the CPU copies of log_sigma and they never update.
            optimizer = make_optimizer(model, loss_fn, lr=lr)

        trainer = AnomalyDetectionTrainer(
            model=model, optimizer=optimizer, device=device_obj,
            prior_adjacency=prior_adjacency, feature_names=feature_cols,
            loss_fn=loss_fn,
        )

        t0 = time.time()
        fit_result = trainer.fit(
            train_loader=train_loader, val_loader=val_loader,
            n_epochs=n_epochs,
            early_stopping_patience=early_stopping_patience,
            verbose=verbose_epochs,
        )
        elapsed = time.time() - t0

        final_train_loss = fit_result["train_history"][-1]["loss"]
        final_val_loss = fit_result["best_val_loss"]
        last = fit_result["train_history"][-1]
        print(f"       train_loss={final_train_loss:.5f}  "
              f"val_loss={final_val_loss:.5f}  time={elapsed:.1f}s")
        print(f"       components  mse={last.get('mse', 0):.5f}  "
              f"physics={last.get('physics', 0):.5f}  "
              f"temporal={last.get('temporal', 0):.5f}  "
              f"graph={last.get('graph', 0):.5f}")

        metrics_by_thr = compute_anomaly_metrics(
            test_result=trainer.test(test_loader),
            thresholds=thresholds, score_type=score_type,
            point_adjust=point_adjust, verbose=False,
        )
        _print_threshold_table(metrics_by_thr)

        for thr, m in metrics_by_thr.items():
            all_results.append({
                "run": run_idx, "loss_config": label,
                "ablation_mode": ablation_mode, "uncertainty": uncertainty,
                "score_type": score_type, "point_adjust": point_adjust,
                **arch,
                "final_train_loss": final_train_loss,
                "final_val_loss": final_val_loss,
                "train_time_sec": elapsed,
                "mse_component": last.get("mse", 0.0),
                "physics_component": last.get("physics", 0.0),
                "temporal_component": last.get("temporal", 0.0),
                "graph_component": last.get("graph", 0.0),
                "threshold": thr,
                **{k: m[k] for k in ("precision", "recall", "f1",
                                     "specificity", "roc_auc",
                                     "tp", "fp", "fn", "tn")},
            })

    results_df = pd.DataFrame(all_results)
    if save_results_csv:
        results_df.to_csv(save_results_csv, index=False)
        print(f"Ablation results saved -> {save_results_csv}")

    print(f"\n{'=' * 78}")
    print(f"  ABLATION SUMMARY — mean F1 across thresholds ({protocol})")
    print(f"{'=' * 78}")
    summary = (results_df
               .groupby(["loss_config", "ablation_mode", "uncertainty"])["f1"]
               .mean().reset_index().sort_values("f1", ascending=False))
    for _, row in summary.iterrows():
        print(f"  {row.loss_config:<30}  mean_F1={row.f1:.4f}")
    print(f"{'=' * 78}\n")

    return results_df


def run_ablation_study_adj(*args, **kwargs) -> pd.DataFrame:
    """
    Loss-component ablation scored with the POINT-ADJUSTED protocol.

    Identical to run_ablation_study in every other respect — same core
    implementation, so the two stay aligned automatically.
    """
    kwargs.setdefault("save_results_csv", "ablation_results_adj.csv")
    kwargs["point_adjust"] = True
    return run_ablation_study(*args, **kwargs)