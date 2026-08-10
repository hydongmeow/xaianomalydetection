import inspect
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.optim.lr_scheduler as _lr_sched
from dataset import complete_swat_preprocessing

_orig_plateau_init = _lr_sched.ReduceLROnPlateau.__init__
if "verbose" not in inspect.signature(_orig_plateau_init).parameters:
    def _patched_plateau_init(self, *args, **kwargs):
        kwargs.pop("verbose", None)
        _orig_plateau_init(self, *args, **kwargs)
    _lr_sched.ReduceLROnPlateau.__init__ = _patched_plateau_init

from preprocess import pre_pipeline, build_dataloaders
from models import GraphTemporalAttnDetector
from piloss import CompositeLoss, PhysicsRuleSet, PhysicsResidualCalculator
from train import AnomalyDetectionTrainer, make_optimizer, compute_anomaly_metrics_absolute
import xai_functions as _xai

_orig_explain_temporal_attention = _xai.explain_temporal_attention

def _patched_explain_temporal_attention(*args, **kwargs):
    kwargs.pop("pred", None)
    return _orig_explain_temporal_attention(*args, **kwargs)

_xai.explain_temporal_attention = _patched_explain_temporal_attention

explain_feature_relevance = _xai.explain_feature_relevance
explain_gat_spatial = _xai.explain_gat_spatial
explain_temporal_attention = _xai.explain_temporal_attention
explain_loss_components = _xai.explain_loss_components
explain_sample = _xai.explain_sample

SEED = 0
DEVICE = torch.device("cpu")


# =============================================================================
# 1. SWaT-style attack-scenario simulator
# =============================================================================

FEATURES = [
    "FIT101", "LIT101", "MV101", "P101",
    "FIT201", "MV201", "P204", "P206",
    "LIT301", "FIT301",
    "LIT401", "FIT401",
    "AIT402",
    "FIT501", "FIT502", "FIT503",
    "PIT501", "PIT503",
]

def extract_attack_schedule(labels: np.ndarray, attack_names: pd.Series = None,
                             min_gap: int = 1) -> list:
    """
    Convert a binary anomaly label array into a list of contiguous attack
    windows: [{"start": int, "end": int, "type": str}, ...]

    Parameters
    ----------
    labels : np.ndarray (n,)  0/1 array (e.g. test_label)
    attack_names : pd.Series, optional
        Same length as `labels`. If your merged.csv has a column identifying
        which attack each row belongs to (e.g. an "Attack_Name" or
        "attack_type" column from the SWaT attack log), pass it here so each
        window gets a real label instead of a generic "attack_i" tag.
    min_gap : int
        Merge windows separated by a gap of <= min_gap normal rows (handles
        single stray 0s inside an attack due to label noise / smoothing).

    Returns
    -------
    attack_schedule : list[dict]
    """
    labels = np.asarray(labels).astype(int)
    n = len(labels)

    # find indices where label == 1
    attack_idx = np.where(labels == 1)[0]
    if len(attack_idx) == 0:
        return []

    # split into contiguous runs (allowing gaps <= min_gap)
    breaks = np.where(np.diff(attack_idx) > (min_gap + 1))[0]
    runs = np.split(attack_idx, breaks + 1)

    attack_schedule = []
    for i, run in enumerate(runs, start=1):
        start, end = int(run[0]), int(run[-1]) + 1  # end exclusive
        if attack_names is not None:
            # majority-vote the name within this window
            names_in_window = attack_names.iloc[start:end]
            kind = names_in_window.mode().iloc[0] if not names_in_window.empty else f"attack_{i}"
        else:
            kind = f"attack_{i}"
        attack_schedule.append({"start": start, "end": end, "type": str(kind)})

    return attack_schedule

def main():
    merged = pd.read_csv("swat/merged.csv")
    X_swat, y_swat = complete_swat_preprocessing(merged)
    X_swat = X_swat.drop(columns=[c for c in X_swat.columns if X_swat[c].nunique() == 1])
    feature_names = X_swat.columns.tolist()

    split_index = int(len(X_swat) * 0.8)
    train_df = X_swat.iloc[:split_index]
    train_label = np.array(y_swat.iloc[:split_index])
    test_df = X_swat.iloc[split_index:]
    test_label = np.array(y_swat.iloc[split_index:])
    assert train_label.sum() == 0, "Training data must be all-normal."

    # If merged.csv has a named attack/label column (common in SWaT dumps,
    # e.g. "Attack" text field), slice the matching rows for the test split
    # and pass it in; otherwise omit attack_names entirely.
    attack_names = merged["Attack"].iloc[split_index:].reset_index(drop=True) \
        if "Attack" in merged.columns else None

    attack_schedule = extract_attack_schedule(test_label, attack_names=attack_names)

    pre_out = pre_pipeline(
        train_df, y=train_label,
        coupling="te",
        final_k=46, #len(FEATURES),
        mrmr_horizon=1,
        verbose=True,
    )
    feature_cols = pre_out["selected_features"]
    relevance = pre_out["relevance"]
    prior_adjacency_df = pre_out["prior_adjacency"]

    train_sel = train_df[feature_cols]
    test_sel = test_df[feature_cols]

    # ── 2. Sliding-window dataloaders ───────────────────────────────────────
    print("\n" + "=" * 72)
    print("STEP 2 — build_dataloaders(): windowing + scaling + train/val/test split")
    print("=" * 72)
    WINDOW_SIZE = 30
    train_loader, val_loader, test_loader = build_dataloaders(
        train_sel, train_label, test_sel, test_label,
        window_size=WINDOW_SIZE,
        step_size=1,
        scaler_type="minmax",
        fit_scaler_on="train",
        batch_size=128,
        shuffle_train=True,
        label_position="last",
        val_split=0.2,
    )

    # ── 3. Model / Loss / Optimizer / Trainer ───────────────────────────────
    print("\n" + "=" * 72)
    print("STEP 3 — Default GraphTemporalAttnDetector + MSE + TemporalConsistencyLoss")
    print("=" * 72)
    n_features = len(feature_cols)
    prior_adjacency = torch.tensor(prior_adjacency_df.values, dtype=torch.float32)

    model = GraphTemporalAttnDetector(
        n_features=n_features,
        window_size=WINDOW_SIZE,
        temporal_arch = 'lstm'
        # all other arguments left at their defaults ("default model")
    )

    rule_set = PhysicsRuleSet(feature_cols, plant="swat")
    residual_calc = PhysicsResidualCalculator(rule_set=rule_set, feature_names=feature_cols)

    # ablation_mode="mse+temporal"  -> MSE reconstruction + TemporalConsistencyLoss
    #                                  only (physics term excluded)
    # uncertainty_weighting="none"  -> fixed default weights, no learned/optimized
    #                                  task weighting (no Kendall/GradNorm/DWA)
    loss_fn = CompositeLoss(
        residual_calculator=residual_calc,
        feature_names=feature_cols,
        uncertainty_weighting="kendall",
        ablation_mode="full",
    )

    optimizer = make_optimizer(model, loss_fn, lr=1e-3)

    trainer = AnomalyDetectionTrainer(
        model=model,
        optimizer=optimizer,
        device=DEVICE,
        prior_adjacency=prior_adjacency,
        feature_names=feature_cols,
        loss_fn=loss_fn,
    )

    print(f"  n_features={n_features}  window_size={WINDOW_SIZE}  "
          f"hidden_size={model.hidden_size}  temporal_arch={model.temporal_arch}")
    print(f"  loss: ablation_mode='{loss_fn.ablation_mode}'  "
          f"uncertainty_weighting=none (fixed physics_weight={loss_fn.physics_weight}, "
          f"temporal_weight={loss_fn.temporal_weight} -- unused/used per ablation mode)")

    # ── 4. Train for 50 epochs ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("STEP 4 — Training for 50 epochs")
    print("=" * 72)
    N_EPOCHS = 20
    fit_result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=N_EPOCHS,
        early_stopping_patience=10,
        verbose=True,
    )
    print(f"\n  Final train loss : {fit_result['train_history'][-1]['loss']:.5f}")
    print(f"  Best val loss     : {fit_result['best_val_loss']:.5f}")

    # Restore best checkpoint before evaluation / explanation
    if trainer.best_model_state is not None:
        model.load_state_dict(trainer.best_model_state)

    # ── 5. Evaluate ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("STEP 5 — Evaluation on the attack-injected test set")
    print("=" * 72)
    test_result = trainer.test(test_loader)
    _ = compute_anomaly_metrics_absolute(
        test_result=test_result,
        thresholds=[0.025, 0.05, 0.1, 0.2, 0.3, 0.5],
        score_type="recon_errors",  # MSE+temporal only -> physics_scores are 0
        verbose=True,
    )

    # ── 6. XAI demonstrations ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("STEP 6 — Explainability (all four XAI levels from xai_functions.py)")
    print("=" * 72)

    test_ds = test_loader.dataset          # TensorDataset(x_win, y_next, label)
    labels_arr = test_ds.tensors[2].numpy()
    windows_arr = test_ds.tensors[0]
    n_windows = len(test_ds)

    # A representative NORMAL window (well clear of any attack)
    normal_positions = np.where(labels_arr == 0)[0]
    normal_idx = int(normal_positions[len(normal_positions) // 2])

    # Map each attack type to one representative window inside its span
    # (window i's label reflects timestep i+WINDOW_SIZE-1 in the raw series,
    # so we just search for label==1 windows and tag them by which attack
    # schedule entry they fall in).
    def find_window_for_attack(atk):
        # a window is "inside" the attack if its last raw timestep index
        # (i + WINDOW_SIZE - 1, before the val/test split shuffling — test
        # loader is NOT shuffled so raw order is preserved) falls in [start, end)
        candidates = [i for i in range(n_windows)
                      if labels_arr[i] == 1
                      and atk["start"] <= (i + WINDOW_SIZE - 1) < atk["end"]]
        return candidates[len(candidates) // 2] if candidates else None

    attack_sample_idx = {}
    for atk in attack_schedule:
        idx = find_window_for_attack(atk)
        if idx is not None:
            attack_sample_idx[atk["type"]] = idx

    model.eval()

    # ── Level 1: global feature relevance + Granger coupling ────────────────
    x_norm, y_norm, l_norm = test_ds[normal_idx]
    x_norm = x_norm.unsqueeze(0).to(DEVICE)
    y_norm = y_norm.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_norm = model(x_norm, prior_adjacency)
    per_feat_mse_norm = torch.nn.functional.mse_loss(
        pred_norm, y_norm, reduction="none"
    ).squeeze(0).numpy()

    explain_feature_relevance(
        relevance=relevance,
        prior_adjacency=prior_adjacency_df,
        feature_cols=feature_cols,
        top_k=min(25, len(feature_cols)),
        prediction_error_per_feature=per_feat_mse_norm,
        save_path="xai_level1_relevance.png",
    )

    # ── Level 2: GAT spatial internals (model-level, sample-independent) ────
    explain_gat_spatial(
        model=model,
        adjacency=prior_adjacency,
        feature_cols=feature_cols,
        top_k=min(25, len(feature_cols)),
        save_path="xai_level2_gat.png",
    )

    # ── Level 3 & 4: temporal attention + loss breakdown, per scenario ───────
    demo_samples = [("normal", normal_idx, int(l_norm))]
    for atk_type, idx in attack_sample_idx.items():
        demo_samples.append((atk_type, idx, 1))

    for tag, idx, lbl in demo_samples:
        x_win, y_true, _ = test_ds[idx]
        x_win = x_win.unsqueeze(0).to(DEVICE)
        y_true = y_true.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = model(x_win, prior_adjacency)
        per_feat_mse = torch.nn.functional.mse_loss(
            pred, y_true, reduction="none"
        ).squeeze(0).numpy()

        print(f"\n---- Sample: {tag}  (window idx {idx}) ----")
        # NOTE: explain_temporal_attention only captures its forward-hook
        # weights when it runs the forward pass itself, so we deliberately
        # do NOT pass our already-computed `pred` here (passing it would
        # skip the internal forward() call and leave the hook's capture
        # buffer empty).
        explain_temporal_attention(
            model=model,
            x_window=x_win,
            adjacency=prior_adjacency,
            feature_cols=feature_cols,
            sample_label=lbl,
            recon_error_per_feature=per_feat_mse,
            save_path=f"xai_level3_temporal_{tag}.png",
        )
        # Panel C of this figure now shows ONLY the weighted share of each
        # top-level loss term (mse / physics / temporal / graph); the
        # per-sensor temporal terms are collapsed into the temporal slice and
        # just the top-1 driving sensor is named on the bar.
        _, loss_breakdown = explain_loss_components(
            model=model,
            loss_fn=loss_fn,
            x_window=x_win,
            y_true=y_true,
            adjacency=prior_adjacency,
            feature_cols=feature_cols,
            save_path=f"xai_level4_loss_{tag}.png",
        )

        # Every loss term the composite objective actually reports is plotted,
        # however many there are (mse / physics / temporal / graph / ...).
        contrib = loss_breakdown.get("loss_term_contributions", {})
        if contrib:
            share = "  ".join(f"{term}={info['fraction'] * 100:.1f}%"
                              for term, info in contrib.items())
            print(f"    final-loss composition ({len(contrib)} terms): {share}")
            for term, info in contrib.items():
                if info.get("top_sub_term") is not None:
                    print(f"      {term}: top-1 of {info['n_sub_terms']} "
                          f"sub-terms = {info['top_sub_term']} "
                          f"({info['top_sub_term_value']:.6f})")
        else:
            print("    final-loss composition: (no scalar terms reported)")

    # ── Composite: all four levels in one call, on one clear attack sample ──
    if attack_sample_idx:
        showcase_type, showcase_idx = next(iter(attack_sample_idx.items()))
        x_show, y_show, l_show = test_ds[showcase_idx]
        explain_sample(
            model=model,
            loss_fn=loss_fn,
            x_window=x_show.unsqueeze(0).to(DEVICE),
            y_true=y_show.unsqueeze(0).to(DEVICE),
            adjacency=prior_adjacency,
            feature_cols=feature_cols,
            sample_label=int(l_show),
            relevance=relevance,
            prior_adjacency_df=prior_adjacency_df,
            save_dir=".",
            prefix=f"xai_composite_{showcase_type}",
        )

    print("\n" + "=" * 72)
    print("DEMO COMPLETE — figures saved in the current working directory.")
    print("=" * 72)


if __name__ == "__main__":
    main()