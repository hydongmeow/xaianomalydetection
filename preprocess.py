"""
preprocess.py
=============
Shared preprocessing for SWaT and WADI.

Stage 1  lagged_mrmr            self-supervised feature ranking (no labels)
Stage 2  granger_screen         directed linear coupling  -> prior graph
         transfer_entropy_screen nonlinear coupling       -> prior graph
Stage 3  pre_pipeline           ranking + coupling in one call
Stage 4  build_dataloaders      windowing, scaling, train/val/test loaders

The training split must be all-normal: Stage 1 is self-supervised and Stage 2
estimates the nominal coupling structure, so any attack rows in `X` would be
baked into the prior graph.
"""

import gc
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    from statsmodels.tsa.stattools import grangercausalitytests
    _HAS_SM = True
except ImportError:                                     # pragma: no cover
    _HAS_SM = False
    warnings.warn("statsmodels not found — Granger screening disabled.")


def _subsample_rows(n: int, max_rows: int, rng) -> np.ndarray:
    if n <= max_rows:
        return np.arange(n)
    return np.sort(rng.choice(n, size=max_rows, replace=False))


# =============================================================================
# Stage 1 — self-supervised feature ranking
# =============================================================================

def lagged_mrmr(
    X: pd.DataFrame,
    horizon: int = 1,
    lag: int = 1,
    k: Optional[int] = None,
    redundancy_metric: str = "correlation",   # 'correlation' (fast) or 'mi'
    max_rows: int = 8000,
    random_state: int = 0,
    verbose: bool = True,
) -> Tuple[List[str], pd.Series, Dict]:
    """
    Self-supervised mRMR for unlabelled (all-normal) training data.

        relevance[j] = mean_k MI( x_j(t-lag) ; x_k(t+horizon) )   predictive coupling
        redundancy   = |spearman| (default) or MI between channels at time t

    Relevance is computed with one `mutual_info_regression` call per FUTURE
    channel rather than one per (past, future) pair: sklearn already returns
    the MI of every column of X against a single target, which turns an
    O(d^2) sweep of kNN estimators into O(d).  The values are unchanged.

    Returns (ranked_features, relevance_series, info).
    """
    rng = np.random.default_rng(random_state)
    cols = list(X.columns)
    d = len(cols)
    k = k or d

    Xv = X.to_numpy(dtype=np.float64)
    n = len(Xv)
    past = Xv[: n - horizon - lag]          # x(t - lag)
    future = Xv[lag + horizon:]             # x(t + horizon)
    idx = _subsample_rows(len(past), max_rows, rng)
    past, future = past[idx], future[idx]

    # ── Relevance ─────────────────────────────────────────────────────────────
    mi_matrix = np.zeros((d, d))            # [source j, target kk]
    for kk in range(d):
        mi_matrix[:, kk] = mutual_info_regression(
            past, future[:, kk], random_state=random_state
        )
    relevance = pd.Series(mi_matrix.mean(axis=1), index=cols)
    relevance = relevance.sort_values(ascending=False)

    # ── Redundancy ────────────────────────────────────────────────────────────
    if redundancy_metric == "correlation":
        red = X.iloc[idx].corr(method="spearman").abs().fillna(0.0)
    elif redundancy_metric == "mi":
        cur = Xv[idx]
        R = np.zeros((d, d))
        for a in range(d):
            R[a] = mutual_info_regression(cur, cur[:, a],
                                          random_state=random_state)
        red = pd.DataFrame((R + R.T) / 2.0, index=cols, columns=cols)
    else:
        raise ValueError("redundancy_metric must be 'correlation' or 'mi'")

    # ── Greedy mRMR ───────────────────────────────────────────────────────────
    selected = [relevance.index[0]]
    candidates = [c for c in cols if c != selected[0]]
    while len(selected) < k and candidates:
        scores = {c: relevance[c] - red.loc[c, selected].mean()
                  for c in candidates}
        best_c = max(scores, key=scores.get)
        selected.append(best_c)
        candidates.remove(best_c)

    if verbose:
        print(f"[mRMR] ranked {len(selected)}/{d} features "
              f"(horizon={horizon}, lag={lag}, redundancy={redundancy_metric})")
    return selected, relevance, {"redundancy": red, "mi_matrix": mi_matrix}


# =============================================================================
# Stage 2 — coupling screens
# =============================================================================

def granger_screen(
    X: pd.DataFrame,
    maxlag: int = 5,
    stride: int = 5,             # downsample 1 Hz -> 0.2 Hz to cut length
    max_len: int = 12000,
    significance: float = 0.05,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Directed Granger coupling.  C[i, j] = strength of  x_j -> x_i
    (does the past of column j help predict column i).

    Returns (strength [-log10 p], adjacency [0/1]).
    """
    if not _HAS_SM:
        raise ImportError("statsmodels required for Granger screening.")

    Xs = X.iloc[::stride].reset_index(drop=True)
    if len(Xs) > max_len:
        Xs = Xs.iloc[:max_len]
    cols = list(Xs.columns)
    m = len(cols)
    values = Xs.to_numpy(dtype=np.float64)
    strength = np.zeros((m, m))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(m):            # target
            for j in range(m):        # source
                if i == j:
                    continue
                pair = values[:, [i, j]]          # [target, source]
                try:
                    # `verbose` was deprecated in statsmodels 0.14 and removed
                    # in 0.15 — passing it raises TypeError on current builds.
                    res = grangercausalitytests(pair, maxlag=maxlag)
                    pmin = min(res[L][0]["ssr_ftest"][1]
                               for L in range(1, maxlag + 1))
                    strength[i, j] = -np.log10(max(pmin, 1e-300))
                except Exception:
                    strength[i, j] = 0.0

    S = pd.DataFrame(strength, index=cols, columns=cols)
    A = (S > -np.log10(significance)).astype(int)
    if verbose:
        print(f"[granger] coupling matrix {m}x{m}, "
              f"{int(A.values.sum())} directed edges @ p<{significance}")
    return S, A


def transfer_entropy_screen(
    X: pd.DataFrame,
    bins: int = 8,
    stride: int = 5,
    max_len: int = 12000,
    te_threshold: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Binned (histogram) transfer entropy at lag 1:

        TE(j -> i) = sum p(i', i, j) log[ p(i'|i,j) / p(i'|i) ]

    Nonparametric alternative to Granger; captures nonlinear coupling.
    """
    Xs = X.iloc[::stride].reset_index(drop=True)
    if len(Xs) > max_len:
        Xs = Xs.iloc[:max_len]
    cols = list(Xs.columns)
    m = len(cols)

    # Equal-frequency discretisation per channel
    D = np.zeros_like(Xs.to_numpy(), dtype=int)
    for c in range(m):
        D[:, c] = pd.qcut(
            Xs.iloc[:, c].rank(method="first"),
            q=min(bins, Xs.iloc[:, c].nunique()),
            labels=False, duplicates="drop",
        )

    def _te(i_idx: int, j_idx: int) -> float:
        i_t1, i_t, j_t = D[1:, i_idx], D[:-1, i_idx], D[:-1, j_idx]
        joint = pd.crosstab([i_t1, i_t], j_t, normalize=True)
        p_it1_it = pd.crosstab(i_t1, i_t, normalize=True)
        p_it_jt = pd.crosstab(i_t, j_t, normalize=True)
        p_it = pd.Series(i_t).value_counts(normalize=True)

        te = 0.0
        for (a, b), row in joint.iterrows():
            for cc, p_abc in row.items():
                if p_abc <= 0:
                    continue
                num = p_abc * p_it.get(b, 1e-12)
                den = (
                    (p_it1_it.loc[a, b]
                     if (a in p_it1_it.index and b in p_it1_it.columns)
                     else 1e-12)
                    * (p_it_jt.loc[b, cc]
                       if (b in p_it_jt.index and cc in p_it_jt.columns)
                       else 1e-12)
                )
                if den > 0:
                    te += p_abc * np.log(num / den + 1e-12)
        return max(te, 0.0)

    TE = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i != j:
                TE[i, j] = _te(i, j)

    S = pd.DataFrame(TE, index=cols, columns=cols)
    positive = TE[TE > 0]
    thr = (te_threshold if te_threshold is not None
           else (np.quantile(positive, 0.75) if positive.size else 0.0))
    A = (S > thr).astype(int)
    if verbose:
        print(f"[transfer-entropy] {m}x{m}, "
              f"{int(A.values.sum())} edges @ TE>{thr:.4f}")
    return S, A


def _add_self_loops_df(adj: pd.DataFrame) -> pd.DataFrame:
    """Self-loops guarantee no all-zero adjacency row (softmax over an
    all-masked row is NaN)."""
    assert adj.shape[0] == adj.shape[1], "adjacency must be square"
    arr = adj.to_numpy(copy=True).astype(float)
    np.fill_diagonal(arr, 1.0)
    return pd.DataFrame(arr, index=adj.index, columns=adj.columns)


# =============================================================================
# Stage 3 — combined pipeline
# =============================================================================

def pre_pipeline(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    *,
    coupling: str = "te",
    final_k: Optional[int] = None,
    mrmr_horizon: int = 1,
    verbose: bool = True,
) -> Dict:
    """
    Feature ranking + coupling graph.

    Returns
    -------
    X_selected        columns reordered by mRMR rank (and truncated to final_k)
    selected_features the ranked column order
    relevance         mRMR relevance -> soft feature weights for XAI Level 1
    coupling_strength raw coupling matrix
    prior_adjacency   binary graph WITH self-loops -> GAT mask and the graph
                      regularisation term in CompositeLoss
    """
    if y is not None:
        assert int(np.nansum(np.asarray(y))) == 0, \
            "Training y must be all 0 (normal regime) for Stage-1 selection."

    ranked, relevance, mrmr_info = lagged_mrmr(
        X, horizon=mrmr_horizon, k=final_k, verbose=verbose)
    X = X[ranked]

    coupling_strength, adjacency = None, None
    if coupling == "granger":
        coupling_strength, adjacency = granger_screen(X, verbose=verbose)
    elif coupling == "te":
        coupling_strength, adjacency = transfer_entropy_screen(X, verbose=verbose)
    elif coupling != "none":
        raise ValueError("coupling must be 'granger', 'te' or 'none'")

    if adjacency is not None:
        adjacency = _add_self_loops_df(adjacency)

    return {
        "X_selected": X,
        "selected_features": ranked,
        "relevance": relevance,
        "coupling_strength": coupling_strength,
        "prior_adjacency": adjacency,
        "redundancy": mrmr_info["redundancy"],
    }


# =============================================================================
# Stage 4 — windowing and loaders
# =============================================================================

def _make_windows(data, labels, win, step, lbl_pos, verbose=True):
    """
    Build next-step prediction windows.

    Returns
    -------
    X_wins : (N, win, d)  input windows           = data[i : i+win]
    y_wins : (N, d)       prediction target       = data[i+win]
    l_wins : (N,)         scalar 0/1 label per window

    Label alignment
    ---------------
    The anomaly score is the reconstruction error of the TARGET timestep
    data[i+win], so the label must be the label of that same timestep.

        lbl_pos="last"  -> labels_1d[i + win]      (target timestep)
        lbl_pos="any"   -> max over the window AND the target,
                           labels_1d[i : i + win + 1]

    Previously "last" used labels_1d[i + win - 1], the last timestep of the
    INPUT window — one step ahead of the point being scored.  On SWaT, where
    attacks run for minutes, the effect on aggregate metrics is small, but
    every window at an attack boundary was mislabelled.
    """
    n = len(data)

    # Multi-feature labels -> one scalar per timestep, OR across features
    labels_1d = (labels.max(axis=1) if labels.ndim == 2 else labels).astype(np.int64)

    indices = range(0, n - win, step)          # i + win <= n - 1
    X_wins, y_wins, l_wins = [], [], []

    for i in indices:
        X_wins.append(data[i: i + win])         # (win, d)
        y_wins.append(data[i + win])            # (d,)
        if lbl_pos == "last":
            l_wins.append(int(labels_1d[i + win]))
        elif lbl_pos == "any":
            l_wins.append(int(labels_1d[i: i + win + 1].max()))
        else:
            raise ValueError("label_position must be 'last' or 'any'")

    X_out = np.stack(X_wins, axis=0)
    y_out = np.stack(y_wins, axis=0)
    l_out = np.asarray(l_wins, dtype=np.int64)

    if verbose:
        print(f"  [_make_windows] data={data.shape}  win={win}  step={step}  "
              f"lbl_pos={lbl_pos}")
        print(f"    -> X={X_out.shape}  y={y_out.shape}  labels={l_out.shape}  "
              f"anomaly_rate={l_out.mean():.4f}")
    return X_out, y_out, l_out


def build_dataloaders(
        train_normal,
        y_train,
        test_df,
        y_test,
        window_size: int = 10,
        step_size: int = 1,
        scaler_type: str = "minmax",
        fit_scaler_on: str = "train",
        batch_size: int = 64,
        shuffle_train: bool = True,
        label_position: str = "last",
        val_split: float = 0.2,
        val_split_mode: str = "random",     # 'random' | 'chronological'
        random_state: int = 42,
        verbose: bool = True,
):
    """
    Scale, window, and wrap SWaT/WADI splits into DataLoaders.

    fit_scaler_on : 'train' | 'all'
        'train' is the honest option.  'all' fits the scaler on train+test and
        leaks the test distribution into preprocessing.

    val_split_mode : 'random' | 'chronological'
        'random' shuffles windows before splitting, so a validation window can
        overlap a training window by up to `window_size - 1` timesteps and
        validation loss reads optimistically.  'chronological' takes the last
        `val_split` fraction of the training period instead, which is the
        correct protocol for time series; it is not the default only because
        it changes existing numbers.

    Returns (train_loader, val_loader, test_loader), each yielding
    (x_window (B, W, d), y_target (B, d), label (B,)).
    """
    X_tr = np.asarray(train_normal, dtype=np.float32)
    X_te = np.asarray(test_df, dtype=np.float32)
    y_tr = np.asarray(y_train, dtype=np.int64)
    y_te = np.asarray(y_test, dtype=np.int64)
    del train_normal, test_df, y_train, y_test
    gc.collect()

    scalers = {
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
        "standard": StandardScaler,
    }
    scaler = scalers.get(scaler_type.lower(), StandardScaler)()

    if fit_scaler_on == "all":
        warnings.warn(
            "fit_scaler_on='all' fits the scaler on train+test — this leaks "
            "test statistics into preprocessing.",
            RuntimeWarning,
        )
        combined = np.concatenate([X_tr, X_te], axis=0)
        scaler.fit(combined)
        del combined
        gc.collect()
    else:
        scaler.fit(X_tr)

    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)
    del scaler
    gc.collect()

    X_tr_w, y_tr_w, l_tr_w = _make_windows(
        X_tr, y_tr, window_size, step_size, label_position, verbose)
    X_te_w, y_te_w, l_te_w = _make_windows(
        X_te, y_te, window_size, step_size, label_position, verbose)
    del X_tr, X_te, y_tr, y_te
    gc.collect()

    if val_split_mode == "chronological":
        cut = int(len(X_tr_w) * (1.0 - val_split))
        X_val_w, y_val_w, l_val_w = X_tr_w[cut:], y_tr_w[cut:], l_tr_w[cut:]
        X_tr_w, y_tr_w, l_tr_w = X_tr_w[:cut], y_tr_w[:cut], l_tr_w[:cut]
    elif val_split_mode == "random":
        (X_tr_w, X_val_w, y_tr_w, y_val_w,
         l_tr_w, l_val_w) = train_test_split(
            X_tr_w, y_tr_w, l_tr_w,
            test_size=val_split, random_state=random_state, shuffle=True,
        )
    else:
        raise ValueError("val_split_mode must be 'random' or 'chronological'")

    def _ds(X, y, l):
        return TensorDataset(torch.from_numpy(np.ascontiguousarray(X)),
                             torch.from_numpy(np.ascontiguousarray(y)),
                             torch.from_numpy(np.ascontiguousarray(l)))

    train_ds = _ds(X_tr_w, y_tr_w, l_tr_w)
    val_ds = _ds(X_val_w, y_val_w, l_val_w)
    test_ds = _ds(X_te_w, y_te_w, l_te_w)

    del X_tr_w, X_val_w, X_te_w, y_tr_w, y_val_w, y_te_w, l_tr_w, l_val_w, l_te_w
    gc.collect()

    if verbose:
        print(f"  [loaders] train={len(train_ds)}  val={len(val_ds)}  "
              f"test={len(test_ds)}  batch_size={batch_size}  "
              f"val_split_mode={val_split_mode}")

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )
