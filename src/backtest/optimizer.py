# src/backtest/optimizer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd


@dataclass
class OptimizerParams:
    # objective: minimize 0.5 w^T Sigma w - lam * mu^T w
    lam: float = 1.0

    # turnover penalty (L1 on weight changes): gamma * sum_i |w_i - w_prev_i|
    turnover_gamma: float = 0.0

    # constraints
    w_max: float = 0.01          # max absolute position weight
    gross_max: float = 2.0       # max sum(|w|); set None to disable

    # covariance shrinkage
    shrink_delta: float = 0.2    # 0 -> no shrink, 1 -> fully diagonal

    # numerical
    max_iter: int = 500
    step_size: float = 0.05
    tol: float = 1e-8


def _align_inputs(
    mu: pd.Series,
    Sigma: pd.DataFrame,
    beta: Optional[pd.Series] = None,
    sector: Optional[pd.Series] = None,
    w_prev: Optional[pd.Series] = None,
) -> Tuple[pd.Series, pd.DataFrame, Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
    """
    Ensure all objects share the same asset index, in the same order.
    We drop any asset missing from Sigma or mu.
    """
    if not isinstance(mu, pd.Series):
        raise TypeError("mu must be a pd.Series indexed by asset id (e.g., permno).")
    if not isinstance(Sigma, pd.DataFrame):
        raise TypeError("Sigma must be a pd.DataFrame with matching index/cols.")

    common = mu.index.intersection(Sigma.index).intersection(Sigma.columns)
    common = common.drop_duplicates()

    mu2 = mu.loc[common].astype(float)
    Sigma2 = Sigma.loc[common, common].astype(float)

    beta2 = beta.loc[common].astype(float) if beta is not None else None
    sector2 = sector.loc[common] if sector is not None else None
    w_prev2 = w_prev.reindex(common).fillna(0.0).astype(float) if w_prev is not None else None

    return mu2, Sigma2, beta2, sector2, w_prev2


def shrink_covariance(Sigma: pd.DataFrame, delta: float) -> pd.DataFrame:
    """
    Simple shrinkage toward diagonal:
        Sigma_shrunk = (1-delta)*Sigma + delta*diag(Sigma)
    """
    if delta < 0 or delta > 1:
        raise ValueError("shrink_delta must be in [0, 1].")

    diag = np.diag(np.diag(Sigma.values))
    shr = (1.0 - delta) * Sigma.values + delta * diag
    return pd.DataFrame(shr, index=Sigma.index, columns=Sigma.columns)


def _project_dollar_neutral(w: np.ndarray) -> np.ndarray:
    # enforce sum(w)=0 by removing mean
    return w - np.mean(w)


def _project_beta_neutral(w: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Project weights to satisfy beta^T w = 0.

    If beta is degenerate (near-zero norm), return w unchanged.
    """
    denom = float(beta @ beta)
    if denom < 1e-12:
        return w
    # Remove the component of w along beta
    return w - beta * float(beta @ w) / denom


def _project_sector_neutral(w: np.ndarray, sector: np.ndarray) -> np.ndarray:
    """Project weights so that within each sector group, sum of weights is zero.

    Sector neutrality constraint:
        For each sector k: sum_{i in k} w_i = 0

    Notes:
    - Assets with missing sector (NaN/NA) are left unchanged by this projection.
    - This is an alternating-projection style operator; combine with dollar/beta/box/gross projections.
    """
    # Treat sector as object array to safely handle strings / ints / pandas NA
    sec = np.asarray(sector, dtype=object)

    # Identify non-missing sector labels
    missing = pd.isna(sec)
    if np.all(missing):
        return w

    w2 = w.copy()
    # Unique labels among non-missing
    labels = pd.unique(sec[~missing])

    for lab in labels:
        idx = (sec == lab)
        # skip empty / degenerate groups
        cnt = int(np.sum(idx))
        if cnt <= 1:
            continue
        # enforce group sum zero by removing within-group mean
        w2[idx] = w2[idx] - float(np.mean(w2[idx]))

    return w2


def _project_box(w: np.ndarray, w_max: float) -> np.ndarray:
    return np.clip(w, -w_max, w_max)


def _project_gross(w: np.ndarray, gross_max: float) -> np.ndarray:
    """
    Enforce sum(|w|) <= gross_max by scaling down if needed.
    """
    g = np.sum(np.abs(w))
    if g <= gross_max + 1e-12:
        return w
    if g == 0:
        return w
    return w * (gross_max / g)


def optimize_mean_variance(
    mu: pd.Series,
    Sigma: pd.DataFrame,
    beta: Optional[pd.Series] = None,
    sector: Optional[pd.Series] = None,
    w_prev: Optional[pd.Series] = None,
    params: Optional[OptimizerParams] = None,
    return_diagnostics: bool = False,
) -> pd.Series | Tuple[pd.Series, Dict[str, Any]]:
    """
    Phase 8 - Step 1:
    - Objective: minimize 0.5 w^T Sigma w - lam * mu^T w
    - Constraints (enabled now): sum(w)=0, beta^T w = 0 (if beta provided), |w_i|<=w_max, sum(|w|)<=gross_max (optional)

    beta/sector are accepted in signature for chaining later. w_prev is used only for turnover penalty if provided.
    """
    if params is None:
        params = OptimizerParams()

    mu, Sigma, beta, sector, w_prev = _align_inputs(mu, Sigma, beta, sector, w_prev)

    # Turnover penalty reference weights (aligned). If not provided, treat as zeros.
    if w_prev is None:
        w_prev_vec = np.zeros(len(mu), dtype=float)
    else:
        w_prev_vec = w_prev.values.astype(float)

    beta_vec = beta.values.astype(float) if beta is not None else None
    sector_vec = sector.values if sector is not None else None

    # shrink covariance for stability
    Sigma_s = shrink_covariance(Sigma, params.shrink_delta)
    S = Sigma_s.values
    m = mu.values

    n = len(mu)
    if n == 0:
        raise ValueError("No assets after alignment.")

    # initialize weights at 0
    w = np.zeros(n, dtype=float)

    # Precompute for cheap grad: grad = S w - lam * m
    # Gradient descent with projections
    for it in range(params.max_iter):
        w_prev_it = w.copy()

        grad = S @ w - params.lam * m

        # L1 turnover penalty: gamma * sum |w - w_prev|
        # Use a smooth approximation to sign(x): x / (|x| + eps)
        if params.turnover_gamma and params.turnover_gamma > 0:
            eps = 1e-6
            diff = w - w_prev_vec
            subgrad = diff / (np.abs(diff) + eps)
            grad = grad + params.turnover_gamma * subgrad

        eff_step = params.step_size / (1.0 + float(params.turnover_gamma))
        w = w - eff_step * grad

        # ------------------------------------------------------------
        # Projections onto constraints.
        #
        # IMPORTANT: Box/gross projections can re-introduce beta exposure.
        # To enforce beta neutrality correctly, we alternate projections a
        # few times so the final iterate is close to the intersection of:
        #   sum(w)=0, beta^T w = 0, |w_i|<=w_max, sum(|w|)<=gross_max.
        # ------------------------------------------------------------
        for _ in range(25):
            # Linear constraints
            w = _project_dollar_neutral(w)
            if beta_vec is not None:
                w = _project_beta_neutral(w, beta_vec)
                w = _project_dollar_neutral(w)
            if sector_vec is not None:
                w = _project_sector_neutral(w, sector_vec)
                w = _project_dollar_neutral(w)

            # Box constraint
            w = _project_box(w, params.w_max)

            # Gross constraint (scaling down cannot violate box)
            if params.gross_max is not None:
                w = _project_gross(w, params.gross_max)

        # convergence check
        diff = np.max(np.abs(w - w_prev_it))
        if diff < params.tol:
            break


    # Final hard projection to reduce residual constraint drift.
    # (Box/gross can fight linear constraints; alternating projections need enough rounds.)
    for _ in range(100):
        w = _project_dollar_neutral(w)
        if beta_vec is not None:
            w = _project_beta_neutral(w, beta_vec)
            w = _project_dollar_neutral(w)
        if sector_vec is not None:
            w = _project_sector_neutral(w, sector_vec)
            w = _project_dollar_neutral(w)
        w = _project_box(w, params.w_max)
        if params.gross_max is not None:
            w = _project_gross(w, params.gross_max)

    w_out = pd.Series(w, index=mu.index, name="w_opt")

    if not return_diagnostics:
        return w_out

    diag = {
        "n_assets": int(n),
        "iters": int(it + 1),
        "max_abs_w": float(np.max(np.abs(w))),
        "gross": float(np.sum(np.abs(w))),
        "net": float(np.sum(w)),
        "beta_exposure": None if beta_vec is None else float(beta_vec @ w),
        "sector_neutral": bool(sector_vec is not None),
        "turnover_l1": float(np.sum(np.abs(w - w_prev_vec))),
        "shrink_delta": float(params.shrink_delta),
        "lam": float(params.lam),
        "turnover_gamma": float(params.turnover_gamma),
        "w_max": float(params.w_max),
        "gross_max": None if params.gross_max is None else float(params.gross_max),
    }
    return w_out, diag