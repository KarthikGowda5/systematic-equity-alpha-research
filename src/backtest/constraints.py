import numpy as np
import pandas as pd


def enforce_exposure_targets(
    w: pd.Series,
    exposures: pd.DataFrame,
    targets: pd.Series | None = None,
    *,
    require_all_exposures: bool = True,
    fill_missing_exposures: float | None = None,
    missing_weight: float = 0.0,
    ridge: float = 1e-10,
) -> pd.Series:
    """
    Enforce linear equality constraints:

        exposures.T @ w = targets

    This is a generic "constraint projector" intended to be reused for many constraints
    (beta neutrality, sector neutrality, size neutrality, etc.).

    Parameters
    ----------
    w : pd.Series
        Candidate weights indexed by asset.
    exposures : pd.DataFrame
        Index = asset, columns = constraint names (K constraints).
    targets : pd.Series, optional
        Target exposure for each constraint column. Default = 0 for all constraints.
    require_all_exposures : bool
        If True, only keep assets that have non-missing exposures for ALL constraints.
        If False, you must provide fill_missing_exposures (to fill NaNs) or accept NaNs (not recommended).
    fill_missing_exposures : float, optional
        If provided, fills missing exposure values with this constant before projecting.
        Useful if you prefer not to drop assets with missing exposures.
    missing_weight : float
        Weight assigned to assets that are dropped due to missing exposures (default 0.0).
    ridge : float
        Numerical ridge passed to the projection solver.

    Returns
    -------
    pd.Series
        Weights projected to satisfy the exposure targets.
    """
    exp = exposures.reindex(w.index)

    if fill_missing_exposures is not None:
        exp = exp.fillna(float(fill_missing_exposures))
        valid = pd.Series(True, index=exp.index)
    else:
        if require_all_exposures:
            valid = exp.notna().all(axis=1)
        else:
            # If not requiring all exposures, any remaining NaNs will break the linear system.
            # Force users to be explicit.
            raise ValueError(
                "require_all_exposures=False requires fill_missing_exposures to be set "
                "to avoid NaNs in exposures."
            )

    w_valid = w.loc[valid].copy()
    exp_valid = exp.loc[valid].copy()

    if len(w_valid) == 0:
        out = w.copy()
        out[:] = missing_weight
        return out

    w_projected = project_linear_constraints(
        w_valid,
        exp_valid,
        targets=targets,
        ridge=ridge,
    )

    out = w.copy()
    out.loc[valid] = w_projected
    out.loc[~valid] = missing_weight
    return out


def project_linear_constraints(
    w: pd.Series,
    exposures: pd.DataFrame,
    targets: pd.Series | None = None,
    ridge: float = 1e-10,
) -> pd.Series:
    """
    Project weights onto linear equality constraints:

        exposures.T @ w = targets

    Parameters
    ----------
    w : pd.Series
        Candidate weights indexed by asset (e.g., permno).
    exposures : pd.DataFrame
        Index = asset, columns = constraint names (e.g., ["ones", "beta"]).
    targets : pd.Series, optional
        Target exposure for each constraint column. Default = 0 for all.
    ridge : float
        Small ridge term added to (A A') for numerical stability.

    Returns
    -------
    pd.Series
        Projected weights satisfying linear equality constraints.
    """
    idx = w.index.intersection(exposures.index)
    if len(idx) == 0:
        return w

    w0 = w.loc[idx].astype(float).values  # (N,)
    A = exposures.loc[idx].astype(float).values.T  # (K, N)

    K = A.shape[0]

    if targets is None:
        b = np.zeros(K)
    else:
        b = (
            targets.reindex(exposures.columns)
            .fillna(0.0)
            .values.astype(float)
        )

    # Solve: (A A') lambda = (A w0 - b)
    AAT = A @ A.T
    AAT = AAT + ridge * np.eye(K)

    rhs = (A @ w0) - b
    lam = np.linalg.solve(AAT, rhs)

    w_proj = w0 - (A.T @ lam)

    out = w.copy()
    out.loc[idx] = w_proj
    return out


def enforce_beta_neutrality(
    w: pd.Series,
    beta: pd.Series,
    dollar_neutral: bool = True,
    *,
    missing_weight: float = 0.0,
    ridge: float = 1e-10,
) -> pd.Series:
    """
    Convenience wrapper for beta neutrality:

        sum_i w_i * beta_i = 0

    Optionally also enforce dollar neutrality:

        sum_i w_i = 0

    This calls `enforce_exposure_targets` under the hood so that adding additional
    constraints later is just a matter of supplying more exposure columns.

    Parameters
    ----------
    w : pd.Series
        Candidate weights indexed by asset.
    beta : pd.Series
        Estimated betas indexed by asset.
    dollar_neutral : bool
        If True, also enforce sum(w) = 0.
    missing_weight : float
        Weight assigned to assets with missing beta (default 0.0).
    ridge : float
        Numerical ridge passed to the projection solver.

    Returns
    -------
    pd.Series
        Beta-neutral (and optionally dollar-neutral) weights.
    """
    exposures = pd.DataFrame(index=w.index)

    if dollar_neutral:
        exposures["ones"] = 1.0

    exposures["beta"] = beta.reindex(w.index)

    targets = pd.Series(0.0, index=exposures.columns)

    return enforce_exposure_targets(
        w=w,
        exposures=exposures,
        targets=targets,
        require_all_exposures=True,
        fill_missing_exposures=None,
        missing_weight=missing_weight,
        ridge=ridge,
    )