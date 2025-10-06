import math
import numpy as np
import pandas as pd
from bt import compute_mle_elo
from evaluate import build_bt_design_aggregated


def contrast_basis(p: int) -> np.ndarray:
    """Orthonormal basis Q for sum-to-zero subspace: Q^T 1 = 0, Q^T Q = I."""
    A = np.ones((1, p), dtype=float)
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    return Vt[1:, :].T  # (p, p-1)

def compute_info_state(subset_df: pd.DataFrame, models: list, base=math.e, scale=1.0, ridge=1e-8):
    """
    Reuse your fit + design to compute model-based information inverse in contrast space.
    Returns dict with {models, m2i, Q, Xr, Binv, beta_full, p_hat}.
    """
    # Current BT fit in your (ξ) units
    elo_series = compute_mle_elo(subset_df, SCALE=scale, BASE=base, INIT_RATING=1000)
    models = list(models)
    p = len(models)
    m2i = {m:i for i,m in enumerate(models)}

    # Design + weights (exactly as your fit)
    X, y, w = build_bt_design_aggregated(subset_df, models, base=base)

    # Coefs for linear predictor; any constant shift cancels in X@β, so just divide by scale
    beta_full = elo_series.values / scale

    # Predicted probabilities
    eta = X @ beta_full
    p_hat = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))

    # Sum-to-zero contrast coordinates
    Q = contrast_basis(p)
    Xr = X @ Q

    # Model-based information (Bread) and its inverse
    W_diag = w * p_hat * (1.0 - p_hat)
    Bread = (Xr.T * W_diag) @ Xr
    Binv  = np.linalg.pinv(Bread + ridge * np.eye(Bread.shape[0]))

    return dict(models=models, m2i=m2i, Q=Q, Xr=Xr, Binv=Binv,
                beta_full=beta_full, p_hat=p_hat, base=base, scale=scale)

def rank_pairs_by_expected_var_drop(state):
    """
    For each unordered pair (i,j), compute expected variance reduction of z_ij
    after adding ONE more observation of that pair, using Sherman–Morrison on Bread.
    z_ij = c*(β_i - β_j), with c = log(base)/scale.
    Returns a Series indexed by (model_i, model_j) sorted desc by ΔVar.
    """
    models = state["models"]; p = len(models)
    m2i = state["m2i"]; Q = state["Q"]; Binv = state["Binv"]
    beta_full = state["beta_full"]; base = state["base"]; scale = state["scale"]

    c = math.log(base) / scale
    pair_keys = []
    deltas = []

    for i in range(p):
        for j in range(i+1, p):
            # full-space vectors
            u = np.zeros(p); u[i] = c; u[j] = -c           # gradient for z_ij
            x = np.zeros(p); x[i] = math.log(base); x[j] = -math.log(base)  # design row for (i over j)

            # reduce to contrast space
            u_r = Q.T @ u
            x_r = Q.T @ x

            # expected info α = p*(1-p) for ONE new trial at current prob q
            z = x @ beta_full
            q = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            alpha = q * (1 - q)

            # Sherman–Morrison reduction in Var(z_ij)
            s1 = float(x_r.T @ Binv @ x_r)       # x^T B^{-1} x
            s2 = float(u_r.T @ Binv @ x_r)       # u^T B^{-1} x
            delta = (alpha * s2 * s2) / (1.0 + alpha * s1)

            pair_keys.append((models[i], models[j]))
            deltas.append(delta)

    return pd.Series(deltas,
                     index=pd.MultiIndex.from_tuples(pair_keys, names=["model_i","model_j"])
                    ).sort_values(ascending=False)

def _make_pair_key(a, b):
    return tuple(sorted((a, b)))

def _build_pair_pool(matches_df: pd.DataFrame, rng=None):
    """
    Map unordered pair -> list of available row indices. Optionally shuffle each list.
    """
    pool = {}
    for idx, (ma, mb) in matches_df[['model_a', 'model_b']].iterrows():
        k = _make_pair_key(ma, mb)
        pool.setdefault(k, []).append(idx)
    if rng is not None:
        for k in pool:
            rng.shuffle(pool[k])
    return pool

def build_sampling_order(matches_df: pd.DataFrame,
                         mode="random",
                         base=math.e, scale=1.0,
                         seed=42,
                         warmup_per_pair=1,
                         max_n=None,
                         verbose=True):
    """
    Reuse your fit + design to produce an order of row indices:
      - mode="random": current behavior (head order).
      - mode="active": Arena-style CI-aware sampling (greedy variance reduction).
    """
    rng = np.random.default_rng(seed)
    models = sorted(set(matches_df["model_a"]) | set(matches_df["model_b"]))
    max_n = len(matches_df) if max_n is None else min(max_n, len(matches_df))

    if mode == "random":
        return list(matches_df.index.values[:max_n])

    # ACTIVE:
    pool = _build_pair_pool(matches_df, rng=rng)
    order = []

    # 1) warm-up for connectivity / invertibility
    for _ in range(warmup_per_pair):
        for pair_key, idx_list in list(pool.items()):
            if len(order) >= max_n: break
            if idx_list:
                order.append(idx_list.pop())
        if len(order) >= max_n: break

    subset_df = matches_df.loc[order].copy()

    # 2) greedy selection by expected variance reduction
    while len(order) < max_n:
        if subset_df.empty:
            # take anything if we somehow start empty
            nonempty = [k for k,v in pool.items() if v]
            if not nonempty: break
            order.append(pool[nonempty[0]].pop())
            subset_df = matches_df.loc[order]
            continue

        
        state = compute_info_state(subset_df, models, base=base, scale=scale)
        rank = rank_pairs_by_expected_var_drop(state)
        

        picked = None
        # try top-ranked pairs first
        for (mi, mj), _ in rank.items():
            k = _make_pair_key(mi, mj)
            if k in pool and pool[k]:
                picked = pool[k].pop()
                break

        # fallback: any available pair
        if picked is None:
            nonempty = [k for k,v in pool.items() if v]
            if not nonempty: break
            picked = pool[nonempty[0]].pop()

        order.append(picked)
        subset_df = matches_df.loc[order]

    return order
