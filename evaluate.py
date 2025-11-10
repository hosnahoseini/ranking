from scipy.stats import spearmanr, kendalltau
import numpy as np
import pandas as pd
import math
from bt import build_bt_design_aggregated

def evaluate_ranking_correlation(true_skills, predicted_ratings):
    common = true_skills.index.intersection(predicted_ratings.index)
    s, sp = spearmanr(true_skills[common], predicted_ratings[common])
    k, kp = kendalltau(true_skills[common], predicted_ratings[common])
    return {
        'spearman_correlation': s, 'spearman_p_value': sp,
        'kendall_correlation': k, 'kendall_p_value': kp
    }

def create_win_matrix(matches_df):
    players = sorted(set(matches_df['model_a']) | set(matches_df['model_b']))
    win_matrix = pd.DataFrame(0, index=players, columns=players)
    for _, r in matches_df.iterrows():
        a, b, w = r['model_a'], r['model_b'], r['winner']
        if w == 'model_a':
            win_matrix.loc[a, b] += 1
        elif w == 'model_b':
            win_matrix.loc[b, a] += 1
    return win_matrix

# --- Sandwich CI, sum-to-zero, returns CI df (compute once per step) ---
# x = (all possible matches (n 2), )
# eta and w works
# write down example 

def compute_sandwich_ci(matches_df, elo_series, scale=1.0, base=math.e, ridge=1e-8):
    models = list(elo_series.index)
    X, y, w = build_bt_design_aggregated(matches_df, models, base=base)
    p = X.shape[1]

    # xi = scale*beta + 0.5  =>  beta = (xi - 0.5)/scale
    beta_full = (elo_series.values - 0.5) / scale

    eta = X @ beta_full
    p_hat = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))

    # contrast basis Q (sum-to-zero)
    A = np.ones((1, p), dtype=float)
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    Q = Vt[1:, :].T
    Xr = X @ Q

    # Bread: Xr^T diag(w p(1-p)) Xr
    W_diag = w * p_hat * (1.0 - p_hat)
    Bread = (Xr.T * W_diag) @ Xr

    # Meat: Xr^T diag(w * r^2) Xr
    r = y - p_hat
    Meat = (Xr.T * (w * (r**2))) @ Xr

    Binv = np.linalg.pinv(Bread + ridge * np.eye(Bread.shape[0]))
    Cov_r = Binv @ Meat @ Binv
    Cov_beta = Q @ Cov_r @ Q.T  # rank p-1 (expected)

    se_beta = np.sqrt(np.clip(np.diag(Cov_beta), 0.0, None))
    se_elo = se_beta * scale

    z = 1.96
    elo = elo_series.values
    lower = elo - z * se_elo
    upper = elo + z * se_elo
    out = pd.DataFrame({"elo": elo, "se_elo": se_elo, "lower": lower, "upper": upper}, index=models)
    active_info_state = {
        "models": models,
        "elo": elo,
        "Q": Q,
        "Bread": Bread,
        "Binv": Binv,
        "Cov_beta": Cov_beta,
        "p_hat": p_hat,
        "X": X,
        "y": y,
        "w": w
    }
    return out.loc[elo_series.index], active_info_state

# --- Bootstrap CI (unchanged) ---
def compute_bootstrap_ci(matches_df, compute_fn, rounds=200, alpha=0.05, random_state=42):
    rng = np.random.default_rng(random_state)
    all_models = sorted(set(matches_df["model_a"])|set(matches_df["model_b"]))
    samples = []
    for _ in range(rounds):
        boot = matches_df.sample(frac=1.0, replace=True,
                                 random_state=int(rng.integers(0, 1e9)))
        s = compute_fn(boot).reindex(all_models)
        samples.append(s)
    S = pd.DataFrame(samples, columns=all_models)
    Qq = S.quantile([alpha/2, 0.5, 1-alpha/2])
    return pd.DataFrame({"lower": Qq.loc[alpha/2], "rating": Qq.loc[0.5], "upper": Qq.loc[1-alpha/2]})
    

def compute_log_likelihood(matches_df, elo_series, base=math.e) -> float:
    """
    Compute the (weighted) log-likelihood of the BT logistic model given matches_df and elo ratings.
    Uses the aggregated design to align with training.
    """
    models = list(elo_series.index)
    X, y, w = build_bt_design_aggregated(matches_df, models, base=base)
    # elo = scale*beta + 0.5 with scale=1.0 in current convention => beta = elo - 0.5
    beta_full = (elo_series.reindex(models).values - 0.5)
    eta = X @ beta_full
    # Stabilize probabilities
    eta = np.clip(eta, -30, 30)
    p = 1.0 / (1.0 + np.exp(-eta))
    eps = 1e-12
    logp = np.log(np.clip(p, eps, 1 - eps))
    log1mp = np.log(np.clip(1.0 - p, eps, 1 - eps))
    ll = float(np.sum(w * (y * logp + (1.0 - y) * log1mp)))
    return ll


def compute_elo_add_reward(
    elo_series: pd.Series,
    model_a: str,
    model_b: str,
    target_model: str,
    K: float = 1.0,
    SCALE: float = 1.0,
    BASE: float = math.e,
) -> float:
    """
    Hypothetical reward if we add one match where model_a wins over model_b.
    Using the user's provided update logic on Elo-like ratings (elo_series):
      - ra, rb from elo_series
      - ea, eb computed from BASE and SCALE
      - update ra1, rb1 with step size K and eb
      - reward = P(target loses to updated A) + P(target loses to updated B)
                = 1/(1+BASE^((ra1-rt)/SCALE)) + 1/(1+BASE^((rb1-rt)/SCALE))
    """
    initial_rating = elo_series.to_dict()
    if model_a not in initial_rating or model_b not in initial_rating or target_model not in initial_rating:
        return float('nan')
    ra = float(initial_rating[model_a])
    rb = float(initial_rating[model_b])
    rt = float(initial_rating[target_model])

    ea = 1.0 / (1.0 + (BASE ** ((rb - ra) / SCALE)))
    eb = 1.0 / (1.0 + (BASE ** ((ra - rb) / SCALE)))

    ra1 = ra + K * eb
    rb1 = rb - K * eb

    term_a = 1.0 / (1.0 + (BASE ** ((ra1 - rt) / SCALE)))
    term_b = 1.0 / (1.0 + (BASE ** ((rb1 - rt) / SCALE)))
    reward = float(term_a + term_b)
    return reward


def compute_elo_add_reward_actual(
    elo_series: pd.Series,
    model_a: str,
    model_b: str,
    winner: str,
    target_model: str,
    K: float = 1.0,
    SCALE: float = 1.0,
    BASE: float = math.e,
) -> float:
    """
    Reward using the actual match outcome in `winner`:
      - winner in {"model_a", "model_b", "tie", "tie (bothbad)"}
      - Updates (ra, rb) according to the outcome, then computes:
        1/(1+BASE^((ra1-rt)/SCALE)) + 1/(1+BASE^((rb1-rt)/SCALE))
      - Ties default to no rating change (ra1=ra, rb1=rb)
    """
    initial_rating = elo_series.to_dict()
    if model_a not in initial_rating or model_b not in initial_rating or target_model not in initial_rating:
        return float('nan')
    ra = float(initial_rating[model_a])
    rb = float(initial_rating[model_b])
    rt = float(initial_rating[target_model])

    ea = 1.0 / (1.0 + (BASE ** ((rb - ra) / SCALE)))
    eb = 1.0 / (1.0 + (BASE ** ((ra - rb) / SCALE)))

    w = str(winner).lower().strip()
    if w in ("model_a", "a"):
        ra1 = ra + K * eb
        rb1 = rb - K * eb
    elif w in ("model_b", "b"):
        rb1 = rb + K * ea
        ra1 = ra - K * ea
    elif w in ("tie", "tie (bothbad)"):
        ra1, rb1 = ra, rb
    else:
        return float('nan')

    term_a = 1.0 / (1.0 + (BASE ** ((ra1 - rt) / SCALE)))
    term_b = 1.0 / (1.0 + (BASE ** ((rb1 - rt) / SCALE)))
    return float(term_a + term_b)


def compute_reward_per_match(matches_df: pd.DataFrame, elo_series: pd.Series, target_player: str | None) -> pd.Series:
    vals = []
    for _, r in matches_df.iterrows():
        a = r['model_a']; b = r['model_b']; w = r['winner']
        if target_player is not None:
            tgt = target_player
            val = compute_elo_add_reward_actual(elo_series, a, b, w, tgt)
        else:
            if w == 'model_a':
                tgt = a
                val = compute_elo_add_reward_actual(elo_series, a, b, w, tgt)
            elif w == 'model_b':
                tgt = b
                val = compute_elo_add_reward_actual(elo_series, a, b, w, tgt)
            else:
                r1 = compute_elo_add_reward_actual(elo_series, a, b, w, a)
                r2 = compute_elo_add_reward_actual(elo_series, a, b, w, b)
                val = float(np.nanmean([r1, r2]))
        vals.append(val)
    return pd.Series(vals, index=matches_df.index, name='reward')


def compute_ci_pair_scores(current_df: pd.DataFrame, elo_series: pd.Series) -> dict:
    """
    Pair-level 'uncertainty' score based on CI geometry:
      score(a,b) = sqrt(v_ij)                              if N_ij == 0
                 = sqrt(v_ij) * (1/sqrt(N_ij) - 1/sqrt(N_ij+1)) otherwise
      where v_ij = Var(beta_i - beta_j) from Cov_beta (sandwich),
            N_ij = number of existing matches between unordered pair {a,b} in current_df.
    Returns a dict keyed by ordered tuple (min(a,b), max(a,b)) -> float score.
    """
    # CI info under current dataset/state
    _, info = compute_sandwich_ci(current_df, elo_series, base=math.e)
    Cov = info["Cov_beta"]
    models = info["models"]
    m2i = {m: i for i, m in enumerate(models)}

    # Unordered pair counts from current_df
    def _pair_key(x, y):
        return (x, y) if str(x) <= str(y) else (y, x)
    pair_counts = {}
    for _, r in current_df.iterrows():
        a, b = r["model_a"], r["model_b"]
        key = _pair_key(a, b)
        pair_counts[key] = pair_counts.get(key, 0) + 1

    pair_scores = {}
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            if i >= j:
                continue
            v_ij = float(Cov[i, i] + Cov[j, j] - 2.0 * Cov[i, j])
            v_ij = max(v_ij, 1e-12)
            key = _pair_key(a, b)
            N_ij = int(pair_counts.get(key, 0))
            if N_ij <= 0:
                score = float(np.sqrt(v_ij))
            else:
                score = float(np.sqrt(v_ij) * ((1.0 / np.sqrt(N_ij)) - (1.0 / np.sqrt(N_ij + 1.0))))
            pair_scores[key] = score
    return pair_scores


def compute_ci_uncertainty_per_match(matches_df: pd.DataFrame, elo_series: pd.Series) -> pd.Series:
    """
    Map the CI-based pair scores (under matches_df) to each match row.
    All matches of the same unordered pair share the same score.
    """
    pair_scores = compute_ci_pair_scores(matches_df, elo_series)
    def _pair_key(x, y):
        return (x, y) if str(x) <= str(y) else (y, x)
    vals = []
    for _, r in matches_df.iterrows():
        a, b = r["model_a"], r["model_b"]
        vals.append(pair_scores.get(_pair_key(a, b), np.nan))
    return pd.Series(vals, index=matches_df.index, name="ci_uncertainty")


def compute_target_variance_drop_pair_scores(
    matches_df: pd.DataFrame,
    elo_series: pd.Series,
    target_player: str,
    base: float = math.e,
    alpha_mode: str = "fixed",
) -> dict:
    """
    Target-specific variance drop score for each unordered pair (mi, mj), inspired by run.py (254-281).
      - Uses compute_sandwich_ci => obtains Q (contrast basis) and Binv (inverse bread)
      - For target 'a': e_full[a]=1 projects to e_r = Q^T e_full
      - For each pair (mi, mj):
          x_full has +log(base) on mi, -log(base) on mj; x_r = Q^T x_full
          s1 = x_r^T Binv x_r
          s2 = e_r^T Binv x_r
          alpha ≈ q*(1-q); we default to 0.25 when alpha_mode='fixed'
          delta_var_a = (alpha * s2^2) / (1 + alpha * s1)
    Returns dict with keys (min(mi,mj), max(mi,mj)) -> delta_var_a (float).
    """
    # CI info under current dataset/state
    _, info = compute_sandwich_ci(matches_df, elo_series, base=base)
    models = info["models"]
    Q = info["Q"]
    Binv = info["Binv"]
    m2i = {m: i for i, m in enumerate(models)}
    if target_player not in m2i:
        return {}

    logb = math.log(base)

    def _pair_key(x, y):
        return (x, y) if str(x) <= str(y) else (y, x)

    # Build e_r for target
    p = len(models)
    e_full = np.zeros(p, dtype=float)
    e_full[m2i[target_player]] = 1.0
    e_r = Q.T @ e_full  # (p-1,)

    # Optional pair counts (to filter to present pairs); use unordered pairs present in df
    present_pairs = set()
    for _, r in matches_df.iterrows():
        present_pairs.add(_pair_key(r["model_a"], r["model_b"]))

    scores = {}
    for i, mi in enumerate(models):
        for j in range(i + 1, len(models)):
            mj = models[j]
            key = _pair_key(mi, mj)
            if key not in present_pairs:
                # skip pairs not present in current dataset
                continue
            x_full = np.zeros(p, dtype=float)
            x_full[m2i[mi]] = +logb
            x_full[m2i[mj]] = -logb
            x_r = Q.T @ x_full
            s1 = float(x_r.T @ Binv @ x_r)
            s2 = float(e_r.T @ Binv @ x_r)
            if alpha_mode == "model":
                # Use current elo to estimate q for pair (mi over mj)
                di = float(elo_series.get(mi, 0.5))
                dj = float(elo_series.get(mj, 0.5))
                eta = np.clip((di - dj), -30, 30)
                q = 1.0 / (1.0 + np.exp(-eta))
                alpha = float(q * (1.0 - q))
            else:
                alpha = 0.25
            delta_var = (alpha * (s2 ** 2.0)) / (1.0 + alpha * s1)
            scores[key] = float(delta_var)
    return scores


def compute_target_variance_drop_per_match(
    matches_df: pd.DataFrame,
    elo_series: pd.Series,
    target_player: str,
    base: float = math.e,
    alpha_mode: str = "fixed",
) -> pd.Series:
    """
    Map target-specific variance drop scores to each match row by its unordered pair.
    """
    pair_scores = compute_target_variance_drop_pair_scores(matches_df, elo_series, target_player, base=base, alpha_mode=alpha_mode)
    def _pair_key(x, y):
        return (x, y) if str(x) <= str(y) else (y, x)
    vals = []
    for _, r in matches_df.iterrows():
        a, b = r["model_a"], r["model_b"]
        vals.append(pair_scores.get(_pair_key(a, b), np.nan))
    return pd.Series(vals, index=matches_df.index, name=f"target_var_drop[{target_player}]")
