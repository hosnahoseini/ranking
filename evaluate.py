from scipy.stats import spearmanr, kendalltau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from tqdm import tqdm
from bt import compute_mle_elo
import math

def evaluate_ranking_correlation(true_skills, predicted_ratings):

    # Get common players
    common_players = true_skills.index.intersection(predicted_ratings.index)
    true_vals = true_skills[common_players]
    pred_vals = predicted_ratings[common_players]
    
    # Calculate correlations
    spearman_corr, spearman_p = spearmanr(true_vals, pred_vals)
    kendall_corr, kendall_p = kendalltau(true_vals, pred_vals)
    
    return {
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p,
        'kendall_correlation': kendall_corr,
        'kendall_p_value': kendall_p
    }

def create_win_matrix(matches_df):
    
    
    # Get unique players
    players = sorted(set(matches_df['model_a'].unique()) | set(matches_df['model_b'].unique()))
    
    # Initialize win matrix
    win_matrix = pd.DataFrame(0, index=players, columns=players)
    
    # Fill in win counts
    for _, row in matches_df.iterrows():
        model_a = row['model_a']
        model_b = row['model_b']
        winner = row['winner']
        
        if winner == 'model_a':
            win_matrix.loc[model_a, model_b] += 1
        elif winner == 'model_b':
            win_matrix.loc[model_b, model_a] += 1
        # Ties are ignored (0 in matrix)
    
    return win_matrix



def build_bt_design_aggregated(matches_df, models, base=10.0):
    """
    Build pair-aggregated BT design:
      - two rows per ordered pair (same x): one with y=1 (A beats B), one with y=0 (B beats A)
      - weights = aggregated counts (wins/ties) exactly as in your compute_mle_elo
      - ties contribute symmetrically via ptbl_tie + ptbl_tie.T
    Returns: X (n x p), y (n,), w (n,)
    """
    models = list(models)
    m2i = {m:i for i,m in enumerate(models)}
    p = len(models)
    logb = math.log(base)

    # Pivot tables (reindexed so all players appear even if zero)
    ptbl_a = pd.pivot_table(
        matches_df[matches_df["winner"] == "model_a"],
        index="model_a", columns="model_b", aggfunc="size", fill_value=0
    ).reindex(index=models, columns=models, fill_value=0)

    ptbl_b = pd.pivot_table(
        matches_df[matches_df["winner"] == "model_b"],
        index="model_a", columns="model_b", aggfunc="size", fill_value=0
    ).reindex(index=models, columns=models, fill_value=0)

    if (matches_df["winner"].isin(["tie", "tie (bothbad)"])).any():
        ptbl_t = pd.pivot_table(
            matches_df[matches_df["winner"].isin(["tie", "tie (bothbad)"])],
            index="model_a", columns="model_b", aggfunc="size", fill_value=0
        ).reindex(index=models, columns=models, fill_value=0)
        ptbl_t = ptbl_t + ptbl_t.T
    else:
        ptbl_t = pd.DataFrame(0, index=models, columns=models)

    # Total "event count" per ordered pair (matches your compute_mle_elo)
    Wmat = ptbl_a * 2 + ptbl_b.T * 2 + ptbl_t

    rows_X, rows_y, rows_w = [], [], []
    for a in models:
        ia = m2i[a]
        for b in models:
            if a == b:
                continue
            w_ab = float(Wmat.loc[a, b])
            w_ba = float(Wmat.loc[b, a])
            if (w_ab == 0 and w_ba == 0) or np.isnan(w_ab) or np.isnan(w_ba):
                continue

            # Same feature vector for both rows (logit is linear in rating diff)
            x = np.zeros(p, dtype=float)
            x[ia] = +logb
            x[m2i[b]] = -logb

            # Row for "A beats B"
            rows_X.append(x)
            rows_y.append(1.0)
            rows_w.append(w_ab)

            # Row for "B beats A"
            rows_X.append(x)
            rows_y.append(0.0)
            rows_w.append(w_ba)

    if not rows_X:
        raise ValueError("Empty design: no pair data after aggregation.")

    X = np.vstack(rows_X)              # (n, p)
    y = np.asarray(rows_y, dtype=float)
    w = np.asarray(rows_w, dtype=float)
    return X, y, w


# ---------- main: sandwich CI with SUM-TO-ZERO (no ref dropped) ----------
def compute_sandwich_ci(matches_df, elo_series, scale=1.0, base=math.e, ridge=1e-8):
    """
    Huber–White (sandwich) robust CIs for Elo/BT with sum-to-zero identifiability.
    - Uses the SAME aggregated design & weights as your fit.
    - Enforces identifiability with a contrast basis Q where Q^T 1 = 0 (sum-to-zero).
    - Returns 95% CIs on Elo scale for all players (no zero-width CI).
    """
    # Ensure order matches the fitted ratings
    models = list(elo_series.index)

    # Build design identical to the fit
    X, y, w = build_bt_design_aggregated(matches_df, models, base=base)
    p = X.shape[1]

    # Parameters on logistic scale: elo = scale*beta + 1000 => beta = (elo-1000)/scale
    beta_full = (elo_series.values + 0.5) / scale

    # Predicted probabilities
    eta = X @ beta_full
    # numerically stable sigmoid
    p_hat = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))

    # -------- sum-to-zero contrast basis Q (p x (p-1)) --------
    # Columns of Q span the subspace orthogonal to ones: Q^T 1 = 0, and Q^T Q = I
    A = np.ones((1, p), dtype=float)
    # SVD of A gives first right-singular vector ~ normalized ones; the rest span the contrast subspace
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    Q = Vt[1:, :].T   # shape: (p, p-1)

    # Project design into contrast coordinates
    Xr = X @ Q          # (n, p-1)

    # -------- Sandwich pieces (with aggregated weights) --------
    # Bread: Xr^T diag(w * p*(1-p)) Xr
    W_diag = w * p_hat * (1.0 - p_hat)
    Bread = (Xr.T * W_diag) @ Xr

    # Meat: Xr^T diag((w * (y - p))^2) Xr
    r = y - p_hat
    Meat = (Xr.T * (w * (r**2))) @ Xr  # = Xr^T diag(w * r^2) Xr


    # Invert (use pinv + tiny ridge for stability)
    Binv = np.linalg.pinv(Bread + ridge * np.eye(Bread.shape[0]))
    Cov_r = Binv @ Meat @ Binv       # covariance in contrast coords (p-1 x p-1)

    # Map back to full p-dim parameter space (rank p-1; singular by construction)
    Cov_full_beta = Q @ Cov_r @ Q.T  # (p x p)

    # Standard errors on Elo scale
    se_beta = np.sqrt(np.clip(np.diag(Cov_full_beta), 0.0, None))
    se_elo  = se_beta * scale

    # 95% CIs
    z = 1.96
    elo = elo_series.values
    lower = elo - z * se_elo
    upper = elo + z * se_elo

    out = pd.DataFrame(
        {"elo": elo, "se_elo": se_elo, "lower": lower, "upper": upper},
        index=models
    )
    return out.loc[elo_series.index]

def compute_bootstrap_ci(matches_df, compute_fn, rounds=200, alpha=0.05, random_state=42):
    rng = np.random.default_rng(random_state)
    all_models = sorted(set(matches_df["model_a"])|set(matches_df["model_b"]))
    samples = []
    for _ in tqdm(range(rounds), desc="bootstrap"):
        boot_df = matches_df.sample(frac=1.0, replace=True,
                                    random_state=int(rng.integers(0, 1e9)))
        s = compute_fn(boot_df).reindex(all_models)
        samples.append(s)
    S = pd.DataFrame(samples, columns=all_models)
    Q = S.quantile([alpha/2, 0.5, 1-alpha/2])
    return pd.DataFrame({"lower": Q.loc[alpha/2], "rating": Q.loc[0.5], "upper": Q.loc[1-alpha/2]})


def ci_learning_curve(matches_df, players_df, step_size=100, min_matches=100, rounds=200, alpha=0.05):
    """
    Build CI curves (sandwich and bootstrap widths) vs number of matches for each model.
    Returns a long-form DataFrame with columns: n_matches, model, method, lower, rating, upper, width
    """
    true_skills = players_df.set_index('player_name')['skill_level']
    results = []
    max_matches = len(matches_df)
    match_counts = range(min_matches, max_matches + 1, step_size)

    for n_matches in tqdm(match_counts, desc="CI curve"):
        subset = matches_df.head(n_matches)
        elo_series = compute_mle_elo(subset)

        # Sandwich CI
        sand = compute_sandwich_ci(subset, elo_series)
        for model, row in sand.iterrows():
            results.append({
                "n_matches": n_matches,
                "model": model,
                "method": "sandwich",
                "lower": row["lower"],
                "rating": row["elo"],
                "upper": row["upper"],
                "width": row["upper"] - row["lower"],
                "true_skill": true_skills.get(model, np.nan),
            })

        # Bootstrap CI
        boot = compute_bootstrap_ci(subset, compute_mle_elo, rounds=rounds, alpha=alpha)
        for model, row in boot.iterrows():
            results.append({
                "n_matches": n_matches,
                "model": model,
                "method": "bootstrap",
                "lower": row["lower"],
                "rating": row["rating"],
                "upper": row["upper"],
                "width": row["upper"] - row["lower"],
                "true_skill": true_skills.get(model, np.nan),
            })

    return pd.DataFrame(results)

