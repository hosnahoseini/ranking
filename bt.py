import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid
import math
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

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


def compute_mle_elo(df, SCALE=1.0, BASE=math.e, INIT_RATING=0.5, sample_weight=None):
    models = sorted(set(df["model_a"]) | set(df["model_b"]))
    X, Y, W = build_bt_design_aggregated(df, models, base=BASE)

    # Guard: not enough information yet
    if X.shape[0] == 0 or float(W.sum()) == 0:
        raise ValueError("Insufficient pairwise data for BT fit (empty design).")

    # Increase iterations and set solver to ensure convergence on large datasets
    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6, max_iter=2000, solver="lbfgs")
    lr.fit(X, Y, sample_weight=W)

    # Work in ξ units (your current convention)
    xi = SCALE * lr.coef_[0]
    xi -= xi.mean()
    xi += INIT_RATING  # here INIT_RATING = 0.5 to center in [0,1] world
    return pd.Series(xi, index=models).sort_values(ascending=False)

# def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000):
#     models = pd.concat([df["model_a"], df["model_b"]]).unique()
#     models = pd.Series(np.arange(len(models)), index=models)

#     # duplicate battles
#     df = pd.concat([df, df], ignore_index=True)
#     p = len(models.index)
#     n = df.shape[0]

#     X = np.zeros([n, p])
#     X[np.arange(n), models[df["model_a"]]] = +math.log(BASE)
#     X[np.arange(n), models[df["model_b"]]] = -math.log(BASE)

#     # one A win => two A win
#     Y = np.zeros(n)
#     Y[df["winner"] == "model_a"] = 1.0

#     # one tie => one A win + one B win
#     # find tie + tie (both bad) index
#     tie_idx = (df["winner"] == "tie") | (df["winner"] == "tie (bothbad)")
#     tie_idx[len(tie_idx)//2:] = False
#     Y[tie_idx] = 1.0

#     lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-8)
#     lr.fit(X,Y)

#     elo_scores = SCALE * lr.coef_[0] + INIT_RATING

#     if "Player_0" in models.index:
#         elo_scores += 1000 - elo_scores[models["Player_0"]]
#         # Mean-center ratings so their average is 1000
#         # elo_scores -= elo_scores.mean()
#         # elo_scores += 1000

#     return pd.Series(elo_scores, index = models.index).sort_values(ascending=False)


