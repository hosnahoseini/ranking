import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid
import math
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm


def compute_mle_elo(
    df, SCALE=1, BASE=math.e, INIT_RATING=1000, sample_weight=None
):
    # Ensure all players appear in the matrices to avoid NaNs for missing pairs
    all_models = sorted(set(df["model_a"]).union(set(df["model_b"])) )

    ptbl_a_win = pd.pivot_table(
        df[df["winner"] == "model_a"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    ).reindex(index=all_models, columns=all_models, fill_value=0)
    # if no tie, create a zero matrix
    if sum(df["winner"].isin(["tie", "tie (bothbad)"])) == 0:
        ptbl_tie = pd.DataFrame(0, index=all_models, columns=all_models)
    else:
        ptbl_tie = pd.pivot_table(
            df[df["winner"].isin(["tie", "tie (bothbad)"])],
            index="model_a",
            columns="model_b",
            aggfunc="size",
            fill_value=0,
        ).reindex(index=all_models, columns=all_models, fill_value=0)
        ptbl_tie = ptbl_tie + ptbl_tie.T

    ptbl_b_win = pd.pivot_table(
        df[df["winner"] == "model_b"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    ).reindex(index=all_models, columns=all_models, fill_value=0)
    ptbl_win = ptbl_a_win * 2 + ptbl_b_win.T * 2 + ptbl_tie

    models = pd.Series(np.arange(len(all_models)), index=all_models)

    p = len(models)
    X = np.zeros([p * (p - 1) * 2, p])
    Y = np.zeros(p * (p - 1) * 2)

    cur_row = 0
    sample_weights = []
    for m_a in ptbl_win.index:
        for m_b in ptbl_win.columns:
            if m_a == m_b:
                continue
            w_ab = ptbl_win.loc[m_a, m_b]
            w_ba = ptbl_win.loc[m_b, m_a]
            # skip pairs with zero evidence both ways
            if (w_ab == 0 and w_ba == 0):
                continue
            X[cur_row, models[m_a]] = +math.log(BASE)
            X[cur_row, models[m_b]] = -math.log(BASE)
            Y[cur_row] = 1.0
            sample_weights.append(w_ab)

            X[cur_row + 1, models[m_a]] = math.log(BASE)
            X[cur_row + 1, models[m_b]] = -math.log(BASE)
            Y[cur_row + 1] = 0.0
            sample_weights.append(w_ba)
            cur_row += 2
    X = X[:cur_row]
    Y = Y[:cur_row]

    # Fallback if no rows were generated (e.g., extremely sparse subset)
    if cur_row == 0:
        elo_scores = np.zeros(len(models))
        return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = SCALE * lr.coef_[0] + INIT_RATING
    # if "Player_0" in models.index:
    #     elo_scores += 1114 - elo_scores[models["Player_0"]]
    elo_scores -= elo_scores.mean() 
    elo_scores += 0.5

    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)

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

