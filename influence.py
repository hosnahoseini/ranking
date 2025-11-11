import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from bt import build_bt_design_aggregated
import math

def run_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    fit_intercept: bool = False,
    penalty: str = None,
    sample_weight: np.ndarray | None = None,
) -> LogisticRegression:
    """
    Fit a logistic regression model.
    """
    model = LogisticRegression(fit_intercept=fit_intercept, penalty=penalty)
    if sample_weight is not None:
        model.fit(X, y, sample_weight=sample_weight)
    else:
        model.fit(X, y)
    return model

def find_closest_matchups(player_scores: np.ndarray, k: int) -> 'list[tuple[int,int,float]]':
    """
    For each top-index t in [0..k-1] and each rest-index r in [k..P-1],
    compute (t, r, player_scores[t] - player_scores[r]) and return as a list.
    """
    P = player_scores.shape[0] + 1
    #breakpoint()
    full_score = np.concatenate((np.array([0]), player_scores))
    asort = np.argsort(full_score)[::-1] # players sorted from big to small

    matchups = []
    for i in range(k):
        for j in range(P-k):
            diff = np.abs(full_score[asort[i]]-full_score[asort[j+k]]).item()
            tm1 = asort[i].item()-1
            tm2 = asort[j+k].item()-1
            if tm1 == -1:
                matchups.append((tm2, None, diff))
            elif tm2 == -1:
                matchups.append((tm1, None, diff))
            else:
                matchups.append((tm1, tm2, diff))

    sorted_matchups = sorted(matchups, key=lambda x: x[2])
    #breakpoint()
    return sorted_matchups


def isRankingRobust(k, alphaN, X, y):
    '''
    Checks if the ranking of the top k players/models is robust to data-dropping.
    Arg: 
        k, int, number of top players to consider. 
        alphaN, int, amount of data willing to drop.
        X, np.ndarray, design matrix.
        y, np.ndarray, response vector.
    Return:
        playerA, playerB: int, indices of players/models.
        new_beta_diff_refit: float, new beta difference.
        indices: list, indices of dropped data.
    '''
    # run logistic regression on X, y
    myAMIP = LogisticAMIP(X, y, fit_intercept=False, penalty=None)
    player_scores = myAMIP.model.coef_[0] # (p,)

    
    close_matchups = find_closest_matchups(player_scores, k)
    for playerA, playerB, diff in close_matchups: # a list of k(p-k) matchups.
        print("testing new matchup: ", playerA, playerB)
        sign_change_amip, sign_change_refit, original_beta_diff, new_beta_diff_amip, new_beta_diff_refit, indices = myAMIP.AMIP_sign_change(alphaN, playerA, playerB)
        if sign_change_refit:
            return playerA, playerB, original_beta_diff, new_beta_diff_refit, indices
    
    return -1, -1, -1, -1, [-1] # when ranking is robust.

class LogisticAMIP():
    def __init__(self, X: np.ndarray, y: np.ndarray, 
                 fit_intercept: bool = False, 
                 penalty: str = None,
                 sample_weight: np.ndarray | None = None
                 ):
        '''
        Class for dealing with AMIP in logistic regression
        Args:
            X: design matrix 
            y: responses, binary
            fit_intercept: bool, whether to fit intercept
            penalty: bool whether to have penalty
            refit: bool, whether to refit when approximating dropping data
        '''
        self.X = X # does X here exclude one of the players?
        self.y = y
        self.fit_intercept = fit_intercept
        self.penalty = penalty
        self.sample_weight = sample_weight
        self.model = run_logistic_regression(X, y, fit_intercept, penalty, sample_weight=sample_weight)
        self.pos_p_hats = self.model.predict_proba(X)[:, 1]

        ### private stuff ###
        self.__IFcache__ = {}
        self.__oneSNcache__ = {}
        self.__n__ = X.shape[0]
        self.__p__ = X.shape[1]
        

        # Per-row variance factor v = w * p(1-p) if weights provided
        if self.sample_weight is not None:
            self.__v__ = self.sample_weight * (self.pos_p_hats * (1 - self.pos_p_hats))
        else:
            self.__v__ = self.pos_p_hats * (1 - self.pos_p_hats) # (n,)

        H = X.T @ (self.__v__[:, None] * X)  # (p, p)
        self.__invH__ = np.linalg.pinv(H)    # safe pseudo-inverse

        self.__resid__ = (y - self.pos_p_hats)  
        self.__Hprod__ = self.X @ self.__invH__
    
    def get_influence_IF(self, dim):
        '''
        get influence approximation with influence function
        Args:
            dim: int the parameter to calculate influence approximation on
        Return:
            res: nd.array, influence approximation for all data points
        '''
        if not (0 <= dim < self.__p__):
            raise IndexError(f"dim must be in [0, {self.__p__}), got {dim}")
        if dim in self.__IFcache__:
            return self.__IFcache__[dim]
        
        invH_col = self.__invH__[:, dim]                     # (p,)
        #    then each row X_i dot d gives a length-n vector
        influence_unscaled = self.__resid__ * (self.X @ invH_col) # (n,)
        self.__IFcache__[dim] = influence_unscaled
        return influence_unscaled
    
    def get_influence_1sN(self, dim):
        '''
        get influence approximation with 1 step Newton
        Args:
            dim: int the parameter to calculate influence approximation on
        Return:
            res: nd.array, influence approximation for all data points
        '''
        if not (0 <= dim < self.__p__):
            raise IndexError(f"dim must be in [0, {self.__p__}), got {dim}")
        if dim in self.__oneSNcache__:
            return self.__oneSNcache__[dim]
        invH_col = self.__invH__[:, dim]                     # (p,)
        #    then each row X_i dot d gives a length-n vector
        influence_unscaled = self.__resid__ * (self.X @ invH_col) # (n,)                         # (n, p)
        h = self.__v__ * np.einsum("ij,ij->i", self.__Hprod__, self.X)  # (n,)
        res = influence_unscaled / (1.0 - h)
        #breakpoint()
        self.__oneSNcache__[dim] = res
        return res


    def AMIP_sign_change(self, alphaN, dim_1, dim_2 = None, 
                         method = "1sN", refit = True):
        '''
        AMIP to detect sign change of a parameter or difference between two parameters
        Arg: alphaN: int amount of data willing to drop
            dim_1, int, first parameter
            dim_2, int, second parameter, if not None, approximate the different between dim_1 and dim_2

        Return:
            change_sign_amip: bool, if amip says there is a sign change
            change_sign_refit: bool, if refit says there is a sign change
            new_beta_diff_amip: predicted new beta, or beta differences by AMIP
            new_beta_diff_refit: new beta or beta differences by refitting
            index
        '''
        if method == "1sN":
            get_influence = self.get_influence_1sN
        elif method == "IF":
            get_influence = self.get_influence_IF
        else:
            raise("method has to be 1sN or IF")
        beta = self.model.coef_[0]
    
        if dim_2 is None: # this is useful when comparing to reference level 0
            beta_i = beta[dim_1]
            influence = -get_influence(dim_1)
            top = np.argsort(influence)
            if beta_i < 0:
                top = top[::-1] # if beta is negative, we want to sort influence score in reverse order.
            change = np.sum(influence[top[:alphaN]])
            new_betai_amip = beta_i + change
            change_sign_amip = np.sign(new_betai_amip) != np.sign(beta_i)
            if refit:
                res = run_logistic_regression(self.X[top[alphaN:,]], 
                                              self.y[top[alphaN:]],
                                              fit_intercept=self.fit_intercept, 
                                              penalty=self.penalty
                                              )
                new_betai_refit = res.coef_[0][dim_1]
                change_sign_refit = np.sign(new_betai_refit) != np.sign(beta_i)
            else:
                new_betai_refit = None
                change_sign_refit = None

            #breakpoint()
            return change_sign_amip, change_sign_refit, beta_i, new_betai_amip, new_betai_refit, top[:alphaN]

        beta_diff = beta[dim_1] - beta[dim_2]

        influence = -(get_influence(dim_1) - get_influence(dim_2))
        top = np.argsort(influence)
        if beta_diff < 0: # if beta is negative, we want the positive part of the influence score
            top = top[::-1]
        #breakpoint()
        change = np.sum(influence[top[:alphaN]])
        new_beta_diff_amip = beta_diff + change
        change_sign_amip = np.sign(new_beta_diff_amip) != np.sign(beta_diff)
        

        if refit:
            res = run_logistic_regression(self.X[top[alphaN:,]], 
                                          self.y[top[alphaN:]],
                                          fit_intercept=self.fit_intercept, 
                                          penalty=self.penalty)
            new_beta_diff_refit = res.coef_[0][dim_1] - res.coef_[0][dim_2]
            change_sign_refit = np.sign(new_beta_diff_refit) != np.sign(beta_diff)
        else:
            new_beta_diff_refit = None
            change_sign_refit = None
        return change_sign_amip, change_sign_refit, beta_diff, new_beta_diff_amip, new_beta_diff_refit, top[:alphaN]


    def get_model(self):
        return self.model

    def get_pos_p_hats(self):
        return self.pos_p_hats



def compute_leverage(
    v_vec: np.ndarray,
    X: np.ndarray,
    index: int,
) -> float:
    """
    v_vec: np.array, shape (n,), the per-row variance multiplier (optionally weighted).
    X: np.array, shape (n, p), the design matrix.
    index: int, the index of the data point whose influence we want to compute.

    Compute the leverage of the index-th data point.
    """
    V = np.diag(v_vec)
    H = V @ X @ np.linalg.pinv(X.T @ V @ X) @ X.T
    return H[index, index]

def compute_influence_leverage(matches_df: pd.DataFrame, elo_series: pd.Series) -> pd.Series:
    """
    Compute a leverage-like influence per match using LogisticAMIP and compute_leverage,
    while handling ties explicitly by averaging the influence from both outcomes.

    Steps:
        1. Filter out tie matches for model fitting.
        2. Compute leverage for non-tie matches (model_a or model_b wins).
        3. Map leverage values back to matches_df.
        4. For ties, average both outcomes' leverage values.
    """
    # --- Step 1: Split matches into tie and non-tie ---
    non_tie_df = matches_df[matches_df["winner"].isin(["model_a", "model_b"])].copy()
    tie_df = matches_df[matches_df["winner"] == "tie"].copy()

    # --- Step 2: Build BT design for non-tie matches ---
    models_order = list(elo_series.index)
    # Guard: if there are no non-tie rows, return NaNs aligned to matches_df
    if non_tie_df.empty:
        return pd.Series(np.nan, index=matches_df.index, name="influence_leverage")
    try:
        X, y, w = build_bt_design_aggregated(non_tie_df, models_order, base=math.e)
    except Exception:
        # If aggregated design cannot be built (e.g., no pair data), return NaNs
        return pd.Series(np.nan, index=matches_df.index, name="influence_leverage")

    # --- Step 3: Fit logistic regression using AMIP ---
    amip = LogisticAMIP(X, y, fit_intercept=False, penalty=None, sample_weight=w)
    pos_p_hats = amip.get_pos_p_hats()

    # --- Step 4: Compute leverage per aggregated non-tie row ---
    v_vec = (w * (pos_p_hats * (1 - pos_p_hats))).astype(float)
    leverages = np.array([compute_leverage(v_vec, X, i) for i in range(X.shape[0])])

    # --- Step 5: Map aggregated leverages back to non-tie matches robustly ---
    # Build label list for each aggregated row: (a,b,'model_a') for y=1; (a,b,'model_b') for y=0
    lbls = []
    for i in range(X.shape[0]):
        row = X[i]
        pos = np.where(np.abs(row) > 0)[0]
        if len(pos) == 2:
            ia, ib = pos[0], pos[1]
            va = row[ia]
            a = models_order[ia] if va > 0 else models_order[ib]
            b = models_order[ib] if va > 0 else models_order[ia]
            outcome = 'model_a' if y[i] == 1.0 else 'model_b'
            lbls.append((a, b, outcome))
        else:
            lbls.append((None, None, None))
    infl_map = {}
    for i, key in enumerate(lbls):
        infl_map[key] = float(leverages[i])

    # Assign per non-tie match directly via map
    non_tie_df["influence_leverage"] = np.nan
    for idx, r in non_tie_df.iterrows():
        a, b, w = r["model_a"], r["model_b"], r["winner"]
        non_tie_df.at[idx, "influence_leverage"] = infl_map.get((a, b, w), np.nan)

    # --- Step 6: Handle tie matches (average both outcomes) ---
    if not tie_df.empty:
        tie_df["influence_leverage"] = np.nan
        for i, row in tie_df.iterrows():
            a, b = row["model_a"], row["model_b"]
            # Get influence when model_a wins
            val_a = non_tie_df.loc[
                (non_tie_df["model_a"] == a)
                & (non_tie_df["model_b"] == b)
                & (non_tie_df["winner"] == "model_a"),
                "influence_leverage",
            ]
            # Get influence when model_b wins
            val_b = non_tie_df.loc[
                (non_tie_df["model_a"] == a)
                & (non_tie_df["model_b"] == b)
                & (non_tie_df["winner"] == "model_b"),
                "influence_leverage",
            ]
            # Average both if available, avoid RuntimeWarning when both empty
            if not val_a.empty and not val_b.empty:
                avg_val = float((val_a.mean() + val_b.mean()) / 2.0)
            elif not val_a.empty:
                avg_val = float(val_a.mean())
            elif not val_b.empty:
                avg_val = float(val_b.mean())
            else:
                avg_val = np.nan
            tie_df.at[i, "influence_leverage"] = avg_val

    # --- Step 7: Merge back to full matches_df order ---
    merged_df = pd.concat([non_tie_df, tie_df]).sort_index()
    return merged_df["influence_leverage"]
