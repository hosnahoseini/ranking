import os
import math
import argparse
import numpy as np
import pandas as pd

from dataset import generate_arena_dataset
from bt import compute_mle_elo
from bt import build_bt_design_aggregated
from evaluate import (
    evaluate_ranking_correlation,
    compute_sandwich_ci,
    compute_log_likelihood,
    compute_elo_add_reward_actual,
    compute_ci_pair_scores,
    compute_reward_per_match,
    compute_ci_uncertainty_per_match,
    compute_target_variance_drop_per_match,
)
from ops import (
    RandomStrategy,
    add_matches_random,
    remove_matches_random,
    flip_matches_random,
)
from influence import compute_influence_leverage
from plot import (
    plot_ops_ranking_cubes,
    plot_ops_metrics,
    plot_ops_ci_widths,
    plot_ops_events,
    plot_ops_reward_over_steps,
    plot_ops_influence_over_steps,
)

def _normalize_selection(selection: str) -> str:
    """Normalize selection aliases."""
    s = str(selection or "random").strip().lower()
    if s == "uncertainty":
        return "ci_uncertainty"
    return s

def _select_indices_for_operation(
    operation: str,
    selection: str,
    matches_df: pd.DataFrame,
    chosen_indices: np.ndarray,
    current_df: pd.DataFrame,
    elo_curr: pd.Series,
    batch_size: int,
    rng: np.random.Generator,
    target_player: str ,
) -> np.ndarray:
    """Shared selector used by both experiment runners."""
    selection = _normalize_selection(selection)
    k_eff = max(1, int(batch_size))
    if operation == "add":
        chosen_set = set(map(int, chosen_indices.tolist()))
        pool = np.array([i for i in matches_df.index.values if i not in chosen_set], dtype=int)
        if pool.size == 0:
            return np.array([], dtype=int)
        if selection == "random" or elo_curr is None:
            k_take = int(min(k_eff, pool.size))
            return rng.choice(pool, size=k_take, replace=False).astype(int)
        sub = matches_df.loc[pool]
        if selection == "ci_uncertainty":
            scores_ser = compute_ci_uncertainty_per_match(sub, elo_curr)
            scores = scores_ser.values.astype(float)
        elif selection == "influence":
            scores_ser = compute_influence_leverage(sub, elo_curr)
            scores = scores_ser.values.astype(float)
        elif selection == "reward":
            scores_ser = compute_reward_per_match(sub, elo_curr, target_player=None)
            scores = scores_ser.values.astype(float)
        elif selection == "target_var_drop":
            if target_player is None:
                try:
                    target_name = pd.Series(elo_curr).sort_values(ascending=False).index[0]
                except Exception:
                    target_name = None
            else:
                target_name = target_player
            if target_name is None:
                scores = np.full(sub.shape[0], np.nan, dtype=float)
            else:
                scores_ser = compute_target_variance_drop_per_match(sub, elo_curr, target_name)
                scores = scores_ser.values.astype(float)
        else:
            scores = np.full(sub.shape[0], np.nan, dtype=float)
        if np.all(np.isnan(scores)):
            k_take = int(min(k_eff, pool.size))
            return rng.choice(pool, size=k_take, replace=False).astype(int)
        order = np.argsort(-np.nan_to_num(scores, nan=-np.inf))
        k_take = int(min(k_eff, pool.size))
        return pool[order[:k_take]].astype(int)

    elif operation == "remove":
        if chosen_indices.size == 0:
            return np.array([], dtype=int)
        if selection == "random" or elo_curr is None:
            k_take = int(min(k_eff, chosen_indices.size))
            return rng.choice(chosen_indices, size=k_take, replace=False).astype(int)
        sub = current_df.loc[chosen_indices]
        if selection == "influence":
            vals_ser = compute_influence_leverage(sub, elo_curr)
            vals = vals_ser.values.astype(float)
        elif selection == "ci_uncertainty":
            vals_ser = compute_ci_uncertainty_per_match(sub, elo_curr)
            vals = vals_ser.values.astype(float)
        elif selection == "reward":
            vals_ser = compute_reward_per_match(sub, elo_curr, target_player=None)
            vals = vals_ser.values.astype(float)
        elif selection == "target_var_drop":
            if target_player is None:
                try:
                    target_name = pd.Series(elo_curr).sort_values(ascending=False).index[0]
                except Exception:
                    target_name = None
            else:
                target_name = target_player
            if target_name is None:
                vals = np.full(sub.shape[0], np.nan, dtype=float)
            else:
                vals_ser = compute_target_variance_drop_per_match(sub, elo_curr, target_name)
                vals = vals_ser.values.astype(float)
        else:
            vals = np.full(sub.shape[0], np.nan, dtype=float)
        if np.all(np.isnan(vals)):
            k_take = int(min(k_eff, chosen_indices.size))
            return rng.choice(chosen_indices, size=k_take, replace=False).astype(int)
        order = np.argsort(-np.nan_to_num(vals, nan=-np.inf))
        k_take = int(min(k_eff, chosen_indices.size))
        return chosen_indices[order[:k_take]].astype(int)

    else:  # operation == "flip"
        eligible_list = []
        for idx in chosen_indices:
            w = current_df.loc[int(idx), "winner"]
            if str(w) in ["model_a", "model_b"]:
                eligible_list.append(int(idx))
        eligible = np.array(eligible_list, dtype=int)
        if eligible.size == 0:
            return np.array([], dtype=int)
        if selection == "random" or elo_curr is None:
            k_take = int(min(k_eff, eligible.size))
            return rng.choice(eligible, size=k_take, replace=False).astype(int)
        sub = current_df.loc[eligible]
        if selection == "influence":
            vals_ser = compute_influence_leverage(sub, elo_curr)
            vals = vals_ser.values.astype(float)
        elif selection == "ci_uncertainty":
            vals_ser = compute_ci_uncertainty_per_match(sub, elo_curr)
            vals = vals_ser.values.astype(float)
        elif selection == "reward":
            vals_ser = compute_reward_per_match(sub, elo_curr, target_player=None)
            vals = vals_ser.values.astype(float)
        elif selection == "target_var_drop":
            if target_player is None:
                try:
                    target_name = pd.Series(elo_curr).sort_values(ascending=False).index[0]
                except Exception:
                    target_name = None
            else:
                target_name = target_player
            if target_name is None:
                vals = np.full(sub.shape[0], np.nan, dtype=float)
            else:
                vals_ser = compute_target_variance_drop_per_match(sub, elo_curr, target_name)
                vals = vals_ser.values.astype(float)
        else:
            vals = np.full(sub.shape[0], np.nan, dtype=float)
        if np.all(np.isnan(vals)):
            k_take = int(min(k_eff, eligible.size))
            return rng.choice(eligible, size=k_take, replace=False).astype(int)
        order = np.argsort(-np.nan_to_num(vals, nan=-np.inf))
        k_take = int(min(k_eff, eligible.size))
        return eligible[order[:k_take]].astype(int)

def _compute_influence_map(current_df: pd.DataFrame, elo_series: pd.Series) -> dict:
    """
    Compute an influence/leverage proxy for each aggregated pair row and map it to
    concrete match outcomes. Returned keys are (model_a, model_b, outcome_label)
    where outcome_label is in {'model_a','model_b'} meaning A beats B or B beats A.
    """
    try:
        models_order = list(elo_series.index)
        X_aggr, y_aggr, w_aggr = build_bt_design_aggregated(current_df, models_order, base=math.e)
        # Compute leverage
        eta = X_aggr @ ((pd.Series(elo_series).reindex(models_order).values - 0.5))
        eta = np.clip(eta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        v = p * (1.0 - p)
        XtVX = X_aggr.T @ (v[:, None] * X_aggr)
        inv = np.linalg.pinv(XtVX)
        Hprod = X_aggr @ inv
        leverage = v * np.einsum("ij,ij->i", Hprod, X_aggr)
        # Map each aggregated row to an ordered pair and an outcome label
        lbls = []
        for i in range(X_aggr.shape[0]):
            row = X_aggr[i]
            pos = np.where(np.abs(row) > 0)[0]
            if len(pos) == 2:
                ia, ib = pos[0], pos[1]
                va = row[ia]
                a = models_order[ia] if va > 0 else models_order[ib]
                b = models_order[ib] if va > 0 else models_order[ia]
                outcome = 'model_a' if y_aggr[i] == 1.0 else 'model_b'
                lbls.append((a, b, outcome))
            else:
                lbls.append((None, None, None))
        infl_map = {}
        for i, key in enumerate(lbls):
            infl_map.setdefault(key, float(leverage[i]))
        return infl_map
    except Exception:
        return {}


def _ranking_series(elo_series: pd.Series) -> pd.Series:
    """Return ranks (1=best) as a Series indexed by model names."""
    s = pd.Series(elo_series).astype(float)
    return s.rank(ascending=False, method='average')


def _order_list(elo_series: pd.Series) -> list:
    """Sorted model order by rating (desc)."""
    s = pd.Series(elo_series).astype(float)
    return list(s.sort_values(ascending=False).index)


def _min_actions_to_change(
    seed: int,
    num_players: int,
    n_matches: int,
    operation: str,
    selection: str = "random",      # 'random' | 'influence' | 'uncertainty'
    change_mode: str = "any",       # 'any' | 'player'
    target_player: str = None,
    max_actions: int = 500,
    batch_size: int = 1,
    matches_df: pd.DataFrame | None = None,
    players_df: pd.DataFrame | None = None,
) -> dict:
    """
    Return dict with number of actions needed to change ranking according to change_mode.
    - selection:
        'random': sample uniformly without replacement (never act on the same match twice)
        'influence': for flip/remove, choose match with largest leverage proxy;
                     for add, fallback to 'uncertainty'
        'uncertainty': choose match with largest p*(1-p) under current ELO (closest to 0.5)
    - change_mode:
        'any': any change in global ranking order vs baseline
        'player': specific player's rank changes vs baseline (needs target_player)
    """
    assert operation in {"add", "remove", "flip"}
    # support aliases

    selection = _normalize_selection(selection)
    assert selection in {"random", "influence", "ci_uncertainty", "reward", "target_var_drop"}
    assert change_mode in {"any", "player"}

    if matches_df is None or players_df is None:
        players_df, matches_df = generate_arena_dataset(
            num_players=num_players, n_matches=n_matches, gamma=2, seed=seed, allow_ties=True
        )

    rng = np.random.default_rng(seed)

    # Initialize chosen set and current df
    if operation == "add":
        init_n = min(1, len(matches_df))  # start with at most 1 row for a well-defined baseline
        init_idx = rng.choice(matches_df.index.values, size=init_n, replace=False) if len(matches_df) > 0 else np.array([], dtype=int)
        chosen_indices = np.array(sorted(init_idx), dtype=int)
        current_df = matches_df.loc[chosen_indices].copy() if chosen_indices.size > 0 else matches_df.iloc[:0].copy()
    else:
        chosen_indices = matches_df.index.values.astype(int)
        current_df = matches_df.copy()

    # Baseline ELO and rankings
    try:
        elo0 = compute_mle_elo(current_df)
        rank0 = _ranking_series(elo0)
        order0 = _order_list(elo0)
    except Exception:
        # If baseline not fit-able, then first valid action that creates a fit defines baseline
        elo0, rank0, order0 = None, None, None

    used_flip_indices = set()   # ensure we flip any given match at most once

    actions_taken = 0
    changed = False

    while actions_taken < max_actions:
        # Try to fit current to determine metric scores
        try:
            elo_curr = compute_mle_elo(current_df)
        except Exception:
            elo_curr = None

        # Select a batch of indices to act on based on operation and selection
        k_eff = max(1, int(batch_size))
        act_indices = np.array([], dtype=int)
        if operation == "add":
            chosen_set = set(map(int, chosen_indices.tolist()))
            pool = np.array([i for i in matches_df.index.values if i not in chosen_set], dtype=int)
            if pool.size == 0:
                break
            if selection == "random":
                k_take = int(min(k_eff, pool.size))
                act_indices = rng.choice(pool, size=k_take, replace=False).astype(int)
            elif elo_curr is None or pool.size == 0:
                k_take = int(min(k_eff, pool.size))
                act_indices = rng.choice(pool, size=k_take, replace=False).astype(int)
            else:
                sub = matches_df.loc[pool]
                if selection == "ci_uncertainty":
                    scores_ser = compute_ci_uncertainty_per_match(sub, elo_curr)
                    scores = scores_ser.values.astype(float)
                elif selection == "influence":
                    scores_ser = compute_influence_leverage(sub, elo_curr)
                    scores = scores_ser.values.astype(float)
                elif selection == "reward":
                    scores_ser = compute_reward_per_match(sub, elo_curr, target_player=None)
                    scores = scores_ser.values.astype(float)
                elif selection == "target_var_drop":
                    # choose target: provided or current leader
                    if target_player is None:
                        try:
                            target_name = pd.Series(elo_curr).sort_values(ascending=False).index[0]
                        except Exception:
                            target_name = None
                    else:
                        target_name = target_player
                    if target_name is None:
                        scores = np.full(sub.shape[0], np.nan, dtype=float)
                    else:
                        scores_ser = compute_target_variance_drop_per_match(sub, elo_curr, target_name)
                        scores = scores_ser.values.astype(float)
                else:
                    scores = np.full(sub.shape[0], np.nan, dtype=float)
                if np.all(np.isnan(scores)):
                    k_take = int(min(k_eff, pool.size))
                    act_indices = rng.choice(pool, size=k_take, replace=False).astype(int)
                else:
                    order = np.argsort(-np.nan_to_num(scores, nan=-np.inf))
                    k_take = int(min(k_eff, pool.size))
                    act_indices = pool[order[:k_take]].astype(int)
            # Apply add
            if act_indices.size > 0:
                chosen_indices = np.unique(np.concatenate([chosen_indices, act_indices])).astype(int)
                current_df = pd.concat([current_df, matches_df.loc[act_indices]], axis=0)

        elif operation == "remove":
            if chosen_indices.size == 0:
                break
            if selection == "random" or elo_curr is None:
                k_take = int(min(k_eff, chosen_indices.size))
                act_indices = rng.choice(chosen_indices, size=k_take, replace=False).astype(int)
            else:
                # score current chosen set by requested metric
                sub = current_df.loc[chosen_indices]
                if selection == "influence":
                    vals_ser = compute_influence_leverage(sub, elo_curr)
                    vals = vals_ser.values.astype(float)
                elif selection == "ci_uncertainty":
                    vals_ser = compute_ci_uncertainty_per_match(sub, elo_curr)
                    vals = vals_ser.values.astype(float)
                elif selection == "reward":
                    vals_ser = compute_reward_per_match(sub, elo_curr, target_player=None)
                    vals = vals_ser.values.astype(float)
                elif selection == "target_var_drop":
                    if target_player is None:
                        try:
                            target_name = pd.Series(elo_curr).sort_values(ascending=False).index[0]
                        except Exception:
                            target_name = None
                    else:
                        target_name = target_player
                    if target_name is None:
                        vals = np.full(sub.shape[0], np.nan, dtype=float)
                    else:
                        vals_ser = compute_target_variance_drop_per_match(sub, elo_curr, target_name)
                        vals = vals_ser.values.astype(float)
                else:
                    vals = np.full(sub.shape[0], np.nan, dtype=float)
                if np.all(np.isnan(vals)):
                    k_take = int(min(k_eff, chosen_indices.size))
                    act_indices = rng.choice(chosen_indices, size=k_take, replace=False).astype(int)
                else:
                    order = np.argsort(-np.nan_to_num(vals, nan=-np.inf))
                    k_take = int(min(k_eff, chosen_indices.size))
                    act_indices = chosen_indices[order[:k_take]].astype(int)
            # Apply remove
            mask_keep = ~np.isin(chosen_indices, act_indices)
            chosen_indices = chosen_indices[mask_keep]
            current_df = matches_df.loc[chosen_indices].copy()

        else:  # operation == "flip"
            # Eligible to flip: non-tie rows not yet flipped
            eligible_list = []
            for idx in chosen_indices:
                if int(idx) in used_flip_indices:
                    continue
                w = current_df.loc[int(idx), "winner"]
                if str(w) in ["model_a", "model_b"]:
                    eligible_list.append(int(idx))
            eligible = np.array(eligible_list, dtype=int)
            if eligible.size == 0:
                break
            if selection == "random" or elo_curr is None:
                k_take = int(min(k_eff, eligible.size))
                act_indices = rng.choice(eligible, size=k_take, replace=False).astype(int)
            else:
                sub = current_df.loc[eligible]
                if selection == "influence":
                    vals_ser = compute_influence_leverage(sub, elo_curr)
                    vals = vals_ser.values.astype(float)
                elif selection == "ci_uncertainty":
                    vals_ser = compute_ci_uncertainty_per_match(sub, elo_curr)
                    vals = vals_ser.values.astype(float)
                elif selection == "reward":
                    vals_ser = compute_reward_per_match(sub, elo_curr, target_player=None)
                    vals = vals_ser.values.astype(float)
                elif selection == "target_var_drop":
                    if target_player is None:
                        try:
                            target_name = pd.Series(elo_curr).sort_values(ascending=False).index[0]
                        except Exception:
                            target_name = None
                    else:
                        target_name = target_player
                    if target_name is None:
                        vals = np.full(sub.shape[0], np.nan, dtype=float)
                    else:
                        vals_ser = compute_target_variance_drop_per_match(sub, elo_curr, target_name)
                        vals = vals_ser.values.astype(float)
                else:
                    vals = np.full(sub.shape[0], np.nan, dtype=float)
                if np.all(np.isnan(vals)):
                    k_take = int(min(k_eff, eligible.size))
                    act_indices = rng.choice(eligible, size=k_take, replace=False).astype(int)
                else:
                    order = np.argsort(-np.nan_to_num(vals, nan=-np.inf))
                    k_take = int(min(k_eff, eligible.size))
                    act_indices = eligible[order[:k_take]].astype(int)
            # Apply flip for all selected indices
            for idx in act_indices:
                cur_w = current_df.loc[int(idx), "winner"]
                if str(cur_w) == "model_a":
                    new_w = "model_b"
                elif str(cur_w) == "model_b":
                    new_w = "model_a"
                else:
                    new_w = cur_w
                current_df.loc[int(idx), "winner"] = new_w
                used_flip_indices.add(int(idx))

        actions_taken += 1

        # Evaluate change criterion
        try:
            elo_now = compute_mle_elo(current_df)
        except Exception:
            # Can't evaluate yet; continue
            continue

        if elo0 is None:
            # Establish baseline at first fit-able state
            elo0 = elo_now.copy()
            rank0 = _ranking_series(elo0)
            order0 = _order_list(elo0)
            continue

        if change_mode == "any":
            order_now = _order_list(elo_now)
            if order_now != order0:
                changed = True
                break
        else:
            if target_player is None:
                # default to the best model at baseline
                target_player = order0[0] if order0 else None
            if target_player is None or target_player not in elo_now.index or target_player not in elo0.index:
                continue
            rk0 = _ranking_series(elo0).get(target_player, np.nan)
            rk1 = _ranking_series(elo_now).get(target_player, np.nan)
            if pd.notna(rk0) and pd.notna(rk1) and float(rk0) != float(rk1):
                changed = True
                break

    return {
        "seed": seed,
        "operation": operation,
        "selection": selection,
        "change_mode": change_mode,
        "target_player": target_player if change_mode == "player" else None,
        "actions_needed": actions_taken if changed else np.nan,
        "changed": bool(changed),
        "num_players": num_players,
        "n_matches": n_matches,
        "batch_size": int(batch_size),
    }


def run_min_actions_experiment(
    seeds: list,
    num_players: int,
    n_matches: int,
    operation: str,
    selection: str,
    change_mode: str,
    target_player: str = None,
    max_actions: int = 500,
    batch_size: int = 1,
    out_suffix: str = "",
    do_plot: bool = True,
    matches_df: pd.DataFrame | None = None,
    players_df: pd.DataFrame | None = None,
):
    rows = []
    for sd in seeds:
        res = _min_actions_to_change(
            seed=sd,
            num_players=num_players,
            n_matches=n_matches,
            operation=operation,
            selection=selection,
            change_mode=change_mode,
            target_player=target_player,
            max_actions=max_actions,
            batch_size=batch_size,
            matches_df=matches_df,
            players_df=players_df,
        )
        rows.append(res)
    df = pd.DataFrame(rows)
    base = f"csv/minact_{operation}_sel{selection}_mode{change_mode}_players{num_players}_matches{n_matches}_batch{batch_size}{out_suffix}"
    _ensure_dir(base + "_placeholder.csv")
    out_csv = base + ".csv"
    df.to_csv(out_csv, index=False)

    if do_plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(7, 4))
            ok = df["changed"] == True
            plt.hist(df.loc[ok, "actions_needed"].dropna().values, bins=20, alpha=0.7, label="actions to change")
            plt.xlabel("Actions needed")
            plt.ylabel("Count (seeds)")
            # Compute counts
            total_runs = int(df.shape[0])
            changed_runs = int(df["changed"].sum())
            unchanged_runs = int(total_runs - changed_runs)
            ttl = f"Min actions until ranking changes\nop={operation}, sel={selection}, mode={change_mode} | {changed_runs}/{total_runs} changed"
            if change_mode == "player" and target_player:
                ttl += f", target={target_player}"
            plt.title(ttl)
            plt.legend()
            out_png = base + ".png"
            plt.tight_layout()
            plt.savefig(out_png, dpi=150)
            plt.close()
        except Exception:
            pass

    # Summary counts CSV
    total_runs = int(df.shape[0])
    changed_runs = int(df["changed"].sum())
    unchanged_runs = int(total_runs - changed_runs)
    change_rate = float(changed_runs / total_runs) if total_runs > 0 else float("nan")
    summary_df = pd.DataFrame([{
        "operation": operation,
        "selection": selection,
        "change_mode": change_mode,
        "target_player": target_player,
        "num_players": num_players,
        "n_matches": n_matches,
        "batch_size": batch_size,
        "total_runs": total_runs,
        "changed_runs": changed_runs,
        "unchanged_runs": unchanged_runs,
        "change_rate": change_rate,
    }])
    out_summary_csv = base + "_summary.csv"
    summary_df.to_csv(out_summary_csv, index=False)

    # Console mention of how many runs could/couldn't change the result
    print(f"[min-actions] Changed in {changed_runs}/{total_runs} runs; not changed in {unchanged_runs}/{total_runs} runs.")

    return {
        "summary": df,
        "csv_path": out_csv,
        "summary_csv_path": out_summary_csv,
        "changed_runs": changed_runs,
        "unchanged_runs": unchanged_runs,
        "total_runs": total_runs,
    }


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def run_operations_till_end_report(
    seed: int,
    num_players: int,
    n_matches: int,
    operation: str,
    selection: str = "random",
    steps: int = 0,
    batch_size: int = 10,
    init_n: int = 30,
    out_suffix: str = "",
    target_player: str = None,
    reward_K: float = 1.0,
    reward_scale: float = 1.0,
    reward_base: float = math.e,
    matches_df: pd.DataFrame | None = None,
    players_df: pd.DataFrame | None = None,
):
    assert operation in {"add", "remove", "flip"}
    selection = _normalize_selection(selection)

    if matches_df is None or players_df is None:
        players_df, matches_df = generate_arena_dataset(
            num_players=num_players, n_matches=n_matches, gamma=2, seed=seed, allow_ties=True
        )
    # true skills available only for synthetic
    if {'player_name','skill_level'}.issubset(players_df.columns):
        true_skills = players_df.set_index('player_name')['skill_level']
    else:
        true_skills = None

    rng = np.random.default_rng(seed)

    # Initialize chosen set and current df
    if operation == "add":
        init_idx = rng.choice(matches_df.index.values, size=min(init_n, len(matches_df)), replace=False)
        chosen_indices = np.array(sorted(init_idx), dtype=int)
        current_df = matches_df.loc[chosen_indices].copy()
    else:
        chosen_indices = matches_df.index.values.astype(int)
        current_df = matches_df.copy()

    suffix = f"_players{num_players}_matches{n_matches}_op_{operation}_seed{seed}{out_suffix}"

    # Logs
    metrics_rows = []              # per step metrics
    ci_rows = []                   # per step, per player CI
    ranking_rows = []              # per step, per player ranking + rating + delta
    events_rows = []               # per step, what changed
    matches_state_rows = []        # per step, full indices present
    per_match_rows = []            # per step, per match reward metrics

    ranking_prev = None
    elo_prev = None

    def _player_num(name: str) -> str:
        try:
            return str(name).split('_')[-1]
        except Exception:
            return str(name)

    def _format_changed(matches_idx: np.ndarray) -> str:
        if matches_idx is None or len(matches_idx) == 0:
            return ''
        labels = []
        for i in matches_idx[:10]:  # cap to avoid overly long labels
            try:
                r = matches_df.loc[int(i)]
                a = _player_num(r['model_a'])
                b = _player_num(r['model_b'])
                w = r['winner']
                if w == 'model_a':
                    lab = f"{a}>{b}"
                elif w == 'model_b':
                    lab = f"{b}>{a}"
                else:
                    lab = f"{a}={b}"
            except Exception:
                lab = str(i)
            labels.append(lab)
        if len(matches_idx) > 10:
            labels.append('…')
        return ' | '.join(labels)

    def record_step(step: int, label: str, changed_idx: np.ndarray):
        nonlocal ranking_prev, elo_prev

        # Fit elo
        elo = compute_mle_elo(current_df)

        # Metrics vs true skills
        eval_res = evaluate_ranking_correlation(true_skills, elo)
        try:
            common_idx = true_skills.index.intersection(elo.index)
            mse_val = float(((true_skills.loc[common_idx] - elo.loc[common_idx]) ** 2).mean())
        except Exception:
            mse_val = float('nan')

        # CI
        sand, _ = compute_sandwich_ci(current_df, elo)
        for model, row in sand.iterrows():
            ci_rows.append({
                'step': step,
                'n_matches': int(len(current_df)),
                'model': model,
                'method': 'sandwich',
                'lower': float(row['lower']),
                'rating': float(row['elo']),
                'upper': float(row['upper']),
                'width': float(row['upper'] - row['lower']),
                'true_skill': float(true_skills.get(model, np.nan)),
            })

        # Log-likelihood
        try:
            loglik = compute_log_likelihood(current_df, elo)
        except Exception:
            loglik = float('nan')

        # Stability tau and per-model rating deltas
        try:
            curr_rank_series = pd.Series(elo).rank(ascending=False, method='average')
            if ranking_prev is not None:
                from scipy.stats import kendalltau
                prev_series = pd.Series(ranking_prev).reindex(curr_rank_series.index)
                tau, _ = kendalltau(prev_series.values, curr_rank_series.values)
                stability_tau = float(tau)
            else:
                stability_tau = float('nan')
            # per-model ranking log (with elo delta)
            elo_change = None
            if elo_prev is not None:
                elo_prev_aligned = pd.Series(elo_prev).reindex(elo.index).astype(float)
                elo_change = (pd.Series(elo).astype(float) - elo_prev_aligned).to_dict()
            for m, rating in elo.items():
                ranking_rows.append({
                    'step': step,
                    'model': m,
                    'rank': float(curr_rank_series[m]),
                    'rating': float(rating),
                    'elo_change': float(elo_change[m]) if elo_change is not None else float('nan'),
                })
            ranking_prev = curr_rank_series.to_dict()
            elo_prev = pd.Series(elo).to_dict()
        except Exception:
            stability_tau = float('nan')

        metrics_rows.append({
            'step': step,
            'operation': label,
            'n_matches': int(len(current_df)),
            'spearman_correlation': float(eval_res['spearman_correlation']),
            'kendall_correlation': float(eval_res['kendall_correlation']),
            'mse': mse_val,
            'log_likelihood': loglik,
            'stability_kendall_tau': stability_tau,
        })

        # Event and matches state
        events_rows.append({
            'step': step,
            'operation': label,
            'k_changed': int(changed_idx.size if changed_idx is not None else 0),
            'changed_indices': ','.join(map(str, np.asarray(changed_idx, dtype=int))) if changed_idx is not None and changed_idx.size > 0 else '',
            'changed_matches': _format_changed(changed_idx),
        })
        matches_state_rows.append({
            'step': step,
            'n_matches': int(len(current_df)),
            'indices_csv': ','.join(map(str, sorted(current_df.index.astype(int).tolist()))),
        })

        # Per-match metrics: reward, influence, CI-uncertainty
        try:
            infl_series = compute_influence_leverage(current_df, elo)
        except Exception:
            infl_series = pd.Series(np.nan, index=current_df.index, name="influence_leverage")
        try:
            ci_unc_series = compute_ci_uncertainty_per_match(current_df, elo)
        except Exception:
            ci_unc_series = pd.Series(np.nan, index=current_df.index, name="ci_uncertainty")
        try:
            if target_player is not None:
                tvd_series = compute_target_variance_drop_per_match(current_df, elo, target_player)
            else:
                # if no explicit target, default to leader at this step
                leader = None
                try:
                    leader = pd.Series(elo).sort_values(ascending=False).index[0]
                except Exception:
                    pass
                tvd_series = compute_target_variance_drop_per_match(current_df, elo, leader) if leader is not None else pd.Series(np.nan, index=current_df.index, name="target_var_drop")
        except Exception:
            tvd_series = pd.Series(np.nan, index=current_df.index, name="target_var_drop")

        for idx, row in current_df.iterrows():
            a = row['model_a']; b = row['model_b']
            w = row.get('winner', None)
            # Single reward per match: if a target is specified, compute w.r.t target;
            # otherwise, compute the average of rewards w.r.t participants (a and b)
            reward_val = float('nan')
            # Single reward per match using actual outcome:
            # If a target is specified, use it; otherwise use the actual winner's model name.
            
            if target_player is not None:
                target_name = target_player
            else:
                if str(w) == 'model_a':
                    target_name = a
                elif str(w) == 'model_b':
                    target_name = b
                else:  # tie: average of both participants
                    ra = compute_elo_add_reward_actual(elo, a, b, w, a, K=reward_K, SCALE=reward_scale, BASE=reward_base)
                    rb = compute_elo_add_reward_actual(elo, a, b, w, b, K=reward_K, SCALE=reward_scale, BASE=reward_base)
                    reward_val = float(np.nanmean([ra, rb]))
                    target_name = None
            if target_name is not None:
                reward_val = compute_elo_add_reward_actual(elo, a, b, w, target_name, K=reward_K, SCALE=reward_scale, BASE=reward_base)
            infl = float(infl_series.get(idx, np.nan))
            ciu = float(ci_unc_series.get(idx, np.nan))
            tvd = float(tvd_series.get(idx, np.nan))
            per_match_rows.append({
                'step': step,
                'match_index': int(idx) if isinstance(idx, (int, np.integer)) else idx,
                'model_a': a,
                'model_b': b,
                'winner': w,
                'reward': reward_val,
                'influence_leverage': infl,
                'ci_uncertainty': ciu,
                'target_var_drop': tvd,
            })

    # Baseline before any change
    record_step(step=0, label='baseline', changed_idx=np.array([], dtype=int))

    # Iterative operations (run to end if steps <= 0)
    step = 0
    while True:
        if steps > 0 and step >= steps:
            break
        elo_curr = None
        try:
            elo_curr = compute_mle_elo(current_df)
        except Exception:
            elo_curr = None

        sel = _select_indices_for_operation(
            operation=operation,
            selection=selection,
            matches_df=matches_df,
            chosen_indices=chosen_indices,
            current_df=current_df,
            elo_curr=elo_curr,
            batch_size=batch_size,
            rng=rng,
            target_player=target_player,
        )
        changed = sel
        if changed.size == 0:
            break
        if operation == "add":
            chosen_indices = np.unique(np.concatenate([chosen_indices, changed])).astype(int)
            current_df = pd.concat([current_df, matches_df.loc[changed]], axis=0)
        elif operation == "remove":
            mask_keep = ~np.isin(chosen_indices, changed)
            chosen_indices = chosen_indices[mask_keep]
            current_df = matches_df.loc[chosen_indices].copy()
        else:  # flip
            for idx in changed:
                cur_w = current_df.loc[int(idx), "winner"]
                if str(cur_w) == "model_a":
                    new_w = "model_b"
                elif str(cur_w) == "model_b":
                    new_w = "model_a"
                else:
                    new_w = cur_w
                current_df.loc[int(idx), "winner"] = new_w

        # If we cannot fit (e.g., design empty), skip logging for this step
        try:
            _ = compute_mle_elo(current_df)
        except Exception:
            continue

        step += 1
        record_step(step=step, label=operation, changed_idx=changed)

    # Save all logs
    metrics_df = pd.DataFrame(metrics_rows)
    ci_df = pd.DataFrame(ci_rows)
    ranking_df = pd.DataFrame(ranking_rows)
    events_df = pd.DataFrame(events_rows)
    matches_state_df = pd.DataFrame(matches_state_rows)
    per_match_df = pd.DataFrame(per_match_rows)

    base = f"csv/ops_{operation}{suffix}"
    paths = {
        'metrics': f"{base}_metrics.csv",
        'ci': f"{base}_ci.csv",
        'rankings': f"{base}_rankings.csv",
        'events': f"{base}_events.csv",
        'matches_state': f"{base}_matches_state.csv",
        'per_match': f"{base}_permatch.csv",
    }
    for _, p in paths.items():
        _ensure_dir(p)
    metrics_df.to_csv(paths['metrics'], index=False)
    ci_df.to_csv(paths['ci'], index=False)
    ranking_df.to_csv(paths['rankings'], index=False)
    events_df.to_csv(paths['events'], index=False)
    matches_state_df.to_csv(paths['matches_state'], index=False)
    per_match_df.to_csv(paths['per_match'], index=False)

    return {
        'metrics': metrics_df,
        'ci': ci_df,
        'rankings': ranking_df,
        'events': events_df,
        'matches_state': matches_state_df,
        'paths': paths,
        'true_skills': true_skills,
        'per_match': per_match_df,
    }


def main():
    ap = argparse.ArgumentParser(description="Operations experiment runner (add/remove/flip)")
    ap.add_argument('--op', '--operation', dest='operation', required=True, choices=['add', 'remove', 'flip'])
    ap.add_argument('--players', type=int, default=3)
    ap.add_argument('--matches', type=int, default=100)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--seeds', type=int, nargs='+', default=None, help='List of seeds for min-actions mode')
    ap.add_argument('--num-seeds', type=int, default=None, help='If provided, uses seeds [1..num-seeds] for min-actions mode')
    ap.add_argument('--steps', type=int, default=70)
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--init-n', type=int, default=30, dest='init_n')
    ap.add_argument('--strategy', type=str, default='random')
    ap.add_argument('--suffix', type=str, default='', dest='out_suffix')
    ap.add_argument('--plot', dest='do_plot', action='store_true')
    ap.add_argument('--no-plot', dest='do_plot', action='store_false')
    ap.set_defaults(do_plot=True)
    ap.add_argument('--target-player', type=str, default="Player_0")
    ap.add_argument('--target-mode', type=str, choices=['any','player'], default='any')
    ap.add_argument('--selection', type=str, choices=['random','influence','uncertainty','reward','target_var_drop'], default='random', help='Selection policy for min-actions mode')
    ap.add_argument('--min-actions', dest='do_min_actions', action='store_true', help='Run minimal-actions-to-change experiment across seeds')
    ap.add_argument('--max-actions', type=int, default=500, help='Max actions to try in min-actions mode')
    ap.add_argument('--reward-K', type=float, default=1.0)
    ap.add_argument('--reward-scale', type=float, default=1.0)
    ap.add_argument('--reward-base', type=float, default=math.e)
    args = ap.parse_args()




    # Branch: minimal actions experiment across seeds
    if getattr(args, 'do_min_actions', False):
        if args.target_mode == 'player' and not args.target_player:
            # Will infer later from baseline if not provided; acceptable
            pass
        if args.seeds is not None:
            seeds = list(args.seeds)
        elif args.num_seeds is not None and args.num_seeds > 0:
            seeds = list(range(1, int(args.num_seeds) + 1))
        else:
            seeds = [args.seed]
        out = run_min_actions_experiment(
            seeds=seeds,
            num_players=args.players,
            n_matches=args.matches,
            operation=args.operation,
            selection=args.selection,
            change_mode=args.target_mode,
            target_player=args.target_player,
            max_actions=args.max_actions,
            batch_size=args.batch,
            out_suffix=args.out_suffix,
            do_plot=args.do_plot,
        )
        print("Min-actions experiment saved:", out["csv_path"])
        return

    out = run_operations_till_end_report(
        seed=args.seed,
        num_players=args.players,
        n_matches=args.matches,
        operation=args.operation,
        selection=args.selection,
        steps=args.steps,
        batch_size=args.batch,
        init_n=args.init_n,
        out_suffix=args.out_suffix,
        target_player=args.target_player,
        reward_K=args.reward_K,
        reward_scale=args.reward_scale,
        reward_base=args.reward_base,
    )
    print("--------------------------------")
    if args.do_plot:
        base_tag = f"plots/ops_{args.operation}_players{args.players}_matches{args.matches}_seed{args.seed}{args.out_suffix}"
        _ensure_dir(base_tag + "_placeholder.png")
        plot_ops_ranking_cubes(out['rankings'], out['ci'], out.get('events'), operation=args.operation, true_skills=out['true_skills'], batch_size=args.batch, save_path=base_tag + "_ranking_cubes.png")
    
        # plot_ops_metrics(out['metrics'], save_path=base_tag + "_metrics.png")
    
        # plot_ops_ci_widths(out['ci'], save_path=base_tag + "_ci_widths.png")
    
        # plot_ops_events(out['events'], save_path=base_tag + "_events.png")
        
        try:
            plot_ops_reward_over_steps(out.get('per_match'), save_path=base_tag + "_reward.png")
        except Exception:
            pass
        try:
            plot_ops_influence_over_steps(out.get('per_match'), save_path=base_tag + "_influence.png")
        except Exception:
            pass


if __name__ == '__main__':
    main()


