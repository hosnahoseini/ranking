
from dataset import generate_arena_dataset
from bt import compute_mle_elo
from evaluate import (
    evaluate_ranking_correlation, 
    create_win_matrix,
    compute_sandwich_ci,
    compute_bootstrap_ci,
)
from plot import (
    plot_learning_curve, 
    plot_win_matrix_heatmap, 
    plot_rating_with_ci_vs_true,
    plot_ci_width_curves_single,
    plot_learning_curve_with_ci,
    plot_skill_trajectories_grid,
    plot_skill_trajectories_compare,
    plot_avg_width_vs_samples_compare,
    plot_learning_curve_mean_ci,
    plot_avg_width_vs_samples_compare_mean_ci,
    plot_learning_curve_compare,
    plot_learning_curve_compare_mean_ci,
    plot_mse_curve,
    plot_mse_compare,
    plot_mse_compare_mean_ci,
)
import math
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def preety_print_model_ratings(ratings):
    df = pd.DataFrame([
        [n, ratings[n]] for n in ratings.keys()
    ], columns=["Model", "Elo rating"]).sort_values("Elo rating", ascending=False).reset_index(drop=True)
    # df["Elo rating"] = (df["Elo rating"] + 0.5).astype(int)
    df.index = df.index + 1
    return df

def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def run_one(seed: int, num_players: int, n_matches: int, order: str, target_player: str = None):
    # Generate the dataset
    logger.info(f"Generating dataset (seed={seed}, players={num_players}, matches={n_matches}, order={order})")
    players_df, matches_df = generate_arena_dataset(
        num_players=num_players, n_matches=n_matches, gamma=2, seed=seed, allow_ties=True
    )
    suffix = f"_players{num_players}_matches{n_matches}_order_{order}_seed{seed}_{target_player}"
    # save to csv
    players_csv_path = f"csv/players_df{suffix}.csv"
    matches_csv_path = f"csv/matches_df{suffix}.csv"
    _ensure_dir(players_csv_path)
    _ensure_dir(matches_csv_path)
    players_df.to_csv(players_csv_path, index=False)
    matches_df.to_csv(matches_csv_path, index=False)
    logger.info(f"Generated {len(players_df)} players and {len(matches_df)} matches")
    
     
    # Get true skills
    true_skills = players_df.set_index('player_name')['skill_level']

    # Create and plot win matrix
    logger.info("=== WIN MATRIX ANALYSIS ===")
    win_matrix = create_win_matrix(matches_df)
    logger.debug("Win Matrix (showing win counts for each player pair):\n%s", win_matrix)
    
    # Plot win matrix heatmap
    # plot_win_matrix_heatmap(win_matrix, save_path=f"plots/win_matrix_heatmap{suffix}.png")
    
   
    
    # LEARNING CURVE ANALYSIS - Main focus
    logger.info("=== LEARNING CURVE ANALYSIS ===")
    logger.info("Computing learning curve (this may take a moment)...")
    
    results = []
    ci_rows = []
    # Logging accumulators
    active_pairs_rows = []          # which pairs were chosen at each step (active)
    active_player_stats_rows = []   # per-player uncertainty and ranking per step (active)
    step_size = min_matches = 10
    
    # Generate learning curve points
    max_matches = len(matches_df)
    # match_counts = range(min_matches, max_matches + 1, step_size)
    match_counts = [5, 10, 30, 50, 70, 100, 130, 150,180, max_matches]
    
    from itertools import combinations

    rng = np.random.default_rng(42)
    players = sorted(true_skills.index)
    pair_counts = {(a, b): 0 for a, b in combinations(players, 2)}
    chosen_indices = []
    subset_matches = pd.DataFrame(columns=matches_df.columns)

    def _pair_key(a, b):
        return (a, b) if a < b else (b, a)

    def _available_indices_for_pair(a, b):
        """Return unused match indices for this pair."""
        idx = matches_df[
            ((matches_df.model_a == a) & (matches_df.model_b == b)) |
            ((matches_df.model_a == b) & (matches_df.model_b == a))
        ].index
        return idx.difference(chosen_indices)

    # ---- Warm-up phase ----
    WARMUP = 5
    warm_idx = rng.choice(matches_df.index, size=WARMUP, replace=False)
    init_matches = subset_matches = pd.concat([subset_matches, matches_df.loc[warm_idx]])
    chosen_indices.extend(warm_idx.tolist())
    for _, r in matches_df.loc[warm_idx].iterrows():
        pair_counts[_pair_key(r.model_a, r.model_b)] += 1
    logger.info(f"Warm-up added {len(warm_idx)} samples; subset size now {len(subset_matches)}")
    last_time = False
    # ---- Active learning loop ----
    for target_n in tqdm(match_counts[1:] + [max_matches], desc="Active learning curve"):
        if len(subset_matches) > target_n:
            continue

        # --- Fit Elo on current subset ---
        logger.info(f"Target n={target_n}; current subset={len(subset_matches)}")
        elo_ratings = compute_mle_elo(subset_matches)

        # --- Evaluate performance ---
        eval_results = evaluate_ranking_correlation(true_skills, elo_ratings)
        # --- Compute MSE vs true skills ---
        try:
            common_idx = true_skills.index.intersection(elo_ratings.index)
            mse_val = float(((true_skills.loc[common_idx] - elo_ratings.loc[common_idx]) ** 2).mean())
            print("--------------------------------")
            print("mse_val")
            print(mse_val)  
            print("true_skills.loc[common_idx] - elo_ratings.loc[common_idx]")
            print(true_skills.loc[common_idx] - elo_ratings.loc[common_idx])
            print("true_skills.loc[common_idx]")
            print(true_skills.loc[common_idx])
            print("elo_ratings.loc[common_idx]")
            print(elo_ratings.loc[common_idx])
            print("--------------------------------")
        except Exception:
            mse_val = float('nan')

        results.append({
            'n_matches': len(subset_matches),
            'spearman_correlation': eval_results['spearman_correlation'],
            'kendall_correlation': eval_results['kendall_correlation'],
            'mse': mse_val,
        })
        logger.info(
            f"Perf at n={len(subset_matches)}: Spearman={eval_results['spearman_correlation']:.3f}, "
            f"Kendall={eval_results['kendall_correlation']:.3f}"
        )

        # --- Compute Sandwich CIs + info state ---
       
        sand, info = compute_sandwich_ci(subset_matches, elo_ratings)

        
        for model, row in sand.iterrows():
            ci_rows.append({
                'n_matches': len(subset_matches),
                'model': model,
                'method': 'sandwich',
                'lower': row['lower'],
                'rating': row['elo'],
                'upper': row['upper'],
                'width': row['upper'] - row['lower'],
                'true_skill': true_skills.get(model, np.nan),
            })

        # bootstrap
        boot = compute_bootstrap_ci(subset_matches, compute_mle_elo, rounds=5, alpha=0.05)
        for m, row in boot.iterrows():
            ci_rows.append({
                'n_matches': len(subset_matches),
                'model': m,
                'method': 'bootstrap',
                'lower': row['lower'],
                'rating': row['rating'],
                'upper': row['upper'],
                'width': row['upper'] - row['lower'],
                'true_skill': true_skills.get(m, np.nan),
            })
        # --- Record per-player uncertainty and ranking at this step (active) ---
        try:
            ranking_sorted = sorted(elo_ratings.items(), key=lambda kv: kv[1], reverse=True)
            rank_of = {m: i + 1 for i, (m, _) in enumerate(ranking_sorted)}
        except Exception:
            rank_of = {}
        for model, row in sand.iterrows():
            try:
                width = float(row['upper'] - row['lower'])
                rating = float(row['elo'])
            except Exception:
                width = float('nan')
                rating = float('nan')
            active_player_stats_rows.append({
                'n_matches': len(subset_matches),
                'player': model,
                'rating': rating,
                'ci_width': width,
                'rank': rank_of.get(model, np.nan),
                'method': 'active',
            })

        # CI width summaries
        try:
            mean_width_sand = float((sand['upper'] - sand['lower']).mean())
        except Exception:
            mean_width_sand = float('nan')
        try:
            mean_width_boot = float((boot['upper'] - boot['lower']).mean())
        except Exception:
            mean_width_boot = float('nan')
        logger.info(
            f"Mean CI width at n={len(subset_matches)}: sandwich={mean_width_sand:.3f}, bootstrap={mean_width_boot:.3f}"
        )

        # active sampling to reduce specific player uncertainty

        if target_n == max_matches and last_time:
            continue
        if target_n == max_matches and not last_time:
            last_time = True
        
        if target_player is not None:
            # active sampling to reduce specific player uncertainty
            # From compute_sandwich_ci(...): Q, Binv, base, scale (or just work in contrast space)
            models = info["models"]
            Q      = info["Q"]           # (p x p-1)
            Binv   = info["Binv"]        # (p-1 x p-1)
            logb   = math.log(10)

            # target player
            a = target_player
            if a not in models:
                logger.warning(f"Target player {a} not in current model set; falling back to random fill for this step.")
                pair_scores = {}
                n_target = 0
                top_pairs = []
            else:
                ia = models.index(a)

                # contrast-projected basis for 'a'
                e_full = np.zeros(len(models)); e_full[ia] = 1.0
                e_r = Q.T @ e_full                      # (p-1,)

                pair_scores = {}
                for i, mi in enumerate(models):
                    for j in range(i+1, len(models)):
                        mj = models[j]
                        # design row for pair (mi over mj) in full space, then project
                        x_full = np.zeros(len(models)); x_full[i] = +logb; x_full[j] = -logb
                        x_r = Q.T @ x_full

                        # alpha ≈ q*(1-q) at current β; a cheap, conservative choice is alpha=0.25
                        # (or compute q from current β if you have it in info)
                        s1 = float(x_r.T @ Binv @ x_r)
                        s2 = float(e_r.T @ Binv @ x_r)
                        alpha = 0.25  # or use model-based q*(1-q)

                        delta_var_a = (alpha * s2 * s2) / (1.0 + alpha * s1)
                        if len(_available_indices_for_pair(mi, mj)) > 0:
                            pair_scores[(mi, mj)] = delta_var_a

                n_target = min(step_size, len(pair_scores))
                # pick the top pairs (biggest variance drop for player a)
                top_pairs = sorted(pair_scores, key=pair_scores.get, reverse=True)[:n_target]
        else:
            # --- Compute pair uncertainty scores ---
            Cov = info["Cov_beta"]
            models = info["models"]
            pair_scores = {}
            for i, a in enumerate(models):
                for j, b in enumerate(models):
                    if i >= j:
                        continue
                    v_ij = Cov[i, i] + Cov[j, j] - 2 * Cov[i, j]
                    v_ij = max(v_ij, 1e-12)
                    N_ij = pair_counts.get(_pair_key(a, b), 0)
                    if N_ij == 0:
                        score = np.sqrt(v_ij)
                    else:
                        score = np.sqrt(v_ij) * ((1 / np.sqrt(N_ij)) - (1 / np.sqrt(N_ij + 1)))
                    if len(_available_indices_for_pair(a, b)) > 0:
                        pair_scores[(a, b)] = score

            if not pair_scores:
                logger.warning("No valid pairs remaining; falling back to random fill.")
                top_pairs = []

            # --- Select top-n_target pairs (most uncertain) ---
            n_target = min(step_size, len(pair_scores))
            top_pairs = [p for p, _ in sorted(pair_scores.items(), key=lambda kv: kv[1], reverse=True)[:n_target]]

        # --- Add up to 10 matches per selected pair ---
        added = 0
        n_before = len(subset_matches)
        for (a_star, b_star) in top_pairs:
            if len(subset_matches) >= target_n:
                break

            available = list(_available_indices_for_pair(a_star, b_star))
            if len(available) == 0:
                continue

            # pick min(10, available, remaining capacity)
            k = min(10, len(available), target_n - len(subset_matches))

            new_idx = rng.choice(available, size=k, replace=False)
            subset_matches = pd.concat([subset_matches, matches_df.loc[new_idx]])
            chosen_indices.extend(new_idx.tolist())

            pair_counts[_pair_key(a_star, b_star)] += k
            added += k

            # Log selected pair for this step
            active_pairs_rows.append({
                'n_before': n_before,
                'n_target': target_n,
                'pair_a': a_star,
                'pair_b': b_star,
                'k_added': int(k),
            })

        best_score = max(pair_scores.values()) if pair_scores else float('nan')
        logger.info(
            f"Selected {len(top_pairs)} pairs; added {added} samples; subset size now {len(subset_matches)}; "
            f"best uncertainty score={best_score:.4g}"
        )
        if len(active_pairs_rows) > 0 and active_pairs_rows[-1]['n_target'] == target_n:
            # Summarize pairs chosen this step
            chosen_summary = [(r['pair_a'], r['pair_b'], r['k_added']) for r in active_pairs_rows if r['n_target'] == target_n]
            # logger.info(f"Active step target_n={target_n}: pairs chosen {chosen_summary}")

        # --- Fallback random fill (if needed) ---
        if len(subset_matches) < target_n:
            remaining_capacity = target_n - len(subset_matches)
            rem = matches_df.index.difference(chosen_indices)
            if len(rem) > 0:
                fill_idx = rng.choice(rem, size=min(remaining_capacity, len(rem)), replace=False)
                subset_matches = pd.concat([subset_matches, matches_df.loc[fill_idx]])
                chosen_indices.extend(fill_idx.tolist())
                for _, r in matches_df.loc[fill_idx].iterrows():
                    pair_counts[_pair_key(r.model_a, r.model_b)] += 1
                logger.info(f"Fallback filled {len(fill_idx)} samples; subset size now {len(subset_matches)}")
        print("--------------------------------")
        print("subset_matches")
        print(len(subset_matches))
        print("--------------------------------")
    learning_curve_df = pd.DataFrame(results)
    ci_df = pd.DataFrame(ci_rows)
    # plot_learning_curve(learning_curve_df, save_path=f"plots/learning_curve{suffix}.png")
    # MSE curve for active selection
    if 'mse' in learning_curve_df.columns:
        plot_mse_curve(learning_curve_df, save_path=f"plots/mse_curve{suffix}.png")

    # Enforce same x-axis for both trajectory grids
    if len(ci_df) > 0:
        x_min = int(ci_df['n_matches'].min())
        x_max = int(ci_df['n_matches'].max())
        xlim = (x_min, x_max)
    else:
        xlim = None
    # plot_skill_trajectories_grid(ci_df, true_skills, method="sandwich", ncols=4, save_path=f"plots/trajectories_grid_sandwich{suffix}.png")
    # plot_skill_trajectories_grid(ci_df, true_skills, method="bootstrap", ncols=4, save_path=f"trajectories_grid_bootstrap{suffix}.png", xlim=xlim)
    # Combined trajectories with both CI bands
    # plot_skill_trajectories_compare(ci_df, true_skills, ncols=4, save_path=f"plots/trajectories_compare{suffix}.png", xlim=xlim)
    
    # Evaluate final ranking correlation
    eval_results = evaluate_ranking_correlation(true_skills, elo_ratings)
    # Print final rankings
    print("\n=== FINAL RANKINGS ===")
    print(preety_print_model_ratings(elo_ratings))
    


    # CI learning curve (generated in the same loop)
    logger.info("=== CI CURVES (SANDWICH & BOOTSTRAP) ===")
    ci_csv_path = f"csv/ci_learning_curve{suffix}.csv"
    _ensure_dir(ci_csv_path)
    ci_df.to_csv(ci_csv_path, index=False)
    # Separate per-method plots
    # plot_ci_width_curves_single(ci_df[ci_df['method']=="sandwich"], method="sandwich", save_path=f"plots/ci_width_sandwich{suffix}.png")
    # plot_ci_width_curves_single(ci_df[ci_df['method']=="bootstrap"], method="bootstrap", save_path=f"plots/ci_width_bootstrap{suffix}.png")

    final_n = len(matches_df)

    # Normalize sandwich CI columns
    sand_norm = sand.copy()
    # ensure index becomes a 'model' column
    sand_norm.index = sand_norm.index.rename("model")
    sand_norm = sand_norm.rename(columns={"elo": "rating"})
    sand_ci_long = sand_norm.reset_index()
    sand_ci_long["method"] = "sandwich"
    sand_ci_long = sand_ci_long[["model", "method", "lower", "rating", "upper"]]

    # Normalize bootstrap CI columns
    boot_norm = boot.copy()
    boot_norm.index = boot_norm.index.rename("model")
    boot_ci_long = boot_norm.reset_index()
    boot_ci_long["method"] = "bootstrap"
    # Force bootstrap center to match the point estimate (MLE) so only CI differs
    boot_ci_long = boot_ci_long[["model", "method", "lower", "rating", "upper"]]
    boot_ci_long["rating"] = boot_ci_long["model"].map(elo_ratings)

    final_ci = pd.concat([sand_ci_long, boot_ci_long])
    true_skills = players_df.set_index('player_name')['skill_level']
    # plot_rating_with_ci_vs_true(final_ci.assign(n_matches=final_n), true_skills, save_path=f"plots/final_ci_vs_true{suffix}.png")

    # Save learning curve data
    lc_csv_path = f"csv/learning_curve_data{suffix}.csv"
    _ensure_dir(lc_csv_path)
    learning_curve_df.to_csv(lc_csv_path, index=False)
    logger.info(f"Learning curve data saved to '{lc_csv_path}'")
    
    # Save win matrix
    win_csv_path = f"csv/win_matrix{suffix}.csv"
    _ensure_dir(win_csv_path)
    win_matrix.to_csv(win_csv_path)
    logger.info(f"Win matrix saved to '{win_csv_path}'")
    
    logger.info("=== SUMMARY ===")
    logger.info(f"Total matches analyzed: {len(matches_df)}")
    logger.info(f"Players: {len(players_df)}")
    logger.info(f"Final Spearman correlation: {eval_results['spearman_correlation']:.4f}")
    logger.info(f"Final Kendall correlation: {eval_results['kendall_correlation']:.4f}")
    
    
    # Compare random vs active: recompute CI rows for the randernate order
    logger.info("=== COMPARING RANDOM VS ACTIVE ===")
    ci_rows_rand = []
    # Logging accumulators for random strategy
    random_pairs_rows = []
    random_player_stats_rows = []
    CI_BOOTSTRAP_ROUNDS = 5
    subset_rand = init_matches
    # print("--------------------------------")
    # print("match_counts")
    # print(match_counts)
    # print("--------------------------------")
    prev_n = len(subset_rand)
    for n in match_counts:
        elo_rand = compute_mle_elo(subset_rand)
        sand_rand, active_info_state_rand = compute_sandwich_ci(subset_rand, elo_rand)
        # Compute MSE for random subset
        try:
            common_idx_r = true_skills.index.intersection(elo_rand.index)
            mse_rand = float(((true_skills.loc[common_idx_r] - elo_rand.loc[common_idx_r]) ** 2).mean())
        except Exception:
            mse_rand = float('nan')
        # Record per-player uncertainty and ranking (random)
        try:
            ranking_sorted_r = sorted(elo_rand.items(), key=lambda kv: kv[1], reverse=True)
            rank_of_r = {m: i + 1 for i, (m, _) in enumerate(ranking_sorted_r)}
        except Exception:
            rank_of_r = {}
        for m, row in sand_rand.iterrows():
            try:
                width_r = float(row['upper'] - row['lower'])
                rating_r = float(row['elo'])
            except Exception:
                width_r = float('nan')
                rating_r = float('nan')
            random_player_stats_rows.append({
                'n_matches': n,
                'player': m,
                'rating': rating_r,
                'ci_width': width_r,
                'rank': rank_of_r.get(m, np.nan),
                'method': 'random',
            })
        for m, row in sand_rand.iterrows():
            ci_rows_rand.append({
                'n_matches': n,
                'model': m,
                'method': 'sandwich',
                'lower': row['lower'],
                'rating': row['elo'],
                'upper': row['upper'],
                'width': row['upper'] - row['lower'],
            })
        boot_rand = compute_bootstrap_ci(subset_rand, compute_mle_elo, rounds=CI_BOOTSTRAP_ROUNDS, alpha=0.05)
        for m, row in boot_rand.iterrows():
            ci_rows_rand.append({
                'n_matches': n,
                'model': m,
                'method': 'bootstrap',
                'lower': row['lower'],
                'rating': row['rating'],
                'upper': row['upper'],
                'width': row['upper'] - row['lower'],
            })
        # Log which pairs were newly added at this random step (difference from prev_n)
        # try:
        idx_prev = matches_df.index.values[:prev_n]
        idx_curr = matches_df.index.values[:n]
        new_idx = np.setdiff1d(idx_curr, idx_prev, assume_unique=True)
        if len(new_idx) > 0:
            # aggregate counts per pair in this increment
            pairs, counts = np.unique([
                tuple(sorted((matches_df.loc[i, 'model_a'], matches_df.loc[i, 'model_b']))) for i in new_idx
            ], axis=0, return_counts=True)
            for (a, b), k in zip(pairs, counts):
                random_pairs_rows.append({
                    'n_prev': int(prev_n),
                    'n_curr': int(n),
                    'pair_a': a,
                    'pair_b': b,
                    'k_added': int(k),
                })
            # logger.info(f"Random step n={n}: added pairs {[(p[0], p[1], int(c)) for p, c in zip(pairs, counts)]}")
        # except Exception:
        #     pass
        prev_n = n
        subset_rand = matches_df.loc[matches_df.index.values[:len(matches_df)][:n]]
        
            
    ci_df_rand = pd.DataFrame(ci_rows_rand)

    # Plot comparison of average width vs samples
    # plot_avg_width_vs_samples_compare(ci_df_rand, ci_df, method="sandwich", save_path=f"plots/avg_width_vs_samples_sandwich{suffix}.png")
    # plot_avg_width_vs_samples_compare(ci_df_rand, ci_df, method="bootstrap", save_path=f"plots/avg_width_vs_samples_bootstrap{suffix}.png")

    # Plot learning curve comparison for this seed: random vs active
    lc_active = learning_curve_df.copy()
    # Build a random-order learning curve using simple head(n)
    lc_rows_rand = []
    subset_rand = init_matches
    for n in match_counts:
        elo_rand = compute_mle_elo(subset_rand)
        eval_rand = evaluate_ranking_correlation(true_skills, elo_rand)
        # Compute MSE for the current random subset size n
        try:
            common_idx_r2 = true_skills.index.intersection(elo_rand.index)
            mse_rand_iter = float(((true_skills.loc[common_idx_r2] - elo_rand.loc[common_idx_r2]) ** 2).mean())
        except Exception:
            mse_rand_iter = float('nan')
        lc_rows_rand.append({
            'n_matches': n,
            'spearman_correlation': eval_rand['spearman_correlation'],
            'kendall_correlation': eval_rand['kendall_correlation'],
            'mse': mse_rand_iter,
        })
        subset_rand = matches_df.loc[matches_df.index.values[:len(matches_df)][:n]]
    lc_random = pd.DataFrame(lc_rows_rand)
    
    # Save random learning curve with MSE so it can be plotted later
    lc_rand_csv_path = f"csv/learning_curve_random_data{suffix}.csv"
    _ensure_dir(lc_rand_csv_path)
    lc_random.to_csv(lc_rand_csv_path, index=False)
    
    # plot_learning_curve_compare(lc_random, lc_active, save_path=f"plots/learning_curve_compare{suffix}.png")
    # Compare MSE: random vs active (single-seed curve)
    # if 'mse' in lc_random.columns and 'mse' in lc_active.columns:
    #     plot_mse_compare(lc_random, lc_active, save_path=f"plots/mse_compare{suffix}.png")

    # Save logs to CSV
    if len(active_pairs_rows) > 0:
        p = f"csv/active_pairs_selected{suffix}.csv"; _ensure_dir(p); pd.DataFrame(active_pairs_rows).to_csv(p, index=False)
    if len(active_player_stats_rows) > 0:
        p = f"csv/active_player_stats{suffix}.csv"; _ensure_dir(p); pd.DataFrame(active_player_stats_rows).to_csv(p, index=False)
    if len(random_pairs_rows) > 0:
        p = f"csv/random_pairs_selected{suffix}.csv"; _ensure_dir(p); pd.DataFrame(random_pairs_rows).to_csv(p, index=False)
    if len(random_player_stats_rows) > 0:
        p = f"csv/random_player_stats{suffix}.csv"; _ensure_dir(p); pd.DataFrame(random_player_stats_rows).to_csv(p, index=False)


    return {
        'learning_curve': learning_curve_df,
        'learning_curve_random': lc_random,
        'ci_df_active': ci_df,
        'ci_df_random': ci_df_rand,
    }

 

def main():
    num_players = 3
    n_matches = 200
    order = "active"  # or "random"
    seeds = [1]
    target_player = None

    for target_player in [None]:
        if len(seeds) == 1:
            run_one(seed=seeds[0], num_players=num_players, n_matches=n_matches, order=order)
            return

        # Multi-seed run and aggregation
        learning_runs = []
        ci_runs = []
        for seed in seeds:
            out = run_one(seed=seed, num_players=num_players, n_matches=n_matches, order=order, target_player=target_player)
            # Active learning curve
            df_lc_act = out['learning_curve'].copy(); df_lc_act['seed'] = seed; df_lc_act['order'] = 'active'
            learning_runs.append(df_lc_act)
            # Random learning curve: use the one computed in run_one to avoid recomputation
            df_lc_rand = out['learning_curve_random'].copy(); df_lc_rand['seed'] = seed; df_lc_rand['order'] = 'random'
            learning_runs.append(df_lc_rand)
            # CI runs
            for ord_name, df in [("active", out['ci_df_active']), ("random", out['ci_df_random'])]:
                d = df[['n_matches','model','method','width']].copy()
                d['seed'] = seed
                d['order'] = ord_name
                ci_runs.append(d)

        learning_runs_df = pd.concat(learning_runs, ignore_index=True)
        ci_runs_df = pd.concat(ci_runs, ignore_index=True)

        suffix_all = f"_players{num_players}_matches{n_matches}_order_{order}_target_{target_player}_seeds{len(seeds)}"
        # plot_learning_curve_mean_ci(learning_runs_df, save_path=f"plots/learning_curve_mean{suffix_all}.png")
        plot_learning_curve_compare_mean_ci(learning_runs_df, save_path=f"plots/learning_curve_compare_mean{suffix_all}.png")
        # Aggregated MSE comparison across seeds (log-scale)
        plot_mse_compare_mean_ci(learning_runs_df, save_path=f"plots/mse_compare_mean{suffix_all}.png")
        plot_avg_width_vs_samples_compare_mean_ci(ci_runs_df, method="sandwich", save_path=f"plots/avg_width_vs_samples_sandwich_mean{suffix_all}.png")
        plot_avg_width_vs_samples_compare_mean_ci(ci_runs_df, method="bootstrap", save_path=f"plots/avg_width_vs_samples_bootstrap_mean{suffix_all}.png")
        # except Exception as e:
        #     # log in file
        #     with open("error.log", "a") as f:
        #         f.write(f"Player {target_player} Error: {e}\n")
        #     # print(f"Error: {e}")
if __name__ == "__main__":
    results = main()
