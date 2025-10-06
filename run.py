
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
)
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def preety_print_model_ratings(ratings):
    df = pd.DataFrame([
        [n, ratings[n]] for n in ratings.keys()
    ], columns=["Model", "Elo rating"]).sort_values("Elo rating", ascending=False).reset_index(drop=True)
    # df["Elo rating"] = (df["Elo rating"] + 0.5).astype(int)
    df.index = df.index + 1
    return df

def main():
    # Generate the dataset
    print("Generating dataset...")
    num_players = 8
    n_matches = 5000
    players_df, matches_df = generate_arena_dataset(
        num_players=num_players, n_matches=n_matches, gamma=2, seed=42, allow_ties=True
    )
    suffix = f"_players{num_players}_matches{n_matches}"
    # save to csv
    players_df.to_csv(f"players_df{suffix}.csv", index=False)
    matches_df.to_csv(f"matches_df{suffix}.csv", index=False)
    print(f"Generated {len(players_df)} players and {len(matches_df)} matches")
    
     
    # Get true skills
    true_skills = players_df.set_index('player_name')['skill_level']

    # Create and plot win matrix
    print("\n=== WIN MATRIX ANALYSIS ===")
    win_matrix = create_win_matrix(matches_df)
    print("Win Matrix (showing win counts for each player pair):")
    print(win_matrix)
    
    # Plot win matrix heatmap
    plot_win_matrix_heatmap(win_matrix, save_path=f"win_matrix_heatmap{suffix}.png")
    
   
    
    # LEARNING CURVE ANALYSIS - Main focus
    print("\n=== LEARNING CURVE ANALYSIS ===")
    print("Computing learning curve (this may take a moment)...")
    
    results = []
    ci_rows = []
    step_size = 100
    min_matches = 100
    
    # Generate learning curve points
    max_matches = len(matches_df)
    match_counts = range(min_matches, max_matches + 1, step_size)
    
    for n_matches in tqdm(match_counts, desc="Computing learning curve"):
        # Use first n_matches
        subset_matches = matches_df.head(n_matches)
        
        # Compute Elo ratings (single BT fit per step)
        try:
            elo_ratings = compute_mle_elo(subset_matches)
            
            # Evaluate correlation
            eval_results = evaluate_ranking_correlation(true_skills, elo_ratings)
            
            results.append({
                'n_matches': n_matches,
                'spearman_correlation': eval_results['spearman_correlation'],
                'kendall_correlation': eval_results['kendall_correlation'],
                'spearman_p_value': eval_results['spearman_p_value'],
                'kendall_p_value': eval_results['kendall_p_value']
            })

            # Compute CIs in the same loop
            # Sandwich CI
            sand = compute_sandwich_ci(subset_matches, elo_ratings)
            for model, row in sand.iterrows():
                ci_rows.append({
                    'n_matches': n_matches,
                    'model': model,
                    'method': 'sandwich',
                    'lower': row['lower'],
                    'rating': row['elo'],
                    'upper': row['upper'],
                    'width': row['upper'] - row['lower'],
                    'true_skill': true_skills.get(model, np.nan),
                })

            # Bootstrap CI (reduced rounds per step for speed)
            CI_BOOTSTRAP_ROUNDS = 50
            boot = compute_bootstrap_ci(subset_matches, compute_mle_elo, rounds=CI_BOOTSTRAP_ROUNDS, alpha=0.05)
            for model, row in boot.iterrows():
                ci_rows.append({
                    'n_matches': n_matches,
                    'model': model,
                    'method': 'bootstrap',
                    'lower': row['lower'],
                    'rating': row['rating'],
                    'upper': row['upper'],
                    'width': row['upper'] - row['lower'],
                    'true_skill': true_skills.get(model, np.nan),
                })
        except Exception as e:
            print(f"Error with {n_matches} matches: {e}")
            continue
    
    learning_curve_df = pd.DataFrame(results)
    ci_df = pd.DataFrame(ci_rows)
    
    plot_learning_curve(learning_curve_df, save_path=f"learning_curve{suffix}.png")

    plot_skill_trajectories_grid(ci_df, true_skills, method="sandwich", ncols=4, save_path=f"trajectories_grid_sandwich{suffix}.png")
    plot_skill_trajectories_grid(ci_df, true_skills, method="bootstrap", ncols=4, save_path=f"trajectories_grid_bootstrap{suffix}.png")
    
    # Evaluate final ranking correlation
    eval_results = evaluate_ranking_correlation(true_skills, elo_ratings)
    # Print final rankings
    print("\n=== FINAL RANKINGS ===")
    print(preety_print_model_ratings(elo_ratings))
    


    # CI learning curve (generated in the same loop)
    print("\n=== CI CURVES (SANDWICH & BOOTSTRAP) ===")
    ci_df.to_csv(f"ci_learning_curve{suffix}.csv", index=False)
    # Separate per-method plots
    plot_ci_width_curves_single(ci_df[ci_df['method']=="sandwich"], method="sandwich", save_path=f"ci_width_sandwich{suffix}.png")
    plot_ci_width_curves_single(ci_df[ci_df['method']=="bootstrap"], method="bootstrap", save_path=f"ci_width_bootstrap{suffix}.png")

    # Final CIs at full data
    sand_ci = sand
    boot_ci = boot
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
    boot_ci_long = boot_ci_long[["model", "method", "lower", "rating", "upper"]]

    final_ci = pd.concat([sand_ci_long, boot_ci_long])
    true_skills = players_df.set_index('player_name')['skill_level']
    plot_rating_with_ci_vs_true(final_ci.assign(n_matches=final_n), true_skills, save_path=f"final_ci_vs_true{suffix}.png")

    # Save learning curve data
    learning_curve_df.to_csv(f"learning_curve_data{suffix}.csv", index=False)
    print(f"\nLearning curve data saved to 'learning_curve_data{suffix}.csv'")
    
    # Save win matrix
    win_matrix.to_csv(f"win_matrix{suffix}.csv")
    print(f"Win matrix saved to 'win_matrix{suffix}.csv'")
    
    print("\n=== SUMMARY ===")
    print(f"Total matches analyzed: {len(matches_df)}")
    print(f"Players: {len(players_df)}")
    print(f"Final Spearman correlation: {eval_results['spearman_correlation']:.4f}")
    print(f"Final Kendall correlation: {eval_results['kendall_correlation']:.4f}")
    
    

    return {
        'learning_curve': learning_curve_df,
        'win_matrix': win_matrix,
        'final_correlations': eval_results,
        'elo_ratings': elo_ratings
    }

if __name__ == "__main__":
    results = main()