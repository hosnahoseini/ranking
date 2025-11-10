import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

def plot_learning_curve(learning_curve_df, save_path=None):
    """
    Plot the learning curve showing correlation vs number of matches.
    
    Args:
        learning_curve_df: DataFrame from learning_curve_analysis
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Both Spearman and Kendall correlations on same plot
    plt.plot(learning_curve_df['n_matches'], learning_curve_df['spearman_correlation'], 
             'b-o', linewidth=2, markersize=4, label='Spearman Correlation')
    plt.plot(learning_curve_df['n_matches'], learning_curve_df['kendall_correlation'], 
             'r-o', linewidth=2, markersize=4, label='Kendall Correlation')
    
    plt.xlabel('Number of Matches')
    plt.ylabel('Correlation Coefficient')
    plt.title('Learning Curve: Ranking Correlation vs Number of Matches')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_mse_curve(learning_curve_df, save_path=None):
    """
    Plot MSE between predicted ratings and true skills vs number of matches.
    Expects column 'mse' in the provided DataFrame.
    """
    if learning_curve_df is None or len(learning_curve_df) == 0:
        return
    if 'mse' not in learning_curve_df.columns:
        return
    df = learning_curve_df.sort_values('n_matches')
    plt.figure(figsize=(10, 6))
    plt.plot(df['n_matches'], df['mse'], 'g-o', linewidth=2, markersize=4, label='MSE')
    plt.xlabel('Number of Matches')
    plt.ylabel('MSE (rating vs true skill)')
    plt.title('MSE vs Number of Matches')
    plt.grid(True, alpha=0.3)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_mse_compare(lc_random_df, lc_active_df, save_path=None):
    """
    Compare MSE curves for random vs active selection.
    Both DataFrames must contain 'n_matches' and 'mse'.
    """
    if lc_random_df is None or len(lc_random_df) == 0:
        return
    if lc_active_df is None or len(lc_active_df) == 0:
        return
    if 'mse' not in lc_random_df.columns or 'mse' not in lc_active_df.columns:
        return
    r = lc_random_df.sort_values('n_matches')
    a = lc_active_df.sort_values('n_matches')
    plt.figure(figsize=(10, 6))
    plt.plot(r['n_matches'], r['mse'], marker='o', markersize=3, linestyle='-', color='#2ca02c', label='MSE (random)')
    plt.plot(a['n_matches'], a['mse'], marker='o', markersize=3, linestyle='--', color='#2ca02c', label='MSE (active)')
    plt.xlabel('Number of Matches')
    plt.ylabel('MSE (rating vs true skill)')
    plt.title('MSE Comparison: Random vs Active')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_mse_compare_mean_ci(learning_runs_df, save_path=None):
    """
    Aggregate and compare MSE across seeds for both orders with std bands.
    Expects columns: n_matches, mse, seed, order in {"random","active"}
    Uses a log y-scale to make late small values visible.
    """
    if learning_runs_df is None or len(learning_runs_df) == 0:
        return
    if 'mse' not in learning_runs_df.columns:
        return

    df = learning_runs_df[['n_matches', 'mse', 'seed', 'order']].dropna(subset=['mse']).copy()
    if len(df) == 0:
        return

    agg = (
        df.groupby(['n_matches', 'order'])['mse']
          .agg(mean='mean', std='std')
          .reset_index()
          .sort_values(['order', 'n_matches'])
    )

    eps = 1e-8
    fig, ax = plt.subplots(figsize=(10, 6))
    for order_name, color in [("random", '#1f77b4'), ("active", '#ff7f0e')]:
        sub = agg[agg['order'] == order_name]
        if len(sub) == 0:
            continue
        x = sub['n_matches'].values
        y = sub['mean'].values
        s = sub['std'].values
        lower = np.maximum(y - s, eps)
        upper = np.maximum(y + s, eps)
        ax.plot(x, np.maximum(y, eps), '-', color=color, linewidth=2, label=f"{order_name} (mean)")
        if not np.isnan(s).all():
            ax.fill_between(x, lower, upper, color=color, alpha=0.15, label=f"{order_name} (std)")

    ax.set_xlabel('Number of Matches')
    ax.set_ylabel('MSE (rating vs true skill)')
    ax.set_title('MSE (Mean ± Std) — Random vs Active')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.legend()

    # Add small padding on both axes so lines are not clipped at the edges
    try:
        x_min = float(agg['n_matches'].min())
        x_max = float(agg['n_matches'].max())
        x_pad = max(1.0, 0.02 * (x_max - x_min))
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
    except Exception:
        pass
    try:
        std_safe = agg['std'].fillna(0.0).values
        y_mean = agg['mean'].values
        y_low = float(np.maximum(y_mean - std_safe, eps).min())
        y_high = float(np.maximum(y_mean + std_safe, eps).max())
        y_pad_low = max(1e-10, 0.08 * y_low)
        y_pad_high = 0.08 * y_high
        ax.set_ylim(y_low - y_pad_low, y_high + y_pad_high)
    except Exception:
        pass
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_mse_strategies_mean_ci(strategy_runs_df, save_path=None):
    """
    Aggregate and compare MSE across seeds for multiple active strategies.
    Expects columns: n_matches, mse, seed, strategy
    Uses log y-scale and shows mean ± std per strategy.
    """
    if strategy_runs_df is None or len(strategy_runs_df) == 0:
        return
    required = {'n_matches', 'mse', 'seed', 'strategy'}
    if not required.issubset(set(strategy_runs_df.columns)):
        return

    df = strategy_runs_df[['n_matches', 'mse', 'seed', 'strategy']].dropna(subset=['mse']).copy()
    if len(df) == 0:
        return

    agg = (
        df.groupby(['n_matches', 'strategy'])['mse']
          .agg(mean='mean', std='std')
          .reset_index()
          .sort_values(['strategy', 'n_matches'])
    )

    strategies = list(agg['strategy'].unique())
    colors = plt.cm.tab10.colors if len(strategies) <= 10 else plt.cm.tab20.colors
    color_map = {s: colors[i % len(colors)] for i, s in enumerate(strategies)}

    eps = 1e-8
    fig, ax = plt.subplots(figsize=(10, 6))
    for s in strategies:
        sub = agg[agg['strategy'] == s]
        x = sub['n_matches'].values
        y = np.maximum(sub['mean'].values, eps)
        std = sub['std'].values
        lower = np.maximum(y - std, eps)
        upper = np.maximum(y + std, eps)
        c = color_map[s]
        ax.plot(x, y, '-', color=c, linewidth=2, label=f"{s} (mean)")
        if not np.isnan(std).all():
            ax.fill_between(x, lower, upper, color=c, alpha=0.15, label=f"{s} (std)")

    ax.set_xlabel('Number of Matches')
    ax.set_ylabel('MSE (rating vs true skill)')
    ax.set_title('MSE (Mean ± Std) — Active Strategies')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.legend(ncol=2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_learning_curve_with_ci(learning_curve_df, ci_df, save_path=None):
    """
    Overlay sandwich and bootstrap CI widths on the learning curve.
    - Left y-axis: Spearman and Kendall correlations
    - Right y-axis: Average CI width (with IQR bands) for sandwich and bootstrap
    """
    # Prepare CI summaries per n_matches and method
    if ci_df is None or len(ci_df) == 0:
        # Fallback to plain learning curve
        return plot_learning_curve(learning_curve_df, save_path)

    summary = (
        ci_df
        .groupby(["n_matches", "method"])['width']
        .agg(
            mean_width='mean',
            p25=lambda s: s.quantile(0.25),
            p75=lambda s: s.quantile(0.75),
        )
        .reset_index()
    )

    # Split methods
    sand = summary[summary['method'] == 'sandwich'].sort_values('n_matches')
    boot = summary[summary['method'] == 'bootstrap'].sort_values('n_matches')

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left axis: correlations
    ax1.plot(learning_curve_df['n_matches'], learning_curve_df['spearman_correlation'],
             'b-o', linewidth=2, markersize=3, label='Spearman')
    ax1.plot(learning_curve_df['n_matches'], learning_curve_df['kendall_correlation'],
             'r-o', linewidth=2, markersize=3, label='Kendall')
    ax1.set_xlabel('Number of Matches')
    ax1.set_ylabel('Correlation Coefficient')
    ax1.grid(True, alpha=0.3)

    # Right axis: CI widths
    ax2 = ax1.twinx()
    if len(sand) > 0:
        ax2.plot(sand['n_matches'], sand['mean_width'], color='#1f77b4', linestyle='--', linewidth=2,
                 label='Sandwich CI width (mean)')
        ax2.fill_between(sand['n_matches'], sand['p25'], sand['p75'], color='#1f77b4', alpha=0.15,
                         label='Sandwich CI width (IQR)')
    if len(boot) > 0:
        ax2.plot(boot['n_matches'], boot['mean_width'], color='#ff7f0e', linestyle='--', linewidth=2,
                 label='Bootstrap CI width (mean)')
        ax2.fill_between(boot['n_matches'], boot['p25'], boot['p75'], color='#ff7f0e', alpha=0.15,
                         label='Bootstrap CI width (IQR)')
    ax2.set_ylabel('CI Width (Elo)')

    # Title and legends combined
    ax1.set_title('Learning Curve with CI Width Overlay')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_win_matrix_heatmap(win_matrix, save_path=None):
    # Keep full matrix
    data = win_matrix.to_numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap='Blues')

    # Add numeric annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{int(data[i, j])}", ha='center', va='center', color='black')


    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Wins')

    plt.title('Head-to-Head Win Matrix (no headers)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def visualize_bootstrap_scores(df, title):
    bars = pd.DataFrame(dict(
        lower = df.quantile(.025),
        rating = df.quantile(.5),
        upper = df.quantile(.975))).reset_index(names="model").sort_values("rating", ascending=False)
    bars['err_up'] = bars['upper'] - bars['rating']
    bars['err_dn'] = bars['rating'] - bars['lower']
    plt.figure(figsize=(10, 6))
    x = np.arange(len(bars))
    plt.errorbar(x, bars['rating'], yerr=np.vstack([bars['err_dn'], bars['err_up']]), fmt='o', capsize=4)
    plt.xticks(x, bars['model'], rotation=45)
    plt.ylabel('Elo (median with 95% CI)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return bars

def plot_ci_width_curves(ci_df, save_path=None):
    """
    Plot CI width vs number of matches per model for both methods.
    """
    plt.figure(figsize=(12, 7))
    for model in sorted(ci_df['model'].unique()):
        for method, style, color in [("sandwich", "-", "#1f77b4"), ("bootstrap", "--", "#ff7f0e")]:
            sub = ci_df[(ci_df['model'] == model) & (ci_df['method'] == method)].sort_values('n_matches')
            if len(sub) == 0:
                continue
            label = f"{model} ({method})"
            plt.plot(sub['n_matches'], sub['width'], linestyle=style, color=color, alpha=0.7, label=label)
    plt.xlabel('Number of Matches')
    plt.ylabel('CI Width (Elo)')
    plt.title('CI Width vs Number of Matches (Sandwich vs Bootstrap)')
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_ci_width_curves_single(ci_df, method, save_path=None):
    """
    Plot CI width vs number of matches per model for a single method.
    method: 'sandwich' or 'bootstrap'
    """
    plt.figure(figsize=(12, 7))
    for model in sorted(ci_df['model'].unique()):
        sub = ci_df[(ci_df['model'] == model) & (ci_df['method'] == method)].sort_values('n_matches')
        if len(sub) == 0:
            continue
        plt.plot(sub['n_matches'], sub['width'], linestyle='-', alpha=0.9, label=model)
    plt.xlabel('Number of Matches')
    plt.ylabel('CI Width (Elo)')
    plt.title(f'CI Width vs Number of Matches ({method.title()})')
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_rating_with_ci_vs_true(ci_df_at_end, true_skills, save_path=None):
    """
    For the final n_matches, plot estimated ratings with CI against true skills for each method.
    """
    # Convert to long form for plotting
    bars = []
    for method in ["sandwich", "bootstrap"]:
        sub = ci_df_at_end[ci_df_at_end['method'] == method]
        for _, row in sub.iterrows():
            bars.append({
                'model': row['model'],
                'method': method,
                'rating': row['rating'],
                'lower': row['lower'],
                'upper': row['upper'],
                'true_skill': true_skills.get(row['model'], np.nan),
            })
    bars = pd.DataFrame(bars)
    if len(bars) == 0:
        return
    # Matplotlib errorbar plot grouped by method
    plt.figure(figsize=(12, 6))
    x = np.arange(len(bars['model'].unique()))
    models = list(bars['model'].unique())
    width = 0.35
    for i, method in enumerate(["sandwich", "bootstrap"]):
        sub = bars[bars['method'] == method].set_index('model').reindex(models)
        y = sub['rating'].values
        # Ensure non-negative asymmetric errors for matplotlib
        err_low = np.maximum(0.0, y - sub['lower'].values)
        err_up = np.maximum(0.0, sub['upper'].values - y)
        yerr = np.vstack([err_low, err_up])
        plt.errorbar(x + (i-0.5)*width, y, yerr=yerr, fmt='o', capsize=4, label=method)
    plt.xticks(x, models, rotation=45)
    plt.ylabel('Estimated Rating (Elo)')
    plt.title('Final Ratings with 95% CI (Sandwich vs Bootstrap)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_skill_trajectories_grid(ci_df, true_skills, method="sandwich", ncols=4, sharey=True, save_path=None, xlim=None):
    """
    Subplot per player: predicted rating trajectory with shaded CI, and
    true-skill dashed line with a band whose half-width matches predicted CI half-width.
    """
    if ci_df is None or len(ci_df) == 0:
        return

    df = ci_df[ci_df['method'] == method].copy()
    if len(df) == 0:
        return

    models = sorted(df['model'].unique())
    n = len(models)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.2*nrows), sharex=True, sharey=sharey)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, model in enumerate(models):
        ax = axes[i]
        sub = df[df['model'] == model].sort_values('n_matches')
        if len(sub) == 0:
            ax.axis('off')
            continue
        x = sub['n_matches'].values
        y = sub['rating'].values
        lo = sub['lower'].values
        up = sub['upper'].values

        ax.plot(x, y, linestyle='-', linewidth=1.6, color='#1f77b4')
        ax.fill_between(x, lo, up, color='#1f77b4', alpha=0.18)

        true_val = true_skills.get(model, np.nan)
        if not math.isnan(true_val):
            ax.axhline(true_val, color='#ff7f0e', linestyle='--', linewidth=1.2)

        ax.set_title(model, fontsize=10)
        ax.grid(True, alpha=0.25)

    # Apply common x-axis limits if provided
    if xlim is not None:
        for ax in axes:
            try:
                ax.set_xlim(xlim)
            except Exception:
                pass

    # Hide any unused axes
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    # Common labels
    fig.suptitle(f'Trajectories per Player (method={method})', fontsize=12)
    fig.supxlabel('Number of Matches')
    fig.supylabel('Rating / True Skill (same scale)')

    # Figure-level legend
    pred_line = mlines.Line2D([], [], color='#1f77b4', linestyle='-', linewidth=1.6, label='Predicted rating')
    pred_band = mpatches.Patch(facecolor='#1f77b4', alpha=0.18, label='Predicted CI', edgecolor='none')
    true_line = mlines.Line2D([], [], color='#ff7f0e', linestyle='--', linewidth=1.2, label='True skill')
    fig.legend(handles=[pred_line, pred_band, true_line], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.04))

    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_skill_trajectories_compare(ci_df, true_skills, ncols=4, sharey=True, save_path=None, xlim=None):
    """
    Subplot per player with BOTH methods overlaid:
      - Center line: sandwich rating trajectory (MLE)
      - Shaded CI bands: sandwich (blue) and bootstrap (orange)
      - True skill dashed line
    """
    if ci_df is None or len(ci_df) == 0:
        return

    sand = ci_df[ci_df['method'] == 'sandwich'].copy()
    boot = ci_df[ci_df['method'] == 'bootstrap'].copy()
    if len(sand) == 0 and len(boot) == 0:
        return

    models = sorted(set(ci_df['model'].unique()))
    n = len(models)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6*ncols, 3.4*nrows), sharex=True, sharey=sharey)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, model in enumerate(models):
        ax = axes[i]
        s = sand[sand['model'] == model].sort_values('n_matches')
        b = boot[boot['model'] == model].sort_values('n_matches')
        if len(s) == 0 and len(b) == 0:
            ax.axis('off'); continue

        # Sandwich center and band
        if len(s) > 0:
            xs = s['n_matches'].values
            ys = s['rating'].values
            ls = s['lower'].values
            us = s['upper'].values
            # Center line reflects the predicted (MLE) rating
            ax.plot(xs, ys, linestyle='-', linewidth=1.6, color='#1f77b4', label='predicted' if i==0 else None)
            ax.fill_between(xs, ls, us, color='#1f77b4', alpha=0.18, label='sandwich CI' if i==0 else None)

        # Bootstrap band (centered as-is from bootstrap quantiles)
        if len(b) > 0:
            xb = b['n_matches'].values
            lb = b['lower'].values
            ub = b['upper'].values
            ax.fill_between(xb, lb, ub, color='#ff7f0e', alpha=0.18, label='bootstrap CI' if i==0 else None)

        # True skill
        tv = true_skills.get(model, np.nan)
        if not math.isnan(tv):
            # Use full x-range available from either method for visibility
            x_all = np.unique(np.concatenate([s['n_matches'].values if len(s)>0 else np.array([]),
                                              b['n_matches'].values if len(b)>0 else np.array([])]))
            if x_all.size > 0:
                ax.plot(x_all, np.full_like(x_all, tv, dtype=float), linestyle='--', linewidth=1.2, color='#2ca02c', label='true' if i==0 else None)

        ax.set_title(model, fontsize=10)
        ax.grid(True, alpha=0.25)

    # Apply common x-axis limits if provided
    if xlim is not None:
        for ax in axes:
            try:
                ax.set_xlim(xlim)
            except Exception:
                pass

    # Hide any unused axes
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    # Common labels and legend
    fig.suptitle('Trajectories per Player (sandwich vs bootstrap)', fontsize=12)
    fig.supxlabel('Number of Matches')
    fig.supylabel('Rating / True Skill (same scale)')
    handles = []
    handles.append(mlines.Line2D([], [], color='#1f77b4', linestyle='-', linewidth=1.6, label='predicted'))
    handles.append(mpatches.Patch(facecolor='#1f77b4', alpha=0.18, label='sandwich CI', edgecolor='none'))
    handles.append(mpatches.Patch(facecolor='#ff7f0e', alpha=0.18, label='bootstrap CI', edgecolor='none'))
    handles.append(mlines.Line2D([], [], color='#2ca02c', linestyle='--', linewidth=1.2, label='true'))
    fig.legend(handles=handles, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.04))

    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_avg_width_vs_samples_compare(ci_df_random, ci_df_active, method="sandwich", save_path=None):
    """
    Plot comparison of average CI width (x-axis) vs number of samples (y-axis)
    for two ordering strategies: random and active.
    """
    if ci_df_random is None or len(ci_df_random) == 0:
        return
    if ci_df_active is None or len(ci_df_active) == 0:
        return

    r = (
        ci_df_random[ci_df_random['method'] == method]
        .groupby('n_matches')['width']
        .mean()
        .reset_index()
        .sort_values('n_matches')
    )
    a = (
        ci_df_active[ci_df_active['method'] == method]
        .groupby('n_matches')['width']
        .mean()
        .reset_index()
        .sort_values('n_matches')
    )

    plt.figure(figsize=(8, 6))
    plt.plot(r['n_matches'], r['width'], '-o', label='random', color='#1f77b4', markersize=3)
    plt.plot(a['n_matches'], a['width'], '-o', label='active', color='#ff7f0e', markersize=3)
    plt.xlabel('Number of Samples (n_matches)')
    plt.ylabel('Average CI Width')
    plt.title(f'Average CI Width vs Samples ({method})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_learning_curve_mean_ci(learning_runs_df, save_path=None):
    """
    Aggregate learning curves across seeds and plot mean with std bands.
    Expects columns: n_matches, spearman_correlation, kendall_correlation, seed
    """
    if learning_runs_df is None or len(learning_runs_df) == 0:
        return

    agg = (
        learning_runs_df
        .groupby('n_matches')
        .agg(
            spearman_mean=('spearman_correlation', 'mean'),
            spearman_std=('spearman_correlation', 'std'),
            kendall_mean=('kendall_correlation', 'mean'),
            kendall_std=('kendall_correlation', 'std'),
        )
        .reset_index()
        .sort_values('n_matches')
    )

    x = agg['n_matches']
    plt.figure(figsize=(10, 6))
    # Spearman
    plt.plot(x, agg['spearman_mean'], 'b-', linewidth=2, label='Spearman (mean)')
    if not agg['spearman_std'].isna().all():
        plt.fill_between(x, agg['spearman_mean'] - agg['spearman_std'], agg['spearman_mean'] + agg['spearman_std'],
                         color='b', alpha=0.15, label='Spearman (std)')
    # Kendall
    plt.plot(x, agg['kendall_mean'], 'r-', linewidth=2, label='Kendall (mean)')
    if not agg['kendall_std'].isna().all():
        plt.fill_between(x, agg['kendall_mean'] - agg['kendall_std'], agg['kendall_mean'] + agg['kendall_std'],
                         color='r', alpha=0.15, label='Kendall (std)')

    plt.xlabel('Number of Matches')
    plt.ylabel('Correlation')
    plt.title('Learning Curve (Mean ± Std across seeds)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_avg_width_vs_samples_compare_mean_ci(ci_runs_df, method="sandwich", save_path=None):
    """
    Aggregate width vs samples across seeds. Expects columns: n_matches, width, order, method, seed
    Plots mean with std band for each order (random vs active).
    """
    if ci_runs_df is None or len(ci_runs_df) == 0:
        return

    df = ci_runs_df[ci_runs_df['method'] == method].copy()
    if len(df) == 0:
        return

    agg = (
        df.groupby(['n_matches', 'order'])['width']
          .agg(mean='mean', std='std')
          .reset_index()
          .sort_values(['order', 'n_matches'])
    )

    plt.figure(figsize=(10, 6))
    for order_name, color in [("random", '#1f77b4'), ("active", '#ff7f0e')]:
        sub = agg[agg['order'] == order_name]
        if len(sub) == 0:
            continue
        x = sub['n_matches']
        y = sub['mean']
        s = sub['std']
        plt.plot(x, y, '-', color=color, linewidth=2, label=f"{order_name} (mean)")
        if not s.isna().all():
            plt.fill_between(x, y - s, y + s, color=color, alpha=0.15, label=f"{order_name} (std)")

    plt.xlabel('Number of Samples (n_matches)')
    plt.ylabel('Average CI Width')
    plt.title(f'Average CI Width vs Samples (Mean ± Std) — {method}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_ops_ranking_cubes(rankings_df, ci_df=None, events_df=None, operation=None, true_skills=None, batch_size=None, save_path=None):
    """
    Visualize ranking trajectories as stacked rectangles per step.
    - X axis: step
    - Y axis: rank position (1 is top)
    - Color: player identity (distinct hues per player)
    - Alpha/intensity: confidence (inverse of sandwich CI width if ci_df provided)
    - If true_skills provided (mapping Series/dict), draw a separate true-ranking panel
    """
    if rankings_df is None or len(rankings_df) == 0:
        return
    models = sorted(rankings_df['model'].unique())
    steps = sorted(rankings_df['step'].unique())
    import seaborn as sns
    # Use HUSL palette for maximally distinct colors across many players
    palette = sns.color_palette('husl', n_colors=max(3, len(models)))
    color_map = {m: palette[i % len(palette)] for i, m in enumerate(models)}

    ci_width_map = None
    if ci_df is not None and len(ci_df) > 0 and 'width' in ci_df.columns:
        sub = ci_df[['step', 'model', 'width']].dropna()
        if len(sub) > 0:
            w = sub['width'].values
            w_min, w_max = float(np.min(w)), float(np.max(w))
            denom = max(1e-12, w_max - w_min)
            ci_width_map = {(int(r['step']), r['model']): float(1.0 - (r['width'] - w_min) / denom) for _, r in sub.iterrows()}

    # Map step -> n_matches for x-axis; prefer ci_df (has n_matches per step)
    step_to_n = {int(s): int(s) for s in steps}
    if ci_df is not None and 'step' in ci_df.columns and 'n_matches' in ci_df.columns:
        df_nm = (
            ci_df[['step', 'n_matches']]
            .drop_duplicates()
            .sort_values('step')
        )
        for _, r in df_nm.iterrows():
            step_to_n[int(r['step'])] = int(r['n_matches'])

    # Build tick labels that include changed indices if provided
    step_tick_labels = []
    xs = []
    step_to_event = {}
    # print("events_df", events_df)
    if events_df is not None and len(events_df) > 0:
        cols = ['step', 'k_changed']
        use_matches = 'changed_matches' in events_df.columns
        if use_matches:
            cols.append('changed_matches')
        else:
            cols.append('changed_indices')
        ev = events_df[cols].drop_duplicates('step')
        for _, r in ev.iterrows():
            if use_matches:
                step_to_event[int(r['step'])] = (int(r['k_changed']), str(r['changed_matches'] or ''))
            else:
                step_to_event[int(r['step'])] = (int(r['k_changed']), str(r['changed_indices'] or ''))
    for s in steps:
        n = step_to_n.get(int(s), int(s))
        kchg, idxs = step_to_event.get(int(s), (None, ''))
        if kchg is not None:
            # Show n on first line, change count on second, and first few indices on third
            if '|' in idxs or '>' in idxs or '=' in idxs:
                # already formatted as match strings like "0>1"
                parts = [p.strip() for p in idxs.split('|') if len(p.strip()) > 0]
            else:
                parts = [t for t in idxs.split(',') if len(t) > 0]
            idx_show = '\n'.join(parts[:]) 
            if operation == 'add':
                delta_str = f"+{kchg}"
            elif operation == 'remove':
                delta_str = f"-{kchg}"
            else:
                delta_str = f"±{kchg}"
            label = f"{n}\n{delta_str}\n{idx_show}"
        else:
            label = f"{n}"
        step_tick_labels.append(label)
    # Use uniform positions across steps to maximize bar width; labels still show n
    step_to_pos = {int(s): i for i, s in enumerate(steps)}
    xs = list(range(len(steps)))

    # Uniform bar width to use available page space generously
    P = len(models)
    base_w = 0.95
    if P <= 5:
        base_w = 0.98

    # Layout: predicted ranking cubes (left) and optional true ranking (right)
    n_x = len(xs)
    fig_w = float(np.clip(0.35 * n_x + 6.0, 8.0, 26.0))
    fig_h = float(np.clip(0.5 * P + 1.8, 3.2, 10.0))
    if true_skills is not None:
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(fig_w + 2.0, fig_h))
        gs = GridSpec(1, 2, width_ratios=[max(2, int(3 * fig_w / 8)), 1], figure=fig, wspace=0.25)
        ax = fig.add_subplot(gs[0, 0])
        ax_true = fig.add_subplot(gs[0, 1])
    else:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax_true = None

    # Predicted cubes
    for s in steps:
        d = rankings_df[rankings_df['step'] == s].copy()
        if len(d) == 0:
            continue
        d = d.sort_values('rating', ascending=False)
        for k, (_, row) in enumerate(d.iterrows()):
            m = row['model']
            y = k + 1
            x = step_to_pos.get(int(s), 0)
            c = color_map.get(m, (0.3, 0.3, 0.3))
            alpha = 0.8
            if ci_width_map is not None:
                alpha = 0.3 + 0.7 * ci_width_map.get((int(s), m), 0.5)
            rect = mpatches.Rectangle((x - base_w/2, y - 0.5), base_w, 1.0, facecolor=c, alpha=alpha, edgecolor='black', linewidth=0.3)
            ax.add_patch(rect)
            # draw compact numeric id instead of full name, centered
            try:
                num = str(m).split('_')[-1]
            except Exception:
                num = str(m)
            ax.text(x, y, num, ha='center', va='center', fontsize=7, color='white', alpha=0.9)

    ax.set_xlabel('Matches (n)')
    ax.set_ylabel('Rank (1 = top)')
    ax.set_title('Ranking cubes vs matches' + (f" — {operation}" if operation else ""))
    ax.set_xlim(-0.5, (len(xs) - 1) + 0.5)
    ax.set_ylim(0.5, len(models) + 0.5)
    ax.invert_yaxis()
    ax.grid(True, axis='y', alpha=0.2)
    # X ticks as n_matches with change info
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(step_tick_labels, fontsize=8)

    # True ranking panel
    if ax_true is not None:
        # Map true_skills to a stable order (descending skill => rank 1..P)
        if isinstance(true_skills, pd.Series):
            ts = true_skills.reindex(models)
        else:
            ts = pd.Series({m: true_skills.get(m, np.nan) for m in models})
        order = ts.sort_values(ascending=False).index.tolist()
        for k, m in enumerate(order):
            y = k + 1
            x = 0
            c = color_map.get(m, (0.3, 0.3, 0.3))
            rect = mpatches.Rectangle((x - 0.45, y - 0.5), 0.9, 1.0, facecolor=c, alpha=0.95, edgecolor='black', linewidth=0.3)
            ax_true.add_patch(rect)
            try:
                num = str(m).split('_')[-1]
            except Exception:
                num = str(m)
            ax_true.text(x, y, num, ha='center', va='center', fontsize=8, color='white', alpha=0.95)
        ax_true.set_title('True ranking')
        ax_true.set_xlim(-1, +1)
        ax_true.set_ylim(0.5, len(models) + 0.5)
        ax_true.invert_yaxis()
        ax_true.set_xticks([])
        ax_true.grid(True, axis='y', alpha=0.2)

    handles = [mpatches.Patch(color=color_map[m], label=m) for m in models]
    ncols = 4 if len(models) > 12 else 2
    ax.legend(handles=handles, loc='upper right', ncol=ncols, fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_ops_metrics(metrics_df, save_path=None):
    """
    Plot core metrics across steps: Spearman, Kendall, MSE (log), LL, stability tau.
    """
    if metrics_df is None or len(metrics_df) == 0:
        return
    df = metrics_df.sort_values('step')
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    ax = axs[0, 0]
    ax.plot(df['step'], df['spearman_correlation'], '-o', ms=3, label='Spearman')
    ax.plot(df['step'], df['kendall_correlation'], '-o', ms=3, label='Kendall')
    ax.set_title('Correlations vs step')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axs[0, 1]
    if 'mse' in df.columns:
        ax.plot(df['step'], df['mse'], '-o', ms=3, color='#2ca02c', label='MSE')
        ax.set_yscale('log')
        ax.set_title('MSE vs step (log)')
        ax.grid(True, alpha=0.3)
        ax.legend()

    ax = axs[1, 0]
    if 'log_likelihood' in df.columns:
        ax.plot(df['step'], df['log_likelihood'], '-o', ms=3, color='#9467bd', label='Log-likelihood')
        ax.set_title('Log-likelihood vs step')
        ax.grid(True, alpha=0.3)
        ax.legend()

    ax = axs[1, 1]
    if 'stability_kendall_tau' in df.columns:
        ax.plot(df['step'], df['stability_kendall_tau'], '-o', ms=3, color='#d62728', label='Stability Kendall τ')
        ax.set_title('Ranking stability vs step')
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axs.flat:
        ax.set_xlabel('Step')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_ops_ci_widths(ci_df, save_path=None):
    """
    Plot CI width over steps per player and mean±std band across players.
    """
    if ci_df is None or len(ci_df) == 0:
        return
    df = ci_df.copy()
    df = df[df['method'] == 'sandwich'] if 'method' in df.columns else df
    fig, ax = plt.subplots(figsize=(12, 6))
    for m in sorted(df['model'].unique()):
        sub = df[df['model'] == m].sort_values('step')
        if len(sub) == 0:
            continue
        ax.plot(sub['step'], sub['width'], '-', alpha=0.7, label=m)
    ax.set_xlabel('Step')
    ax.set_ylabel('CI width (sandwich)')
    ax.set_title('Per-player CI width over steps')
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)

    # Mean ± std
    agg = (
        df.groupby('step')['width']
          .agg(mean='mean', std='std')
          .reset_index()
          .sort_values('step')
    )
    ax2 = ax.twinx()
    ax2.plot(agg['step'], agg['mean'], '--', color='black', linewidth=1.8, label='mean width')
    if not agg['std'].isna().all():
        ax2.fill_between(agg['step'], agg['mean'] - agg['std'], agg['mean'] + agg['std'], color='gray', alpha=0.15, label='std')
    ax2.set_ylabel('Mean CI width')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_ops_events(events_df, save_path=None):
    """
    Plot number of changed indices per step.
    """
    if events_df is None or len(events_df) == 0:
        return
    d = events_df.sort_values('step')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(d['step'], d['k_changed'], color='#1f77b4', alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('# changed matches')
    ax.set_title('Changed matches per step')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_ops_reward_over_steps(per_match_df, save_path=None):
    """
    Plot per-step aggregated reward metrics from per-match logs.
    Shows mean ± std over steps for available columns among:
      - reward_vs_target, reward_vs_a, reward_vs_b
    """
    if per_match_df is None or len(per_match_df) == 0:
        return
    # Prefer single 'reward' column; fallback to legacy per-target columns if present
    if 'reward' in per_match_df.columns:
        df = per_match_df[['step', 'reward']].copy()
        agg = (
            df.groupby('step')['reward']
              .agg(mean='mean', std='std')
              .reset_index()
              .sort_values('step')
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        x = agg['step'].values
        y = agg['mean'].values
        s = agg['std'].values
        ax.plot(x, y, '-', linewidth=2, color='#1f77b4', label='reward (mean)')
        if not np.isnan(s).all():
            ax.fill_between(x, y - s, y + s, alpha=0.15, color='#1f77b4')
    else:
        cols = [c for c in ['reward_vs_target', 'reward_vs_a', 'reward_vs_b'] if c in per_match_df.columns]
        if not cols:
            return
        df = per_match_df[['step'] + cols].copy()
        agg = (
            df.groupby('step')[cols]
              .agg(['mean', 'std'])
              .reset_index()
              .sort_values('step')
        )
        # Flatten multiindex columns
        agg.columns = ['step'] + [f"{m}_{stat}" for m in cols for stat in ['mean', 'std']]

        fig, ax = plt.subplots(figsize=(10, 6))
        color_map = {
            'reward_vs_target': '#1f77b4',
            'reward_vs_a': '#ff7f0e',
            'reward_vs_b': '#2ca02c',
        }
        for m in cols:
            y = agg[f"{m}_mean"].values
            s = agg[f"{m}_std"].values
            x = agg['step'].values
            ax.plot(x, y, '-', linewidth=2, label=m, color=color_map.get(m))
            if not np.isnan(s).all():
                ax.fill_between(x, y - s, y + s, alpha=0.15, color=color_map.get(m))
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward (mean ± std)')
    ax.set_title('Reward metrics over steps')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_ops_influence_over_steps(per_match_df, save_path=None):
    """
    Plot per-step aggregated influence (leverage proxy) from per-match logs.
    Shows mean ± std over steps.
    """
    if per_match_df is None or len(per_match_df) == 0:
        return
    if 'influence_leverage' not in per_match_df.columns:
        return
    df = per_match_df[['step', 'influence_leverage']].copy()
    agg = (
        df.groupby('step')['influence_leverage']
          .agg(mean='mean', std='std')
          .reset_index()
          .sort_values('step')
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    x = agg['step'].values
    y = agg['mean'].values
    s = agg['std'].values
    ax.plot(x, y, '-', linewidth=2, color='#9467bd', label='Influence (mean)')
    if not np.isnan(s).all():
        ax.fill_between(x, y - s, y + s, color='#9467bd', alpha=0.18, label='std')
    ax.set_xlabel('Step')
    ax.set_ylabel('Influence (mean ± std)')
    ax.set_title('Influence over steps')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_learning_curve_compare(lc_random_df, lc_active_df, save_path=None):
    """
    Compare learning curves (Spearman, Kendall) for random vs active orders.
    Expects DataFrames with columns: n_matches, spearman_correlation, kendall_correlation
    """
    if lc_random_df is None or len(lc_random_df) == 0:
        return
    if lc_active_df is None or len(lc_active_df) == 0:
        return

    r = lc_random_df.sort_values('n_matches')
    a = lc_active_df.sort_values('n_matches')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(r['n_matches'], r['spearman_correlation'], marker='o', markersize=3, linestyle='-', color='#1f77b4', label='Spearman (random)')
    ax.plot(a['n_matches'], a['spearman_correlation'], marker='o', markersize=3, linestyle='--', color='#1f77b4', label='Spearman (active)')
    ax.plot(r['n_matches'], r['kendall_correlation'], marker='o', markersize=3, linestyle='-', color='#ff7f0e', label='Kendall (random)')
    ax.plot(a['n_matches'], a['kendall_correlation'], marker='o', markersize=3, linestyle='--', color='#ff7f0e', label='Kendall (active)')
    ax.set_xlabel('Number of Matches')
    ax.set_ylabel('Correlation')
    ax.set_title('Learning Curve Comparison: Random vs Active')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_learning_curve_compare_mean_ci(learning_runs_df, save_path=None):
    """
    Aggregate and compare learning curves across seeds for both orders.
    Expects columns: n_matches, spearman_correlation, kendall_correlation, seed, order in {"random","active"}
    """
    if learning_runs_df is None or len(learning_runs_df) == 0:
        return

    print("--------------------------------")
    print("learning_runs_df")
    print(learning_runs_df)
    print("--------------------------------")
    agg = (
        learning_runs_df
        .groupby(['n_matches', 'order'])
        .agg(
            spearman_mean=('spearman_correlation', 'mean'),
            spearman_std=('spearman_correlation', 'std'),
            kendall_mean=('kendall_correlation', 'mean'),
            kendall_std=('kendall_correlation', 'std'),
        )
        .reset_index()
        .sort_values(['order','n_matches'])
    )
    print("--------------------------------")
    print("agg")
    print(agg)
    print("--------------------------------")

    fig, ax = plt.subplots(figsize=(10, 6))
    for order_name, color in [("random", '#1f77b4'), ("active", '#ff7f0e')]:
        sub = agg[agg['order'] == order_name]
        if len(sub) == 0:
            continue
        x = sub['n_matches']
        # Spearman band
        ax.plot(x, sub['spearman_mean'], '-', color=color, linewidth=2, label=f'Spearman ({order_name})')
        if not sub['spearman_std'].isna().all():
            ax.fill_between(x, sub['spearman_mean'] - sub['spearman_std'], sub['spearman_mean'] + sub['spearman_std'],
                            color=color, alpha=0.12)
        # Kendall band (lighter shade or dashed)
        ax.plot(x, sub['kendall_mean'], '--', color=color, linewidth=2, label=f'Kendall ({order_name})')
        if not sub['kendall_std'].isna().all():
            ax.fill_between(x, sub['kendall_mean'] - sub['kendall_std'], sub['kendall_mean'] + sub['kendall_std'],
                            color=color, alpha=0.06)

    ax.set_xlabel('Number of Matches')
    ax.set_ylabel('Correlation (mean ± std)')
    ax.set_title('Learning Curve (Mean ± Std) — Random vs Active')
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()