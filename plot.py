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
        yerr = np.vstack([y - sub['lower'].values, sub['upper'].values - y])
        plt.errorbar(x + (i-0.5)*width, y, yerr=yerr, fmt='o', capsize=4, label=method)
    plt.xticks(x, models, rotation=45)
    plt.ylabel('Estimated Rating (Elo)')
    plt.title('Final Ratings with 95% CI (Sandwich vs Bootstrap)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_skill_trajectories_grid(ci_df, true_skills, method="sandwich", ncols=4, sharey=True, save_path=None):
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