
import os
import re
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# add csv path to the system path
BASE_DIR = Path(__file__).resolve().parent / "csv" 
OUTPUT_DIR = BASE_DIR / "analysis_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _numeric_suffix(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else -1


def _find_seed_from_path(path: str) -> Optional[int]:
    # Expecting segments like ..._seed3_<Target>.csv
    match = re.search(r"_seed(\d+)_", path)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _glob_files(pattern: str) -> List[str]:
    return sorted(glob.glob(pattern))


def _ensure_runs_available(run_map: Dict[str, str]) -> Dict[str, str]:
    available: Dict[str, str] = {}
    for label, target in run_map.items():
        # look for any of the three file families across any seeds to decide availability
        pats = [
            str(BASE_DIR / f"active_player_stats_players8_matches200_order_active_seed*_{target}.csv"),
            str(BASE_DIR / f"active_pairs_selected_players8_matches200_order_active_seed*_{target}.csv"),
            str(BASE_DIR / f"learning_curve_data_players8_matches200_order_active_seed*_{target}.csv"),
        ]
        print(pats)
        found_any = any(_glob_files(p) for p in pats)
        if found_any:
            available[label] = target
    return available


def load_active_player_stats(target_suffix: str) -> pd.DataFrame:
    # Columns: n_matches,player,rating,ci_width,rank,method
    files = _glob_files(
        str(BASE_DIR / f"active_player_stats_players8_matches200_order_active_seed*_{target_suffix}.csv")
    )
    frames: List[pd.DataFrame] = []
    for fp in files:
        seed = _find_seed_from_path(fp)
        df = pd.read_csv(fp)
        df["seed"] = seed
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["n_matches", "player", "ci_width", "seed"])  # empty
    df_all = pd.concat(frames, ignore_index=True)
    # Ensure types
    df_all["n_matches"] = df_all["n_matches"].astype(int)
    df_all["player"] = df_all["player"].astype(str)
    df_all["ci_width"] = df_all["ci_width"].astype(float)
    return df_all


def load_active_pairs_selected(target_suffix: str) -> pd.DataFrame:
    # Columns: n_before,n_target,pair_a,pair_b,k_added
    files = _glob_files(
        str(BASE_DIR / f"active_pairs_selected_players8_matches200_order_active_seed*_{target_suffix}.csv")
    )
    frames: List[pd.DataFrame] = []
    for fp in files:
        seed = _find_seed_from_path(fp)
        df = pd.read_csv(fp)
        df["seed"] = seed
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["pair_a", "pair_b", "k_added", "seed"])  # empty
    df_all = pd.concat(frames, ignore_index=True)
    # Ensure types
    df_all["pair_a"] = df_all["pair_a"].astype(str)
    df_all["pair_b"] = df_all["pair_b"].astype(str)
    if "k_added" in df_all.columns:
        df_all["k_added"] = df_all["k_added"].astype(float)
    else:
        df_all["k_added"] = 1.0
    return df_all


def load_learning_curve(target_suffix: str) -> pd.DataFrame:
    # Columns: n_matches,spearman_correlation,kendall_correlation
    files = _glob_files(
        str(BASE_DIR / f"learning_curve_data_players8_matches200_order_active_seed*_{target_suffix}.csv")
    )
    frames: List[pd.DataFrame] = []
    for fp in files:
        seed = _find_seed_from_path(fp)
        df = pd.read_csv(fp)
        df["seed"] = seed
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["n_matches", "spearman_correlation", "kendall_correlation", "seed"])  # empty
    df_all = pd.concat(frames, ignore_index=True)
    df_all["n_matches"] = df_all["n_matches"].astype(int)
    df_all["spearman_correlation"] = df_all["spearman_correlation"].astype(float)
    df_all["kendall_correlation"] = df_all["kendall_correlation"].astype(float)
    return df_all


def compute_mean_ci_width(df_stats: pd.DataFrame) -> pd.DataFrame:
    # mean ci width across players per seed and n_matches, then aggregate seeds
    if df_stats.empty:
        return df_stats
    per_seed = (
        df_stats.groupby(["seed", "n_matches"], as_index=False)["ci_width"].mean()
    )
    agg = (
        per_seed.groupby("n_matches")["ci_width"]
        .agg(["mean", "std", "count"]).reset_index()
        .rename(columns={"mean": "ci_mean", "std": "ci_std", "count": "num_seeds"})
    )
    agg["ci_sem"] = agg["ci_std"] / np.sqrt(agg["num_seeds"].clip(lower=1))
    return agg


def plot_mean_ci_width(run_to_stats: Dict[str, pd.DataFrame], title: str) -> Path:
    plt.figure(figsize=(8.5, 5.0))
    tab10 = cm.get_cmap("tab10")
    palette = [tab10(i % tab10.N) for i in range(len(run_to_stats))]
    for (label, df_stats), color in zip(run_to_stats.items(), palette):
        agg = compute_mean_ci_width(df_stats)
        if agg.empty:
            continue
        x = agg["n_matches"].values
        y = agg["ci_mean"].values
        sem = agg["ci_sem"].values
        plt.plot(x, y, label=label, color=color, linewidth=2.0)
        plt.fill_between(x, y - sem, y + sem, color=color, alpha=0.2)
    plt.xlabel("Number of matches")
    plt.ylabel("Mean CI width (across players)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path = OUTPUT_DIR / "compare_mean_ci_width.png"
    plt.savefig(out_path, dpi=200)
    return out_path


def plot_per_player_ci_width(run_to_stats: Dict[str, pd.DataFrame], title: str) -> Path:
    # Get union of players across runs
    players: List[str] = []
    for df in run_to_stats.values():
        if not df.empty:
            players.extend(sorted(df["player"].unique(), key=_numeric_suffix))
    players = sorted(sorted(set(players)), key=_numeric_suffix)
    if not players:
        # nothing to plot
        return OUTPUT_DIR / "compare_per_player_ci_width.png"

    num_players = len(players)
    ncols = 4
    nrows = int(np.ceil(num_players / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.0 * ncols, 2.8 * nrows), sharex=True)
    axes = np.array(axes).reshape(nrows, ncols)
    tab10 = cm.get_cmap("tab10")
    palette = [tab10(i % tab10.N) for i in range(len(run_to_stats))]
    for idx, player in enumerate(players):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]
        for (label, df_stats), color in zip(run_to_stats.items(), palette):
            df_p = df_stats[df_stats["player"] == player]
            if df_p.empty:
                continue
            per_seed = (
                df_p.groupby(["seed", "n_matches"], as_index=False)["ci_width"].mean()
            )
            agg = (
                per_seed.groupby("n_matches")["ci_width"]
                .agg(["mean", "std", "count"]).reset_index()
                .rename(columns={"mean": "ci_mean", "std": "ci_std", "count": "num_seeds"})
            )
            agg["ci_sem"] = agg["ci_std"] / np.sqrt(agg["num_seeds"].clip(lower=1))
            x = agg["n_matches"].values
            y = agg["ci_mean"].values
            sem = agg["ci_sem"].values
            ax.plot(x, y, label=label, color=color, linewidth=2.0)
            ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.15)
        ax.set_title(player)
        if r == nrows - 1:
            ax.set_xlabel("Matches")
        ax.set_ylabel("CI width")
        ax.grid(True, alpha=0.2, linewidth=0.6)
    # Remove unused axes
    for k in range(num_players, nrows * ncols):
        r = k // ncols
        c = k % ncols
        axes[r, c].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=1.04)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "compare_per_player_ci_width.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return out_path


def _players_from_stats(run_to_stats: Dict[str, pd.DataFrame]) -> List[str]:
    players: List[str] = []
    for df in run_to_stats.values():
        if not df.empty:
            players.extend(df["player"].unique().tolist())
    players = sorted(sorted(set(players)), key=_numeric_suffix)
    return players


def plot_pair_selection_heatmaps(
    run_to_pairs: Dict[str, pd.DataFrame],
    title: str,
    n_cutoff: Optional[int] = None,
    annotate: bool = True,
) -> Path:
    players = []
    # Derive player list heuristically from pair tables and/or fallback to stats lookup
    for df in run_to_pairs.values():
        if not df.empty:
            players.extend(df["pair_a"].unique().tolist())
            players.extend(df["pair_b"].unique().tolist())
    players = sorted(sorted(set(players)), key=_numeric_suffix)
    if not players:
        # Fallback: cannot determine players
        return OUTPUT_DIR / "compare_pair_selection_heatmaps.png"

    n = len(players)
    index_map = {p: i for i, p in enumerate(players)}

    n_runs = len(run_to_pairs)
    fig, axes = plt.subplots(1, n_runs, figsize=(4.0 * n_runs, 3.8), squeeze=False)
    axes = axes[0]
    for ax, (label, df_pairs) in zip(axes, run_to_pairs.items()):
        mat = np.zeros((n, n), dtype=float)
        if not df_pairs.empty:
            if n_cutoff is not None and "n_target" in df_pairs.columns:
                df_pairs = df_pairs[df_pairs["n_target"] <= n_cutoff]
            for _, row in df_pairs.iterrows():
                a = row["pair_a"]
                b = row["pair_b"]
                w = float(row.get("k_added", 1.0))
                if a not in index_map or b not in index_map:
                    continue
                i, j = index_map[a], index_map[b]
                if i == j:
                    mat[i, j] += w
                else:
                    mat[i, j] += w
                    mat[j, i] += w
        # Build augmented matrix with sums
        row_sums = mat.sum(axis=1)
        col_sums = mat.sum(axis=0)
        total_sum = float(mat.sum())
        n_aug = n + 1
        mat_aug = np.zeros((n_aug, n_aug), dtype=float)
        mat_aug[:n, :n] = mat
        mat_aug[:n, n] = row_sums
        mat_aug[n, :n] = col_sums
        mat_aug[n, n] = total_sum

        # Normalize by max for visibility
        vmax = mat_aug.max() if mat_aug.max() > 0 else 1.0
        norm_mat = mat_aug / vmax
        im = ax.imshow(norm_mat, cmap="viridis", vmin=0.0, vmax=1.0)
        # add colorbar for each subplot
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(n_aug))
        ax.set_yticks(range(n_aug))
        ax.set_xticklabels(players + ["Sum"], rotation=90)
        ax.set_yticklabels(players + ["Sum"], rotation=0)
        subtitle = label if n_cutoff is None else f"{label} (≤ {n_cutoff})"
        ax.set_title(subtitle)
        ax.set_aspect('equal')
        if annotate:
            for i in range(n_aug):
                for j in range(n_aug):
                    val = mat_aug[i, j]
                    if val <= 0:
                        continue
                    # format as int when close to integer, else 1 decimal
                    txt = str(int(round(val))) if abs(val - round(val)) < 1e-6 else f"{val:.1f}"
                    color = "white" if norm_mat[i, j] > 0.5 else "black"
                    ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=8)
    if n_cutoff is not None:
        fig.suptitle(f"{title} (pairs up to n_target = {n_cutoff})")
    else:
        fig.suptitle(title)
    plt.tight_layout()
    out_name = (
        f"compare_pair_selection_heatmaps_upto_{n_cutoff}.png" if n_cutoff is not None else "compare_pair_selection_heatmaps.png"
    )
    out_path = OUTPUT_DIR / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return out_path


def plot_learning_curves(run_to_lc: Dict[str, pd.DataFrame], title: str) -> Path:
    plt.figure(figsize=(8.5, 5.0))
    tab10 = cm.get_cmap("tab10")
    palette = [tab10(i % tab10.N) for i in range(len(run_to_lc))]
    for (label, df), color in zip(run_to_lc.items(), palette):
        if df.empty:
            continue
        per_seed = (
            df.groupby(["seed", "n_matches"], as_index=False)["spearman_correlation"].mean()
        )
        agg = (
            per_seed.groupby("n_matches")["spearman_correlation"]
            .agg(["mean", "std", "count"]).reset_index()
            .rename(columns={"mean": "rho_mean", "std": "rho_std", "count": "num_seeds"})
        )
        agg["rho_sem"] = agg["rho_std"] / np.sqrt(agg["num_seeds"].clip(lower=1))
        x = agg["n_matches"].values
        y = agg["rho_mean"].values
        sem = agg["rho_sem"].values
        plt.plot(x, y, label=label, color=color, linewidth=2.0)
        plt.fill_between(x, y - sem, y + sem, color=color, alpha=0.2)
    plt.xlabel("Number of matches")
    plt.ylabel("Spearman correlation")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path = OUTPUT_DIR / "compare_learning_curves_spearman.png"
    plt.savefig(out_path, dpi=200)
    return out_path


def _final_player_ci(df_stats: pd.DataFrame) -> pd.DataFrame:
    if df_stats.empty:
        return pd.DataFrame(columns=["player", "final_ci_mean"])  # empty
    max_n = df_stats["n_matches"].max()
    df_final = df_stats[df_stats["n_matches"] == max_n]
    # Average across seeds
    res = (
        df_final.groupby("player")["ci_width"].mean().reset_index().rename(columns={"ci_width": "final_ci_mean"})
    )
    return res


def _auc_over_matches(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float(np.nan)
    order = np.argsort(x)
    return float(np.trapz(y[order], x[order]))


def summarize_effects(run_to_stats: Dict[str, pd.DataFrame]) -> str:
    labels = list(run_to_stats.keys())
    if len(labels) < 2:
        return "Not enough runs to compare."
    # Compare any 'Target: ...' run vs 'General' if present
    primary_a = next((l for l in labels if l.lower().startswith("general")), labels[0])
    primary_b = next(
        (l for l in labels if l.lower().startswith("target:")),
        next((l for l in labels if l != primary_a), labels[min(1, len(labels) - 1)])
    )

    df_a = run_to_stats[primary_a]
    df_b = run_to_stats[primary_b]
    if df_a.empty or df_b.empty:
        return "Insufficient data for summary."

    # Final CI comparison per player
    fa = _final_player_ci(df_a)
    fb = _final_player_ci(df_b)
    merged = fa.merge(fb, on="player", how="inner", suffixes=("_gen", "_t5"))
    merged["delta_final_ci"] = merged["final_ci_mean_gen"] - merged["final_ci_mean_t5"]
    if merged.empty:
        return "No overlapping players to compare."

    # Player most benefited (largest positive delta)
    most_benefited_row = merged.loc[merged["delta_final_ci"].idxmax()]
    most_benefited_player = str(most_benefited_row["player"])
    most_benefit_val = float(most_benefited_row["delta_final_ci"])

    # Acceleration: compare AUC of CI width over matches for the inferred target
    summary_lines: List[str] = []
    summary_lines.append(
        f"Player benefiting most (final CI reduction, {primary_b} vs {primary_a}): "
        f"{most_benefited_player} (Δ={most_benefit_val:.3f})."
    )

    target_num = _numeric_suffix(primary_b)
    inferred_target = f"player_{target_num}" if target_num >= 0 else "player_0"
    for target_player in [inferred_target]:
        dfa_p = df_a[df_a["player"] == target_player]
        dfb_p = df_b[df_b["player"] == target_player]
        if dfa_p.empty or dfb_p.empty:
            summary_lines.append(
                f"{target_player} data not present in both runs; skipping acceleration analysis."
            )
            continue
        # Aggregate across seeds: mean at each n_matches
        agg_a = (
            dfa_p.groupby(["n_matches"])["ci_width"].mean().reset_index()
        )
        agg_b = (
            dfb_p.groupby(["n_matches"])["ci_width"].mean().reset_index()
        )
        common = pd.merge(agg_a, agg_b, on="n_matches", suffixes=("_gen", "_t5"))
        if common.empty:
            summary_lines.append(
                f"{target_player}: no overlapping match counts for AUC comparison."
            )
            continue
        auc_a = _auc_over_matches(common["n_matches"].to_numpy(), common["ci_width_gen"].to_numpy())
        auc_b = _auc_over_matches(common["n_matches"].to_numpy(), common["ci_width_t5"].to_numpy())
        delta_auc = auc_a - auc_b
        summary_lines.append(
            f"{target_player}: lower AUC of CI width under {primary_b} by {delta_auc:.3f} (larger is better)."
        )

    # Effect on others (exclude player_0)
    others = merged[merged["player"] != "player_0"]["delta_final_ci"].mean()
    if pd.notnull(others):
        summary_lines.append(
            f"Average final CI reduction for non-targeted players: {others:.3f} (positive means {primary_b} helped)."
        )

    return " \n".join(summary_lines)


def _augment_with_sums(mat: np.ndarray) -> np.ndarray:
    row_sums = mat.sum(axis=1)
    col_sums = mat.sum(axis=0)
    total_sum = float(mat.sum())
    n = mat.shape[0]
    mat_aug = np.zeros((n + 1, n + 1), dtype=float)
    mat_aug[:n, :n] = mat
    mat_aug[:n, n] = row_sums
    mat_aug[n, :n] = col_sums
    mat_aug[n, n] = total_sum
    return mat_aug


def _safe_label(label: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", label).strip("_")


def _parse_targets_arg(targets: Optional[str]) -> List[str]:
    # Accepts inputs like "0,3,5" or "Player_0, Player_3"; returns suffixes like ["Player_0", ...]
    if not targets:
        return []
    suffixes: List[str] = []
    for tok in re.split(r"[,\s]+", targets.strip()):
        if not tok:
            continue
        if re.fullmatch(r"\d+", tok):
            suffixes.append(f"Player_{int(tok)}")
            continue
        low = tok.lower()
        if low.startswith("player_"):
            n = _numeric_suffix(tok)
            if n >= 0:
                suffixes.append(f"Player_{n}")
            continue
    # de-duplicate and sort in numeric order
    suffixes = sorted(sorted(set(suffixes)), key=_numeric_suffix)
    return suffixes


def plot_interval_heatmaps_per_run(
    run_to_pairs: Dict[str, pd.DataFrame],
    title_prefix: str,
    annotate: bool = True,
    max_cols: Optional[int] = None,
) -> List[Path]:
    outputs: List[Path] = []
    # Determine global players set for consistent axes across columns within each run
    all_players: List[str] = []
    for df in run_to_pairs.values():
        if not df.empty:
            all_players.extend(df["pair_a"].astype(str).unique().tolist())
            all_players.extend(df["pair_b"].astype(str).unique().tolist())
    all_players = sorted(sorted(set(all_players)), key=_numeric_suffix)
    if not all_players:
        return outputs
    n_players = len(all_players)
    index_map = {p: i for i, p in enumerate(all_players)}

    for label, df_pairs in run_to_pairs.items():
        if df_pairs.empty:
            continue
        # Identify intervals present
        df_pairs = df_pairs.copy()
        df_pairs["n_before"] = df_pairs["n_before"].astype(int)
        df_pairs["n_target"] = df_pairs["n_target"].astype(int)
        intervals = (
            df_pairs[["n_before", "n_target"]].drop_duplicates().sort_values(["n_before", "n_target"]).to_records(index=False)
        )
        intervals = [(int(a), int(b)) for (a, b) in intervals]
        if not intervals:
            continue
        if max_cols is not None:
            intervals = intervals[:max_cols]

        num_cols = len(intervals)
        num_rows = 2  # Added, All (cumulative through end of interval)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(3.6 * num_cols, 3.6 * num_rows), squeeze=False)

        for col_idx, (nb, nt) in enumerate(intervals):
            # Added in interval: exactly rows with this (n_before, n_target)
            df_added = df_pairs[(df_pairs["n_before"] == nb) & (df_pairs["n_target"] == nt)]
            mat_added = np.zeros((n_players, n_players), dtype=float)
            for _, row in df_added.iterrows():
                a = str(row["pair_a"])
                b = str(row["pair_b"])
                w = float(row.get("k_added", 1.0))
                if a in index_map and b in index_map:
                    i, j = index_map[a], index_map[b]
                    if i == j:
                        mat_added[i, j] += w
                    else:
                        mat_added[i, j] += w
                        mat_added[j, i] += w

            # All for end of interval: cumulative rows with n_target <= nt (previous + new)
            df_used = df_pairs[df_pairs["n_target"] <= nt]
            mat_used = np.zeros((n_players, n_players), dtype=float)
            for _, row in df_used.iterrows():
                a = str(row["pair_a"])
                b = str(row["pair_b"])
                w = float(row.get("k_added", 1.0))
                if a in index_map and b in index_map:
                    i, j = index_map[a], index_map[b]
                    if i == j:
                        mat_used[i, j] += w
                    else:
                        mat_used[i, j] += w
                        mat_used[j, i] += w

            # Augment with sums
            mat_added_aug = _augment_with_sums(mat_added)
            mat_used_aug = _augment_with_sums(mat_used)

            # Normalize for display per panel
            for row_idx, (title_row, mat_panel) in enumerate([("Added", mat_added_aug), ("All", mat_used_aug)]):
                ax = axes[row_idx, col_idx]
                vmax = mat_panel.max() if mat_panel.max() > 0 else 1.0
                norm_mat = mat_panel / vmax
                im = ax.imshow(norm_mat, cmap="viridis", vmin=0.0, vmax=1.0)
                if col_idx == num_cols - 1:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                n_aug = n_players + 1
                ax.set_xticks(range(n_aug))
                ax.set_yticks(range(n_aug))
                ax.set_xticklabels(all_players + ["Sum"], rotation=90)
                ax.set_yticklabels(all_players + ["Sum"], rotation=0)
                if row_idx == 0:
                    ax.set_title(f"{nb}-{nt}")
                if col_idx == 0:
                    ax.set_ylabel(title_row)
                ax.set_aspect("equal")
                if annotate:
                    for i in range(n_aug):
                        for j in range(n_aug):
                            val = mat_panel[i, j]
                            if val <= 0:
                                continue
                            txt = str(int(round(val))) if abs(val - round(val)) < 1e-6 else f"{val:.1f}"
                            color = "white" if norm_mat[i, j] > 0.5 else "black"
                            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=7)

        safe = _safe_label(label)
        fig.suptitle(f"{title_prefix}: {label}")
        plt.tight_layout()
        out_path = OUTPUT_DIR / f"interval_heatmaps_{safe}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        outputs.append(out_path)

    return outputs


def main(heatmap_n_cutoff: Optional[int] = None, targets: Optional[str] = None):
    # Define runs to load
    desired_runs: Dict[str, str] = {
        "General": "None",
    }
    target_suffixes = _parse_targets_arg(targets)
    if not target_suffixes:
        # default behavior: compare General vs Player_0 if no targets provided
        desired_runs["Target: Player 0"] = "Player_0"
    else:
        for sfx in target_suffixes:
            num = _numeric_suffix(sfx)
            label = f"Target: Player {num}" if num >= 0 else f"Target: {sfx}"
            desired_runs[label] = sfx
    runs = _ensure_runs_available(desired_runs)
    if not runs:
        print("No runs found. Please check file paths.")
        return

    missing = [lbl for lbl in desired_runs.keys() if lbl not in runs]
    if missing:
        print(f"Warning: missing runs not found on disk and will be skipped: {missing}")

    # Load data
    run_to_stats: Dict[str, pd.DataFrame] = {}
    run_to_pairs: Dict[str, pd.DataFrame] = {}
    run_to_lc: Dict[str, pd.DataFrame] = {}
    for label, target_sfx in runs.items():
        run_to_stats[label] = load_active_player_stats(target_sfx)
        run_to_pairs[label] = load_active_pairs_selected(target_sfx)
        run_to_lc[label] = load_learning_curve(target_sfx)

    # Titles
    target_labels = [lbl for lbl in runs.keys() if lbl.lower().startswith("target:")]
    if not target_labels:
        high_level_title = "Targeted Active Sampling: Player 0 vs General"
    elif len(target_labels) == 1:
        high_level_title = f"Targeted Active Sampling for {target_labels[0].split(':',1)[1].strip()} vs General"
    else:
        short_names = [lbl.split(":", 1)[1].strip() for lbl in target_labels]
        high_level_title = f"Targeted Active Sampling: {', '.join(short_names)} vs General"

    # (a) mean CI width vs number of matches (global uncertainty)
    out_a = plot_mean_ci_width(run_to_stats, title=high_level_title)
    print(f"Saved: {out_a}")

    # (b) per-player CI width curves
    out_b = plot_per_player_ci_width(run_to_stats, title="Per-Player CI Width vs Matches")
    print(f"Saved: {out_b}")

    # (c) heatmaps of pair-selection frequency
    out_c = plot_pair_selection_heatmaps(
        run_to_pairs,
        title="Pair-Selection Frequency (normalized)",
        n_cutoff=heatmap_n_cutoff,
        annotate=True,
    )
    print(f"Saved: {out_c}")

    # (d) learning-curve (Spearman) comparisons
    out_d = plot_learning_curves(run_to_lc, title="Learning Curves (Spearman) Across Runs")
    print(f"Saved: {out_d}")

    # Summary
    summary = summarize_effects(run_to_stats)
    print("\nInterpretation Summary:")
    print(summary)

    # Interval heatmaps per run
    outs_intervals = plot_interval_heatmaps_per_run(
        run_to_pairs,
        title_prefix="Pair-Selection Interval Heatmaps (Added vs Used)",
        annotate=True,
        max_cols=None,
    )
    for p in outs_intervals:
        print(f"Saved: {p}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare active sampling runs and generate plots.")
    parser.add_argument(
        "--heatmap_n",
        type=int,
        default=None,
        help="Filter pair-selection heatmaps to rows with n_target <= this value.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default=None,
        help="Comma-separated list of target players (e.g., '0,3,5' or 'Player_0,Player_3').",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Set a readable context using matplotlib rcParams
    plt.rcParams.update({
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })
    args = _parse_args()
    main(heatmap_n_cutoff=args.heatmap_n, targets=args.targets)



