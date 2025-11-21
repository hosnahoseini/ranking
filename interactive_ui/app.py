import os
import sys
import math
import pandas as pd
import numpy as np
from scipy.special import expit
import streamlit as st
import matplotlib.pyplot as plt

# Ensure project root is on sys.path so imports work when running from interactive_ui/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataset import generate_arena_dataset, load_chatbot_arena_matches, load_arena_hp55k_matches
from bt import compute_mle_elo, build_bt_design_aggregated
from evaluate import evaluate_ranking_correlation, compute_sandwich_ci, compute_reward_per_match, compute_ci_uncertainty_per_match, compute_target_variance_drop_per_match
from plot import plot_ops_ranking_cubes
from influence import compute_influence_leverage

st.set_page_config(page_title="BT Arena Interactive", layout="wide")


def _ensure_session_state_defaults():
    if 'players_df' not in st.session_state:
        st.session_state.players_df = None
    if 'matches_df' not in st.session_state:
        st.session_state.matches_df = None
    if 'base_elo' not in st.session_state:
        st.session_state.base_elo = None
    if 'subset_indices' not in st.session_state:
        st.session_state.subset_indices = None
    if 'target_player' not in st.session_state:
        st.session_state.target_player = None
    if 'has_fit' not in st.session_state:
        st.session_state.has_fit = False
    if 'history' not in st.session_state:
        st.session_state.history = []  # list of dicts per fit



def fit_bt_and_metrics(matches_df: pd.DataFrame, players_df: pd.DataFrame, target_player: str | None, full_matches_df: pd.DataFrame | None = None):
    elo = compute_mle_elo(matches_df)
    sand, _ = compute_sandwich_ci(matches_df, elo)
    # True skills only exist for synthetic datasets; arena datasets have no ground truth
    if {'player_name','skill_level'}.issubset(players_df.columns):
        true_skills = players_df.set_index('player_name')['skill_level']
    else:
        true_skills = None
    reward = compute_reward_per_match(matches_df, elo, target_player)
    influence = compute_influence_leverage(matches_df, elo)
    ci_unc = compute_ci_uncertainty_per_match(matches_df, elo)
    # Target variance drop: use provided target; if None, default to current leader
    try:
        leader = pd.Series(elo).sort_values(ascending=False).index[0]
    except Exception:
        leader = None
    tv_target = target_player if target_player else leader
    if tv_target:
        target_var_drop = compute_target_variance_drop_per_match(matches_df, elo, tv_target)
    else:
        target_var_drop = pd.Series(np.nan, index=matches_df.index, name="target_var_drop")
    # Optionally compute "all matches" metrics using the same elo
    all_metrics = {}
    if full_matches_df is not None:
        try:
            all_metrics['reward_all'] = compute_reward_per_match(full_matches_df, elo, target_player)
        except Exception:
            all_metrics['reward_all'] = pd.Series(np.nan, index=full_matches_df.index, name='reward')
        try:
            all_metrics['influence_all'] = compute_influence_leverage(full_matches_df, elo)
        except Exception:
            all_metrics['influence_all'] = pd.Series(np.nan, index=full_matches_df.index, name='influence')
        try:
            all_metrics['ci_unc_all'] = compute_ci_uncertainty_per_match(full_matches_df, elo)
        except Exception:
            all_metrics['ci_unc_all'] = pd.Series(np.nan, index=full_matches_df.index, name='ci_uncertainty')
        try:
            if tv_target:
                all_metrics['target_var_drop_all'] = compute_target_variance_drop_per_match(full_matches_df, elo, tv_target)
            else:
                all_metrics['target_var_drop_all'] = pd.Series(np.nan, index=full_matches_df.index, name='target_var_drop')
        except Exception:
            all_metrics['target_var_drop_all'] = pd.Series(np.nan, index=full_matches_df.index, name='target_var_drop')
    return elo, sand, true_skills, reward, influence, ci_unc, target_var_drop, all_metrics


def build_players_table(elo: pd.Series, ci_df: pd.DataFrame, true_skills: pd.Series | None) -> pd.DataFrame:
    d = pd.DataFrame({'pred_skill': elo}).copy()
    d.index.name = 'player'
    d = d.reset_index()
    d = d.merge(ci_df[['lower','elo','upper']].rename(columns={'elo':'pred_skill_ci_center'}), left_on='player', right_index=True)
    # True skill and true rank (if available)
    if true_skills is not None:
        ts = true_skills.rename('true_skill').reset_index().rename(columns={'player_name': 'player'})
        d = d.merge(ts, on='player', how='left')
        try:
            tr = true_skills.rank(ascending=False, method='min').rename('true_rank').reset_index().rename(columns={'player_name': 'player'})
            d = d.merge(tr, on='player', how='left')
        except Exception:
            d['true_rank'] = np.nan
    else:
        d['true_skill'] = np.nan
        d['true_rank'] = np.nan
    # Predicted rank
    try:
        pr = pd.Series(elo).rank(ascending=False, method='min').rename('pred_rank')
        d = d.merge(pr, left_on='player', right_index=True, how='left')
    except Exception:
        d['pred_rank'] = np.nan
    d['ci_width'] = d['upper'] - d['lower']
    # Drop raw CI columns and center to keep table compact
    d = d.drop(columns=['lower','upper','pred_skill_ci_center'], errors='ignore')
    return d.sort_values('pred_skill', ascending=False)


def main():
    _ensure_session_state_defaults()

    with st.sidebar:
        st.header("Config")
        dataset_mode = st.radio("Dataset", options=["Synthetic", "Chatbot Arena", "Arena HP55k"], index=0, horizontal=True)
        allow_ties = st.checkbox("Allow ties", value=True)
        target_player = st.text_input("Target player (optional exact name)", value="") or None
        if dataset_mode == "Synthetic":
            n_players = st.number_input("Players", value=3, min_value=2, step=1)
            n_matches = st.number_input("Matches", value=15, min_value=10, step=10)
            seed = st.number_input("Seed", value=1, min_value=0, step=1)
        elif dataset_mode == "Chatbot Arena":
            hf_name = st.text_input("HF dataset", value="lmsys/chatbot_arena_conversations")
            hf_split = st.text_input("Split", value="train")
            max_rows = st.number_input("Max rows", value=2000, min_value=100, step=100)
        else:
            hf_name = st.text_input("HF dataset", value="lmarena-ai/arena-human-preference-55k")
            hf_split = st.text_input("Split", value="train")
            max_rows = st.number_input("Max rows", value=20000, min_value=100, step=100)
        refresh = st.button("Load dataset")

    if st.session_state.players_df is None or refresh:
        if dataset_mode == "Synthetic":
            players_df, matches_df = generate_arena_dataset(
                num_players=int(n_players), n_matches=int(n_matches), gamma=2, seed=int(seed), allow_ties=allow_ties
            )
        elif dataset_mode == "Chatbot Arena":
            try:
                matches_df = load_chatbot_arena_matches(
                    split=hf_split, hf_dataset_name=hf_name, allow_ties=allow_ties, max_rows=int(max_rows)
                )
            except Exception as e:
                st.error(f"Failed to load Chatbot Arena data: {e}")
                matches_df = pd.DataFrame(columns=['model_a','model_b','winner'])
            models = sorted(set(matches_df.get('model_a', pd.Series()).astype(str)) | set(matches_df.get('model_b', pd.Series()).astype(str)))
            players_df = pd.DataFrame({'player_name': models})
        else:
            try:
                matches_df = load_arena_hp55k_matches(
                    split=hf_split, hf_dataset_name=hf_name, allow_ties=allow_ties, max_rows=int(max_rows)
                )
            except Exception as e:
                st.error(f"Failed to load Arena HP55k data: {e}")
                matches_df = pd.DataFrame(columns=['model_a','model_b','winner'])
            models = sorted(set(matches_df.get('model_a', pd.Series()).astype(str)) | set(matches_df.get('model_b', pd.Series()).astype(str)))
            players_df = pd.DataFrame({'player_name': models})
        st.session_state.players_df = players_df
        st.session_state.matches_df = matches_df
        st.session_state.subset_indices = matches_df.index.tolist()  # default all selected
        # Base ranking on full dataset
        base_elo, _, _, _, _, _, _, _ = fit_bt_and_metrics(matches_df, players_df, target_player=None, full_matches_df=matches_df)
        st.session_state.base_elo = base_elo
        st.session_state.target_player = target_player
        st.session_state.dataset_mode = dataset_mode

    players_df = st.session_state.players_df
    matches_df = st.session_state.matches_df
    target_player = st.session_state.target_player if st.session_state.target_player else None

    st.title("BT Arena — Interactive Analysis")
    col_players, col_matches = st.columns([1, 2])

    # --- Players table (top) ---
    col_players.subheader("Players")
    # True skills only exist for synthetic dataset
    if {'player_name','skill_level'}.issubset(players_df.columns):
        true_skills = players_df.set_index('player_name')['skill_level']
    else:
        true_skills = None
    if st.session_state.has_fit and st.session_state.history:
        last = st.session_state.history[-1]
        tbl_players = build_players_table(last['elo'], last['ci'], true_skills)
        col_players.dataframe(tbl_players.style.background_gradient(subset=['pred_skill'], cmap='Blues'), use_container_width=True)
    else:
        # Before first fit: show all columns with NaN where not applicable
        if true_skills is not None:
            empty = pd.DataFrame({'player': true_skills.index, 'true_skill': true_skills.values})
            empty['true_rank'] = true_skills.rank(ascending=False, method='min').values
        else:
            base_players = list(players_df['player_name']) if 'player_name' in players_df.columns else []
            empty = pd.DataFrame({'player': base_players, 'true_skill': np.nan})
            empty['true_rank'] = np.nan
        empty['pred_skill'] = np.nan; empty['pred_rank'] = np.nan
        empty['ci_width'] = np.nan
        col_players.dataframe(empty, use_container_width=True)

    try:
        # Probability matrix: prefer ground-truth skills if available, else use current/base elo
        if true_skills is not None and len(true_skills) > 0:
            players_list = list(true_skills.index)
            ratings = true_skills.values.astype(float)
        else:
            # Use latest fitted elo if available, else base elo
            if st.session_state.has_fit and st.session_state.history:
                elo_for_probs = st.session_state.history[-1].get('elo', st.session_state.base_elo)
            else:
                elo_for_probs = st.session_state.base_elo
            if isinstance(elo_for_probs, pd.Series):
                players_list = list(elo_for_probs.index)
                ratings = elo_for_probs.loc[players_list].astype(float).values
            else:
                players_list, ratings = [], np.array([])
        if len(players_list) > 0:
            diff = ratings[:, None] - ratings[None, :]
            prob_mat = 1 / (1 + np.exp(-diff))
            prob_df = pd.DataFrame(prob_mat, index=players_list, columns=players_list)
            np.fill_diagonal(prob_df.values, np.nan)
            col_players.subheader("BT win probability matrix P(i beats j)")
            col_players.dataframe(prob_df.style.format("{:.3f}").background_gradient(cmap='Blues'), use_container_width=True)
            # Win counts matrix: # times i beat j (ties excluded)
            try:
                wins_ab = pd.pivot_table(
                    matches_df[matches_df['winner'] == 'model_a'],
                    index='model_a', columns='model_b', aggfunc='size', fill_value=0
                )
                wins_ba = pd.pivot_table(
                    matches_df[matches_df['winner'] == 'model_b'],
                    index='model_b', columns='model_a', aggfunc='size', fill_value=0
                )
                wins_ab = wins_ab.reindex(index=players_list, columns=players_list, fill_value=0)
                wins_ba = wins_ba.reindex(index=players_list, columns=players_list, fill_value=0)
                wins_mat = wins_ab.add(wins_ba, fill_value=0).astype(int)
                np.fill_diagonal(wins_mat.values, 0)
                col_players.subheader("Win counts matrix # wins(i over j)")
                col_players.dataframe(wins_mat.style.format("{:d}").background_gradient(cmap='Greens'), use_container_width=True)
            except Exception:
                pass
    except Exception:
        pass

    st.markdown("---")

    # --- Matches selection and table (right) ---
    col_matches.subheader("Matches — batch selection + per-row editing")
    # Optional prior metrics to assist sorting
    last_reward = None; last_infl = None; last_ci_unc = None; last_tvd = None
    if st.session_state.has_fit and st.session_state.history:
        last = st.session_state.history[-1]
        last_reward = last.get('reward')
        last_infl = last.get('influence')
        last_ci_unc = last.get('ci_uncertainty')
        last_tvd = last.get('target_var_drop')
    df_matches_view = matches_df.copy().reset_index().rename(columns={'index':'match_index'})
    # Ensure numeric sort for match_index (avoid lexicographic behavior)
    try:
        df_matches_view['match_index'] = pd.to_numeric(df_matches_view['match_index'], errors='coerce')
    except Exception:
        pass
    # Always include metric columns; fill with NaN before first fit
    df_matches_view['reward'] = last_reward.reindex(matches_df.index).values if last_reward is not None else np.nan
    df_matches_view['influence'] = last_infl.reindex(matches_df.index).values if last_infl is not None else np.nan
    df_matches_view['ci_uncertainty'] = last_ci_unc.reindex(matches_df.index).values if last_ci_unc is not None else np.nan
    df_matches_view['target_var_drop'] = last_tvd.reindex(matches_df.index).values if last_tvd is not None else np.nan

    # Sorting and batch selection controls
    sort_options = [c for c in df_matches_view.columns]
    if not sort_options:
        sort_options = ['match_index','model_a','model_b','winner']
    col_sort, col_asc, col_range = col_matches.columns([1,1,2])
    # Default to 'reward' if available, else first option
    default_sort_index = sort_options.index('match_index')
    sort_col = col_sort.selectbox("Sort by", options=sort_options, index=default_sort_index)
    ascending = col_asc.checkbox("Ascending", value=True)
    df_sorted = df_matches_view.sort_values(sort_col, ascending=ascending, na_position='last')
    r = col_range.slider("Range (by row index in sorted table)", min_value=0, max_value=max(0, len(df_sorted)-1), value=(0, min(len(df_sorted)-1, 19)))
    cr_sel, cr_fit = col_range.columns([1,1])
    if cr_sel.button("Select Range"):
        sel = df_sorted.iloc[r[0]:r[1]+1]['match_index'].tolist()
        st.session_state.subset_indices = sel
    if cr_fit.button("Fit BT on selected matches"):
        subset = matches_df.loc[st.session_state.subset_indices]
        elo, ci_df, true_skills, reward_series, influence_series, ci_unc_series, tvd_series, all_metrics = fit_bt_and_metrics(subset, players_df, target_player, full_matches_df=matches_df)
        reward_all = all_metrics.get('reward_all')
        influence_all = all_metrics.get('influence_all')
        ci_unc_all = all_metrics.get('ci_unc_all')
        tvd_all = all_metrics.get('target_var_drop_all')
        hist_item = {
            'step': len(st.session_state.history),
            'elo': elo,
            'ci': ci_df.assign(step=len(st.session_state.history)),
            'n_matches': len(subset),
            'reward': reward_all,
            'influence': influence_all,
            'ci_uncertainty': ci_unc_all,
            'target_var_drop': tvd_all,
            'selection': list(st.session_state.subset_indices),
            'winners': matches_df['winner'].copy(),
        }
        try:
            corr = evaluate_ranking_correlation(true_skills, elo)
            hist_item['spearman'] = float(corr['spearman_correlation'])
            hist_item['kendall'] = float(corr['kendall_correlation'])
        except Exception:
            pass
        st.session_state.history.append(hist_item)
        st.session_state.has_fit = True

    # Per-row checkbox selection on the sorted view
    df_sorted['selected'] = df_sorted['match_index'].isin(st.session_state.subset_indices)
    edited = col_matches.data_editor(
        df_sorted,
        column_config={
            "selected": st.column_config.CheckboxColumn("Use", help="Include in BT fit"),
            "winner": st.column_config.SelectboxColumn(
                "Winner",
                options=(['model_a','model_b','tie'] if allow_ties else ['model_a','model_b']),
                help="Edit the match result to see its effect on refit",
            ),
        },
        disabled=[c for c in df_sorted.columns if c not in ['selected','winner']],
        use_container_width=True,
        height=min(600, 48 + 28 * min(20, len(df_sorted))),
    )
    if 'selected' in edited.columns:
        st.session_state.subset_indices = edited.loc[edited['selected'], 'match_index'].tolist()
    # Apply winner edits back to the source matches_df; effect will be visible after next fit
    try:
        if {'match_index','winner'}.issubset(set(edited.columns)):
            win_updates = edited[['match_index','winner']].dropna()
            win_updates['match_index'] = win_updates['match_index'].astype(int)
            for _, r in win_updates.iterrows():
                try:
                    st.session_state.matches_df.at[int(r['match_index']), 'winner'] = str(r['winner'])
                except Exception:
                    pass
            # keep local copy in sync for immediate UI continuity
            matches_df = st.session_state.matches_df
    except Exception:
        pass

    # (Fit/Refit handled above next to Select Range)

    # No second matches table: metrics are included in the editable table above

    # --- History inspector: track any cell(s) over steps ---
    if st.session_state.history:
        st.markdown("---")
        st.subheader("History inspector — track cell values over steps (x = n_matches)")
        hist = st.session_state.history
        left_ctrl, right_plot = st.columns([1, 2])

        entity = left_ctrl.selectbox("Entity", options=["Players", "Matches"], index=0)
        if entity == "Players":
            available_players = sorted(list(set().union(*[set(h['elo'].index) for h in hist if isinstance(h.get('elo'), pd.Series)])))
            players_sel = left_ctrl.multiselect("Players", options=available_players, default=available_players[:min(3, len(available_players))])
            metric = left_ctrl.selectbox("Metric", options=["pred_skill", "pred_rank", "ci_lower", "ci_upper", "ci_width"], index=0)

            # Build series per selected player over steps
            steps = [h.get('step', i) for i, h in enumerate(hist)]
            n_list = [h.get('n_matches', np.nan) for h in hist]
            fig, ax = plt.subplots(figsize=(6.5, 3.0))
            for player in players_sel:
                vals = []
                for h in hist:
                    val = np.nan
                    try:
                        if metric == "pred_skill":
                            val = float(h['elo'].get(player, np.nan))
                        elif metric == "pred_rank":
                            val = float(pd.Series(h['elo']).rank(ascending=False, method='min').get(player, np.nan))
                        elif metric in ("ci_lower", "ci_upper", "ci_width"):
                            if isinstance(h.get('ci'), pd.DataFrame) and player in h['ci'].index:
                                lo = float(h['ci'].loc[player, 'lower'])
                                up = float(h['ci'].loc[player, 'upper'])
                                val = lo if metric == "ci_lower" else (up if metric == "ci_upper" else up - lo)
                    except Exception:
                        val = np.nan
                    vals.append(val)
                ax.plot(steps, vals, '-o', linewidth=1.5, label=player)
            ax.set_xticks(steps)
            ax.set_xticklabels([f"{int(s)} ({int(n) if not pd.isna(n) else 'NA'})" for s, n in zip(steps, n_list)])
            ax.set_xlabel('Update step (n_matches)')
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, ncol=2)
            right_plot.pyplot(fig, clear_figure=True)

        # General metrics over steps (x = step, ticks show (n_matches))
        st.markdown("---")
        st.subheader("General metrics over steps")
        dfm = pd.DataFrame([
            {
                'step': h.get('step', i),
                'n_matches': h.get('n_matches', np.nan),
                'spearman': h.get('spearman', np.nan),
                'kendall': h.get('kendall', np.nan),
                'mean_ci_width': (h['ci']['upper'] - h['ci']['lower']).mean() if isinstance(h.get('ci'), pd.DataFrame) else np.nan,
                'mean_ci_uncertainty': float(h['ci_uncertainty'].mean()) if isinstance(h.get('ci_uncertainty'), pd.Series) else np.nan,
                'mean_target_var_drop': float(h['target_var_drop'].mean()) if isinstance(h.get('target_var_drop'), pd.Series) else np.nan,
            }
            for i, h in enumerate(hist)
        ]).sort_values('step')
        c1, c2 = st.columns(2)
        # Spearman/Kendall
        try:
            fig1, ax1 = plt.subplots(figsize=(6.0, 3.0))
            steps = dfm['step'].values.astype(int)
            n_list = dfm['n_matches'].values
            if dfm['spearman'].notna().any():
                ax1.plot(steps, dfm['spearman'].values, '-o', label='Spearman')
            if dfm['kendall'].notna().any():
                ax1.plot(steps, dfm['kendall'].values, '-o', label='Kendall')
            ax1.set_xticks(steps)
            ax1.set_xticklabels([f"{s} ({int(n) if not pd.isna(n) else 'NA'})" for s, n in zip(steps, n_list)])
            ax1.set_xlabel('Update step (n_matches)')
            ax1.set_ylabel('Correlation')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            c1.pyplot(fig1, clear_figure=True)
        except Exception:
            pass
        # Mean CI width
        try:
            fig2, ax2 = plt.subplots(figsize=(6.0, 3.0))
            steps = dfm['step'].values.astype(int)
            n_list = dfm['n_matches'].values
            ax2.plot(steps, dfm['mean_ci_width'].values, '-o', color='#1f77b4')
            ax2.set_xticks(steps)
            ax2.set_xticklabels([f"{s} ({int(n) if not pd.isna(n) else 'NA'})" for s, n in zip(steps, n_list)])
            ax2.set_xlabel('Update step (n_matches)')
            ax2.set_ylabel('Mean CI width')
            ax2.grid(True, alpha=0.3)
            c2.pyplot(fig2, clear_figure=True)
        except Exception:
            pass


        # Ranking cube plot using history
        try:
            rankings_rows = []
            for h in hist:
                for m, r in h['elo'].items():
                    rankings_rows.append({'step': h['step'], 'model': m, 'rating': float(r)})
            rankings_df = pd.DataFrame(rankings_rows)
            ci_concat = pd.concat([h['ci'] for h in hist if isinstance(h.get('ci'), pd.DataFrame)], ignore_index=True)
            plot_ops_ranking_cubes(rankings_df, ci_concat, events_df=None, operation="interactive", true_skills=true_skills, save_path=None)
            st.pyplot(plt.gcf(), clear_figure=True)
        except Exception:
            pass


        # Save full history (all steps) with all interactive plots
        if st.button("Save full history"):
            try:
                base_dir = os.path.join(ROOT, 'interactive_ui', 'snapshots')
                os.makedirs(base_dir, exist_ok=True)
                out_dir = os.path.join(base_dir, 'history_all')
                os.makedirs(out_dir, exist_ok=True)
                # Players history
                try:
                    rows = []
                    for h in hist:
                        step = h.get('step')
                        elo_s = h.get('elo')
                        ci_h = h.get('ci')
                        if isinstance(elo_s, pd.Series):
                            for p, rating in elo_s.items():
                                lo = up = width = np.nan
                                if isinstance(ci_h, pd.DataFrame) and p in ci_h.index:
                                    lo = float(ci_h.loc[p, 'lower']); up = float(ci_h.loc[p, 'upper']); width = up - lo
                                rows.append({'step': step, 'player': p, 'pred_skill': float(rating), 'ci_lower': lo, 'ci_upper': up, 'ci_width': width})
                    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'players_history.csv'), index=False)
                except Exception:
                    pass
                # Matches reward/influence history
                try:
                    rows_r, rows_i, rows_cu, rows_tvd, rows_w = [], [], [], [], []
                    for h in hist:
                        step = h.get('step')
                        rs = h.get('reward')
                        infl = h.get('influence')
                        ciu = h.get('ci_uncertainty')
                        tvd = h.get('target_var_drop')
                        win = h.get('winners')
                        if isinstance(rs, pd.Series):
                            for mid, val in rs.items():
                                rows_r.append({'step': step, 'match_index': int(mid), 'reward': float(val) if pd.notna(val) else np.nan})
                        if isinstance(infl, pd.Series):
                            for mid, val in infl.items():
                                rows_i.append({'step': step, 'match_index': int(mid), 'influence': float(val) if pd.notna(val) else np.nan})
                        if isinstance(ciu, pd.Series):
                            for mid, val in ciu.items():
                                rows_cu.append({'step': step, 'match_index': int(mid), 'ci_uncertainty': float(val) if pd.notna(val) else np.nan})
                        if isinstance(tvd, pd.Series):
                            for mid, val in tvd.items():
                                rows_tvd.append({'step': step, 'match_index': int(mid), 'target_var_drop': float(val) if pd.notna(val) else np.nan})
                        if isinstance(win, pd.Series):
                            for mid, wv in win.items():
                                rows_w.append({'step': step, 'match_index': int(mid), 'winner': str(wv)})
                    if rows_r:
                        pd.DataFrame(rows_r).to_csv(os.path.join(out_dir, 'matches_reward_history.csv'), index=False)
                    if rows_i:
                        pd.DataFrame(rows_i).to_csv(os.path.join(out_dir, 'matches_influence_history.csv'), index=False)
                    if rows_cu:
                        pd.DataFrame(rows_cu).to_csv(os.path.join(out_dir, 'matches_ci_uncertainty_history.csv'), index=False)
                    if rows_tvd:
                        pd.DataFrame(rows_tvd).to_csv(os.path.join(out_dir, 'matches_target_var_drop_history.csv'), index=False)
                    if rows_w:
                        pd.DataFrame(rows_w).to_csv(os.path.join(out_dir, 'winners_history.csv'), index=False)
                except Exception:
                    pass
                # Selection per step
                try:
                    rows_s = []
                    for h in hist:
                        step = h.get('step')
                        sel = h.get('selection', [])
                        rows_s.append({'step': step, 'selection_csv': ','.join(map(str, sel))})
                    pd.DataFrame(rows_s).to_csv(os.path.join(out_dir, 'selection_history.csv'), index=False)
                except Exception:
                    pass
                # Plots for reward/influence mean ± std
                try:
                    # Reward
                    df_reward = pd.DataFrame([
                        {'step': h.get('step', i), 'mean_reward': float(h['reward'].mean()) if isinstance(h.get('reward'), pd.Series) else np.nan}
                        for i, h in enumerate(hist)
                    ]).sort_values('step')
                    figR, axR = plt.subplots(figsize=(6.0, 3.0))
                    axR.plot(df_reward['step'].values, df_reward['mean_reward'].values, '-o', color='#6a3d9a')
                    axR.set_xlabel('Update step'); axR.set_ylabel('Mean reward'); axR.grid(True, alpha=0.3)
                    figR.savefig(os.path.join(out_dir, 'reward_mean.png'), dpi=200, bbox_inches='tight'); plt.close(figR)
                except Exception:
                    pass
                try:
                    # Influence
                    df_infl = pd.DataFrame([
                        {'step': h.get('step', i), 'mean_influence': float(h['influence'].mean()) if isinstance(h.get('influence'), pd.Series) else np.nan}
                        for i, h in enumerate(hist)
                    ]).sort_values('step')
                    figI, axI = plt.subplots(figsize=(6.0, 3.0))
                    axI.plot(df_infl['step'].values, df_infl['mean_influence'].values, '-o', color='#e31a1c')
                    axI.set_xlabel('Update step'); axI.set_ylabel('Mean influence'); axI.grid(True, alpha=0.3)
                    figI.savefig(os.path.join(out_dir, 'influence_mean.png'), dpi=200, bbox_inches='tight'); plt.close(figI)
                except Exception:
                    pass
                try:
                    # CI-based uncertainty
                    df_ciu = pd.DataFrame([
                        {'step': h.get('step', i), 'mean_ci_uncertainty': float(h['ci_uncertainty'].mean()) if isinstance(h.get('ci_uncertainty'), pd.Series) else np.nan}
                        for i, h in enumerate(hist)
                    ]).sort_values('step')
                    figU, axU = plt.subplots(figsize=(6.0, 3.0))
                    axU.plot(df_ciu['step'].values, df_ciu['mean_ci_uncertainty'].values, '-o', color='#33a02c')
                    axU.set_xlabel('Update step'); axU.set_ylabel('Mean CI-based uncertainty'); axU.grid(True, alpha=0.3)
                    figU.savefig(os.path.join(out_dir, 'ci_uncertainty_mean.png'), dpi=200, bbox_inches='tight'); plt.close(figU)
                except Exception:
                    pass
                try:
                    # Target-specific variance drop
                    df_tvd = pd.DataFrame([
                        {'step': h.get('step', i), 'mean_target_var_drop': float(h['target_var_drop'].mean()) if isinstance(h.get('target_var_drop'), pd.Series) else np.nan}
                        for i, h in enumerate(hist)
                    ]).sort_values('step')
                    figT, axT = plt.subplots(figsize=(6.0, 3.0))
                    axT.plot(df_tvd['step'].values, df_tvd['mean_target_var_drop'].values, '-o', color='#ff7f00')
                    axT.set_xlabel('Update step'); axT.set_ylabel('Mean target variance drop'); axT.grid(True, alpha=0.3)
                    figT.savefig(os.path.join(out_dir, 'target_var_drop_mean.png'), dpi=200, bbox_inches='tight'); plt.close(figT)
                except Exception:
                    pass
                # Ranking cubes for all steps already saved can be regenerated similarly if needed
                try:
                    rankings_rows = []
                    for h in hist:
                        for m, r in h['elo'].items():
                            rankings_rows.append({'step': h['step'], 'model': m, 'rating': float(r)})
                    rankings_df = pd.DataFrame(rankings_rows)
                    ci_concat = pd.concat([h['ci'] for h in hist if isinstance(h.get('ci'), pd.DataFrame)], ignore_index=True)
                    plot_ops_ranking_cubes(rankings_df, ci_concat, events_df=None, operation="interactive", true_skills=true_skills, save_path=os.path.join(out_dir, 'ranking_cubes.png'))
                except Exception:
                    pass
                st.success(f"Saved full history to {out_dir}")
            except Exception as e:
                st.error(f"Save history failed: {e}")
        else:
            # Matches
            match_options = list(matches_df.index)
            matches_sel = left_ctrl.multiselect("Match indices", options=match_options, default=match_options[:min(3, len(match_options))])
            metric = left_ctrl.selectbox("Metric", options=["reward", "influence", "ci_uncertainty", "target_var_drop"], index=0)
            steps = [h.get('step', i) for i, h in enumerate(hist)]
            n_list = [h.get('n_matches', np.nan) for h in hist]
            fig, ax = plt.subplots(figsize=(6.5, 3.0))
            for mid in matches_sel:
                vals = []
                for h in hist:
                    val = np.nan
                    try:
                        series = h.get(metric)
                        if isinstance(series, pd.Series):
                            val = float(series.get(mid, np.nan))
                    except Exception:
                        val = np.nan
                    vals.append(val)
                ax.plot(steps, vals, '-o', linewidth=1.5, label=str(mid))
            ax.set_xticks(steps)
            ax.set_xticklabels([f"{int(s)} ({int(n) if not pd.isna(n) else 'NA'})" for s, n in zip(steps, n_list)])
            ax.set_xlabel('Update step (n_matches)')
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, ncol=2, title='match')
            right_plot.pyplot(fig, clear_figure=True)



if __name__ == "__main__":
    main()


