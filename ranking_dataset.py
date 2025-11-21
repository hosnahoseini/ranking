
import numpy as np
import pandas as pd

def generate_bt_coeffs(num_players=20, gamma=2, seed=0):
    """Generate player skill coefficients (BT parameters)."""
    rng = np.random.default_rng(seed)
    xi = rng.beta(1/gamma, 1/gamma, size=num_players)
    return xi

def simulate_match(xi, m, m_prime, rng=None, allow_ties=False):
    """Simulate one match outcome between players m and m_prime."""
    if rng is None:
        rng = np.random.default_rng()
    p_win = 1 / (1 + np.exp(xi[m_prime] - xi[m]))
    r = rng.random()
    if allow_ties:
        # Simple tie model: 10% chance of tie, otherwise BT
        if r < 0.1:
            return "tie"
        r = rng.random()
    return "m" if r < p_win else "m_prime"

def generate_arena_dataset(num_players=10, n_matches=1000, gamma=2, seed=0, allow_ties=False):
    """
    Generate a dataset in Arena-compatible format.
    Columns: model_a, model_b, winner (in {"model_a","model_b","tie"})
    """
    rng = np.random.default_rng(seed)
    xi = generate_bt_coeffs(num_players=num_players, gamma=gamma, seed=seed)

    data = []
    for _ in range(n_matches):
        m, m_prime = rng.choice(num_players, size=2, replace=False)

        outcome = simulate_match(xi, m, m_prime, rng, allow_ties=allow_ties)
        if outcome == "tie":
            data.append({
                "model_a": f"Player_{m}",
                "model_b": f"Player_{m_prime}",
                "winner": "tie"
            })
        elif outcome == "m":
            data.append({
                "model_a": f"Player_{m}",
                "model_b": f"Player_{m_prime}",
                "winner": "model_a"
            })
        else:  # outcome == "m_prime"
            data.append({
                "model_a": f"Player_{m}",
                "model_b": f"Player_{m_prime}",
                "winner": "model_b"
            })

    players_df = pd.DataFrame({
        "player_id": range(num_players),
        "skill_level": xi,
        "player_name": [f"Player_{i}" for i in range(num_players)]
    })

    matches_df = pd.DataFrame(data)
    return players_df, matches_df

def load_chatbot_arena_matches(
    split="train",
    hf_dataset_name="lmsys/chatbot_arena_conversations",
    allow_ties=True,
    max_rows=None,
) -> pd.DataFrame:
    """
    Load Chatbot Arena conversations from Hugging Face and convert to matches_df.
    Returns a DataFrame with columns: model_a, model_b, winner.
    - winner ∈ {"model_a","model_b","tie","tie (bothbad)"} (normalized)
    - If allow_ties is False, tie rows are dropped.
    Requires:
      pip install datasets
      huggingface-cli login  (if dataset access is gated)
    """
    try:
        from datasets import load_dataset  # local import to avoid hard dependency at import time
    except Exception as e:
        raise ImportError("The 'datasets' package is required to load Chatbot Arena data. Please `pip install datasets`.") from e

    ds = load_dataset(hf_dataset_name, split=split)
    df = ds.to_pandas()

    def _pick_col(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    col_a = _pick_col(["model_a", "modelA", "left_model", "model_left", "model_1", "model1"])
    col_b = _pick_col(["model_b", "modelB", "right_model", "model_right", "model_2", "model2"])
    col_w = _pick_col(["winner", "final_winner", "vote_winner", "result", "outcome"])

    if col_a is None or col_b is None:
        raise ValueError("Could not find model columns in the loaded dataset. Expected columns like 'model_a' and 'model_b'.")
    if col_w is None:
        raise ValueError("Could not find a winner column in the loaded dataset. Expected a column like 'winner'.")

    def _normalize_winner(x):
        if pd.isna(x):
            return None
        v = str(x).strip().lower()
        if v in {"model_a", "a", "left", "first", "0"}:
            return "model_a"
        if v in {"model_b", "b", "right", "second", "1"}:
            return "model_b"
        if v in {"tie", "draw", "both", "bothbad", "tie (bothbad)"}:
            return "tie (bothbad)" if "bad" in v else "tie"
        return None

    out = pd.DataFrame({
        "model_a": df[col_a].astype(str),
        "model_b": df[col_b].astype(str),
        "winner": df[col_w].map(_normalize_winner),
    })

    # Drop rows where winner could not be normalized
    out = out.dropna(subset=["winner"]).copy()

    if not allow_ties:
        out = out[~out["winner"].isin(["tie", "tie (bothbad)"])].copy()

    if max_rows is not None:
        try:
            k = int(max_rows)
            if k > 0:
                out = out.iloc[:k].copy()
        except Exception:
            pass

    out.reset_index(drop=True, inplace=True)
    return out

def load_arena_hp55k_matches(
    split: str = "train",
    hf_dataset_name: str = "lmarena-ai/arena-human-preference-55k",
    allow_ties: bool = True,
    max_rows: int = None,
    streaming: bool = True,
) -> pd.DataFrame:
    """
    Load lmarena-ai/arena-human-preference-55k and convert to matches_df with:
      columns: model_a, model_b, winner in {"model_a","model_b","tie"}
    The dataset stores one-hot winner flags: winner_model_a, winner_model_b, winner_tie.
    Rows with ambiguous/missing flags are dropped. If allow_ties is False, ties are removed.
    """
    try:
        from datasets import load_dataset
    except Exception as e:
        raise ImportError("The 'datasets' package is required. Please `pip install datasets`.") from e

    required = ["model_a", "model_b", "winner_model_a", "winner_model_b", "winner_tie"]

    # Fast path: try streaming to avoid downloading large prompt/response text
    if streaming:
        try:
            ids = load_dataset(hf_dataset_name, split=split, streaming=True)
            rows = []
            limit = max_rows if (max_rows is not None and max_rows > 0) else None
            for i, r in enumerate(ids):
                a = r.get("model_a")
                b = r.get("model_b")
                wa = int(r.get("winner_model_a", 0) or 0)
                wb = int(r.get("winner_model_b", 0) or 0)
                wt = int(r.get("winner_tie", 0) or 0)
                if a is None or b is None:
                    continue
                total = wa + wb + wt
                if total != 1:
                    continue
                if wa == 1:
                    w = "model_a"
                elif wb == 1:
                    w = "model_b"
                else:
                    w = "tie"
                if not allow_ties and w == "tie":
                    pass
                else:
                    rows.append({"model_a": str(a), "model_b": str(b), "winner": w})
                if limit is not None and len(rows) >= limit:
                    break
            if rows:
                return pd.DataFrame(rows).reset_index(drop=True)
            # Fall through to non-streaming if no rows collected
        except Exception:
            # Fall back to non-streaming
            pass

    # Non-streaming fallback: load once, then drop heavy columns before materializing
    ds = load_dataset(hf_dataset_name, split=split)
    missing = [c for c in required if c not in ds.column_names]
    if missing:
        raise ValueError(f"Missing expected columns in {hf_dataset_name}: {missing}")
    keep = required
    drop_cols = [c for c in ds.column_names if c not in keep]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)
    df = ds.to_pandas()

    def _pick_winner(row) -> str | None:
        a = int(row.get("winner_model_a", 0) or 0)
        b = int(row.get("winner_model_b", 0) or 0)
        t = int(row.get("winner_tie", 0) or 0)
        total = a + b + t
        if total != 1:
            return None
        if a == 1:
            return "model_a"
        if b == 1:
            return "model_b"
        return "tie"

    out = pd.DataFrame({
        "model_a": df["model_a"].astype(str),
        "model_b": df["model_b"].astype(str),
    })
    out["winner"] = df.apply(_pick_winner, axis=1)
    out = out.dropna(subset=["winner"]).copy()

    if not allow_ties:
        out = out[out["winner"] != "tie"].copy()

    if max_rows is not None:
        try:
            k = int(max_rows)
            if k > 0:
                out = out.iloc[:k].copy()
        except Exception:
            pass

    out.reset_index(drop=True, inplace=True)
    return out
