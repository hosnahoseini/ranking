
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
