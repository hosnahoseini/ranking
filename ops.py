import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict


class OperationStrategy:
    """
    Strategy interface for selecting matches to add/remove/flip.
    Implementations should return a list of integer indices into matches_df.
    """

    def select_for_add(self, pool_indices: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def select_for_remove(self, current_indices: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def select_for_flip(self, current_df: pd.DataFrame, current_indices: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


class RandomStrategy(OperationStrategy):
    """
    Random selection for add/remove/flip operations.
    - add: sample uniformly from pool_indices
    - remove: sample uniformly from current_indices
    - flip: sample uniformly from non-tie rows in current_df[current_indices]
    """

    def select_for_add(self, pool_indices: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        if pool_indices.size == 0 or k <= 0:
            return np.array([], dtype=int)
        k_eff = int(min(k, pool_indices.size))
        return rng.choice(pool_indices, size=k_eff, replace=False)

    def select_for_remove(self, current_indices: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        if current_indices.size == 0 or k <= 0:
            return np.array([], dtype=int)
        k_eff = int(min(k, current_indices.size))
        return rng.choice(current_indices, size=k_eff, replace=False)

    def select_for_flip(self, current_df: pd.DataFrame, current_indices: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        if current_indices.size == 0 or k <= 0:
            return np.array([], dtype=int)
        sub = current_df.loc[current_indices]
        non_tie_idx = sub[~sub["winner"].isin(["tie", "tie (bothbad)"])].index.values
        if non_tie_idx.size == 0:
            return np.array([], dtype=int)
        k_eff = int(min(k, non_tie_idx.size))
        return rng.choice(non_tie_idx, size=k_eff, replace=False)


def add_matches_random(
    matches_df: pd.DataFrame,
    chosen_indices: np.ndarray,
    k: int,
    rng: np.random.Generator,
    strategy: Optional[OperationStrategy] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select k indices from the remaining pool (not in chosen_indices) to add.
    Returns (new_indices_to_add, updated_chosen_indices).
    """
    if strategy is None:
        strategy = RandomStrategy()
    chosen_set = set(map(int, chosen_indices.tolist()))
    pool = np.array([i for i in matches_df.index.values if i not in chosen_set], dtype=int)
    sel = strategy.select_for_add(pool, k, rng)
    if sel.size == 0:
        return sel, chosen_indices
    updated = np.unique(np.concatenate([chosen_indices, sel])).astype(int)
    return sel, updated


def remove_matches_random(
    chosen_indices: np.ndarray,
    k: int,
    rng: np.random.Generator,
    strategy: Optional[OperationStrategy] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select k indices from chosen_indices to remove.
    Returns (removed_indices, updated_chosen_indices).
    """
    if strategy is None:
        strategy = RandomStrategy()
    sel = strategy.select_for_remove(chosen_indices, k, rng)
    if sel.size == 0:
        return sel, chosen_indices
    mask_keep = ~np.isin(chosen_indices, sel)
    updated = chosen_indices[mask_keep]
    return sel, updated


def flip_matches_random(
    current_df: pd.DataFrame,
    chosen_indices: np.ndarray,
    k: int,
    rng: np.random.Generator,
    strategy: Optional[OperationStrategy] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Select up to k non-tie matches within current_df[chosen_indices] and flip winner.
    Returns (flipped_indices, updated_current_df).
    Notes:
      - Ties are skipped by default.
      - Flipping means 'model_a' -> 'model_b' and vice versa.
    """
    if strategy is None:
        strategy = RandomStrategy()
    sel = strategy.select_for_flip(current_df, chosen_indices, k, rng)
    if sel.size == 0:
        return sel, current_df
    updated = current_df.copy()
    sub = updated.loc[sel]
    flipped = sub["winner"].map({"model_a": "model_b", "model_b": "model_a"})
    updated.loc[sel, "winner"] = flipped.values
    return sel, updated


