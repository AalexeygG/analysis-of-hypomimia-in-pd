import numpy as np
from scipy.stats import mannwhitneyu
from typing import Optional


def mann_whitney_stats(
    bp_features: list[dict[str, float]],
    kg_features: list[dict[str, float]],
    feature_names: list[str],
) -> dict[str, dict[str, Optional[float]]]:
    """
    Возвращает для каждого признака U и p_value.
    Если данных недостаточно (менее 2 наблюдений в любой группе) -> None.
    """
    out: dict[str, dict[str, Optional[float]]] = {}

    for name in feature_names:
        bp = np.array([f.get(name) for f in bp_features if name in f], dtype=float)
        kg = np.array([f.get(name) for f in kg_features if name in f], dtype=float)

        if len(bp) < 2 or len(kg) < 2:
            out[name] = {"U": None, "p_value": None}
            continue

        u, p = mannwhitneyu(bp, kg, alternative="two-sided")
        out[name] = {"U": float(u), "p_value": float(p)}

    return out
