"""
Risk combiner: combine DRF and DARA into a single risk score in [0, 1].
"""

from typing import Optional


def compute_risk(
    drf: float,
    dara: float,
    *,
    weight_drf: float = 0.5,
    weight_dara: float = 0.5,
) -> float:
    """
    Combine DRF and DARA into total risk in [0.0, 1.0].

    Weights are normalized so weight_drf + weight_dara can be any positive;
    risk = (weight_drf * drf + weight_dara * dara) / (weight_drf + weight_dara).
    If both weights 0, returns max(drf, dara).

    Args:
        drf: Ego motion risk from compute_drf().
        dara: Scene risk from compute_dara().
        weight_drf: Weight for DRF (default 0.5).
        weight_dara: Weight for DARA (default 0.5).

    Returns:
        Combined risk in [0.0, 1.0]. Increases when either drf or dara increases.
    """
    total_w = weight_drf + weight_dara
    if total_w <= 0:
        return max(0.0, min(1.0, max(drf, dara)))
    r = (weight_drf * drf + weight_dara * dara) / total_w
    return max(0.0, min(1.0, r))
