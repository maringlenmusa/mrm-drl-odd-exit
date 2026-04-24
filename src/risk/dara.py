"""
DARA: scene/environmental risk (simplified, TTC-based).

Supervisor: DARA out of focus; use simple placeholder (e.g. TTC-based).
Returns 1.0 when TTC is very low (danger), 0.0 when TTC is high (safe).
"""

import math
from typing import Union


def compute_dara(
    min_ttc_s: Union[float, type(math.inf)],
    *,
    ttc_safe_s: float = 5.0,
    ttc_critical_s: float = 2.0,
) -> float:
    """
    Compute DARA (scene risk) from minimum TTC.

    Linear interpolation: 1.0 when TTC <= ttc_critical_s, 0.0 when TTC >= ttc_safe_s.

    Args:
        min_ttc_s: Minimum TTC among relevant actors (seconds), or math.inf if none.
        ttc_safe_s: TTC above which risk is 0 (default 5 s).
        ttc_critical_s: TTC below which risk is 1 (default 2 s).

    Returns:
        DARA value in [0.0, 1.0].
    """
    if min_ttc_s == math.inf or min_ttc_s >= ttc_safe_s:
        return 0.0
    if min_ttc_s <= ttc_critical_s:
        return 1.0
    # Linear between ttc_critical_s and ttc_safe_s
    span = ttc_safe_s - ttc_critical_s
    if span <= 0:
        return 1.0
    return 1.0 - (min_ttc_s - ttc_critical_s) / span
