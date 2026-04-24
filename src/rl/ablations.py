"""
Phase 3 ablation flags.

Central place to parse and normalize ablation toggles from config so the rest
of the RL pipeline can use one consistent flag format.

Supported config styles:
1) Positive flags:
   ablations:
     use_risk_features: true
     use_risk_penalty_in_reward: true
     use_shield: true

2) Negative flags:
   ablations:
     no_risk_features: false
     no_risk_penalty: false
     no_shield: false

Returned flags include both representations for convenience.
"""

from typing import Dict, Any


def _bool(value: Any, default: bool) -> bool:
    """Coerce config value to bool with a safe default."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
        if v in ("0", "false", "no", "off"):
            return False
    return default


def apply_ablations(cfg: dict) -> Dict[str, bool]:
    """
    Read ablation flags from cfg and return normalized flags.

    Returns a dict with both:
    - positive names: use_risk_features, use_risk_penalty_in_reward, use_shield
    - negative names: no_risk_features, no_risk_penalty, no_shield

    Defaults are "full system enabled" (no ablation):
      use_risk_features=True, use_risk_penalty_in_reward=True, use_shield=True
    """
    ab = cfg.get("ablations", {}) or {}

    # Positive form defaults (full system).
    use_risk_features = _bool(ab.get("use_risk_features"), True)
    use_risk_penalty = _bool(ab.get("use_risk_penalty_in_reward"), True)
    use_shield = _bool(ab.get("use_shield"), True)

    # Negative form overrides if provided explicitly.
    if "no_risk_features" in ab:
        use_risk_features = not _bool(ab.get("no_risk_features"), False)
    if "no_risk_penalty" in ab:
        use_risk_penalty = not _bool(ab.get("no_risk_penalty"), False)
    if "no_shield" in ab:
        use_shield = not _bool(ab.get("no_shield"), False)

    return {
        "use_risk_features": use_risk_features,
        "use_risk_penalty_in_reward": use_risk_penalty,
        "use_shield": use_shield,
        # Convenience aliases used by current env/reward modules.
        "no_risk_features": not use_risk_features,
        "no_risk_penalty": not use_risk_penalty,
        "no_shield": not use_shield,
    }

