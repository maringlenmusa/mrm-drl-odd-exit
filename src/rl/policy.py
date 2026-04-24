"""
Policy/model helpers for Phase 3 training.

Stable-Baselines3 PPO with architecture required by To_Do:
  - net_arch = [256, 256]
  - activation_fn = ELU
"""

from __future__ import annotations

import os
from typing import Optional

from torch import nn
from stable_baselines3 import PPO


def create_model(env, cfg: dict, *, tensorboard_log: Optional[str] = None) -> PPO:
    """
    Create PPO model for OddExitEnv.

    Reads hyperparameters from cfg["training"] with safe defaults.
    """
    tr = cfg.get("training", {})
    seed = cfg.get("run", {}).get("seed", None)
    policy_kwargs = {
        "net_arch": [256, 256],
        "activation_fn": nn.ELU,
    }

    model = PPO(
        policy=tr.get("policy", "MlpPolicy"),
        env=env,
        learning_rate=float(tr.get("learning_rate", 3e-4)),
        n_steps=int(tr.get("n_steps", 256)),
        batch_size=int(tr.get("batch_size", 64)),
        n_epochs=int(tr.get("n_epochs", 10)),
        gamma=float(tr.get("gamma", 0.99)),
        gae_lambda=float(tr.get("gae_lambda", 0.95)),
        clip_range=float(tr.get("clip_range", 0.2)),
        ent_coef=float(tr.get("ent_coef", 0.0)),
        vf_coef=float(tr.get("vf_coef", 0.5)),
        max_grad_norm=float(tr.get("max_grad_norm", 0.5)),
        verbose=int(tr.get("verbose", 1)),
        seed=seed,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
    )
    return model


def load_model(model_path: str, env=None) -> PPO:
    """Load a saved PPO policy.zip model."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return PPO.load(model_path, env=env)

