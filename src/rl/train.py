"""
Phase 3 RL training entrypoint.

Keeps Phase 2 simulation stack intact by training on OddExitEnv directly.
Adds:
- ablation flags
- risk-prioritized scenario sampling
- SB3 PPO model creation/saving
"""

from __future__ import annotations

import json
import os
from typing import List

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym

from stable_baselines3.common.vec_env import DummyVecEnv

from src.logger import make_run_dir
from src.rl.env_odd_exit import OddExitEnv
from src.rl.ablations import apply_ablations
from src.rl.scenario_sampler import ScenarioSampler
from src.rl.policy import create_model, load_model
from src.rl.callbacks import EpisodeCollector, TrainingCallbacks
from src.rl.episode_variation import Phase2StyleVariationEnv


class _TinyTrainEnv(gym.Env):
    """Fast mock env used only for local smoke tests (no simulator)."""

    def __init__(self):
        super().__init__()
        import numpy as np
        self.np = np
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(10,), dtype=np.float32
        )
        self._t = 0
        self._scenario = "mock://scenario"

    def reset(self, *, seed=None, options=None):
        self._t = 0
        obs = self.np.zeros((10,), dtype=self.np.float32)
        return obs, {}

    def step(self, action):
        self._t += 1
        obs = self.np.random.rand(10).astype(self.np.float32)
        reward = float(self.np.random.uniform(-0.1, 1.0))
        terminated = self._t >= 10
        truncated = False
        info = {
            "scenario_path": self._scenario,
            "max_risk": float(self.np.random.uniform(0.0, 1.0)),
            "collision": False,
            "mrc_reached": terminated,
            "timeout": False,
        }
        return obs, reward, terminated, truncated, info


class SampledScenarioEnv(gym.Wrapper):
    """
    Wrapper that injects scenario_sampler.sample() into env.reset(options=...).
    This enables risk-prioritized sampling without touching simulation logic.
    """

    def __init__(self, env: OddExitEnv, sampler: ScenarioSampler):
        super().__init__(env)
        self.sampler = sampler
        self._last_scenario = None

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        self._last_scenario = self.sampler.sample()
        options["scenario_path"] = self._last_scenario
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        out = self.env.step(action)
        # gymnasium: obs, reward, terminated, truncated, info
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            info = dict(info)
            info["scenario_path"] = self._last_scenario
            return obs, reward, terminated, truncated, info
        # gym: obs, reward, done, info
        obs, reward, done, info = out
        info = dict(info)
        info["scenario_path"] = self._last_scenario
        return obs, reward, done, info


def _get_training_scenarios(cfg: dict) -> List[str]:
    sampler_cfg = cfg.get("scenario_sampler", {}) or {}
    scenarios = sampler_cfg.get("scenarios")
    if isinstance(scenarios, str):
        scenarios = [scenarios]
    if scenarios:
        return [str(s) for s in scenarios]
    return [str(cfg.get("sim", {}).get("scenario_path", ""))]


def train(cfg: dict, run_dir: str | None = None) -> str:
    """
    Train PPO on OddExitEnv.

    Returns:
        run_dir where artifacts/logs were saved.
    """
    if run_dir is None:
        cfg = dict(cfg)
        cfg.setdefault("run", {})
        cfg["run"]["name"] = cfg["run"].get("name", "phase3_train")
        run_dir = make_run_dir(cfg)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "artifacts"), exist_ok=True)

    flags = apply_ablations(cfg)
    tr = cfg.get("training", {})
    rp = cfg.get("risk_priority", {})
    scenarios = _get_training_scenarios(cfg)

    sampler = ScenarioSampler(
        scenarios=scenarios,
        seed=cfg.get("run", {}).get("seed"),
        collision_weight=float(rp.get("collision_weight", 2.0)),
        risk_weight=float(rp.get("risk_weight", 1.0)),
        failure_weight=float(rp.get("failure_weight", 1.0)),
        min_weight=float(rp.get("min_weight", 0.1)),
    ) if bool(rp.get("sampler_enabled", True)) else None

    smoke_mock = bool(tr.get("smoke_test_mock_env", False))

    def _make_env():
        if smoke_mock:
            return _TinyTrainEnv()
        base_env = OddExitEnv(cfg, run_dir=run_dir, ablations=flags)
        # First apply Phase-2-style speed/gap/distance variation every episode.
        # This keeps training distribution aligned with phase2_baseline and
        # phase2_short_front_gap runs (prevents fixed-scenario collapse).
        varied_env = Phase2StyleVariationEnv(
            base_env,
            cfg=cfg,
            seed=cfg.get("run", {}).get("seed"),
        )
        if sampler is None:
            return varied_env
        # Then apply risk-prioritized scenario-path sampling.
        return SampledScenarioEnv(varied_env, sampler)

    vec_env = DummyVecEnv([_make_env])

    use_tensorboard = bool(tr.get("use_tensorboard", False))
    tb_log_dir = os.path.join(run_dir, "tb") if use_tensorboard else None
    model = create_model(vec_env, cfg, tensorboard_log=tb_log_dir)

    collector = EpisodeCollector()
    callbacks = TrainingCallbacks(
        run_dir=run_dir,
        episode_collector=collector,
        scenario_sampler=sampler,
        update_every_episodes=int(rp.get("update_every_episodes", 10)),
        checkpoint_every_steps=int(tr.get("checkpoint_every_steps", 0)),
        verbose=int(tr.get("verbose", 1)),
    )

    total_timesteps = int(tr.get("total_timesteps", 1000))
    artifacts_dir = os.path.join(run_dir, "artifacts")
    status_path = os.path.join(artifacts_dir, "training_status.json")
    interrupted = False

    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
    except KeyboardInterrupt:
        interrupted = True
        print(
            "\n[Phase3 Train] Interrupted (Ctrl+C). Saving current policy and status..."
        )

    model_path = os.path.join(artifacts_dir, "policy.zip")
    if interrupted:
        # Partial run: still valid for eval / inspection; name avoids confusion with full train.
        interrupt_path = os.path.join(artifacts_dir, "policy_interrupted.zip")
        model.save(interrupt_path)
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "completed": False,
                    "reason": "keyboard_interrupt",
                    "num_timesteps": int(getattr(model, "num_timesteps", 0)),
                    "total_timesteps_requested": total_timesteps,
                    "policy_path": "policy_interrupted.zip",
                },
                f,
                indent=2,
            )
        try:
            vec_env.close()
        except Exception:
            pass
        print(f"[Phase3 Train] Partial policy saved to: {interrupt_path}")
        print(f"[Phase3 Train] Status written to: {status_path}")
        print(
            "[Phase3 Train] Tip: periodic checkpoints are in artifacts/checkpoint_*.zip "
            "if training.checkpoint_every_steps > 0. Logs in this run_dir are kept."
        )
        return run_dir

    model.save(model_path)
    _ = load_model(model_path, env=vec_env)  # save/load smoke check
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "completed": True,
                "num_timesteps": int(getattr(model, "num_timesteps", 0)),
                "total_timesteps_requested": total_timesteps,
                "policy_path": "policy.zip",
            },
            f,
            indent=2,
        )
    try:
        vec_env.close()
    except Exception:
        pass
    print(f"[Phase3 Train] Done. Policy saved to: {model_path}")
    return run_dir

