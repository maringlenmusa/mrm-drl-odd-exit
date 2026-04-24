"""
Training callbacks for Phase 3.

- EpisodeCollector: extracts finished-episode summaries from SB3 step infos.
- TrainingCallbacks: periodic sampler updates + model checkpoints.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from stable_baselines3.common.callbacks import BaseCallback


class EpisodeCollector:
    """Collect terminal episode summaries emitted by the environment."""

    def __init__(self):
        self._episodes: List[Dict] = []

    def on_step_infos(self, infos, dones) -> int:
        """
        Consume vectorized env infos/dones and store terminal summaries.
        Returns number of episodes added.
        """
        if infos is None or dones is None:
            return 0
        added = 0
        for i, done in enumerate(dones):
            if not done:
                continue
            info = infos[i] if i < len(infos) else {}
            self._episodes.append({
                "scenario_path": info.get("scenario_path"),
                "max_risk": float(info.get("max_risk", 0.0)),
                "collision": bool(info.get("collision", False)),
                "success_mrc": bool(info.get("mrc_reached", False)),
                "timeout": bool(info.get("timeout", False)),
                "episode_id": info.get("episode_id"),
            })
            added += 1
        return added

    def pop_all(self) -> List[Dict]:
        out = list(self._episodes)
        self._episodes = []
        return out

    @property
    def size(self) -> int:
        return len(self._episodes)


class TrainingCallbacks(BaseCallback):
    """
    SB3 callback:
    - collects completed episodes
    - updates scenario sampler every N episodes
    - saves periodic checkpoints
    """

    def __init__(
        self,
        *,
        run_dir: str,
        episode_collector: Optional[EpisodeCollector] = None,
        scenario_sampler=None,
        update_every_episodes: int = 10,
        checkpoint_every_steps: int = 0,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.run_dir = run_dir
        self.episode_collector = episode_collector or EpisodeCollector()
        self.scenario_sampler = scenario_sampler
        self.update_every_episodes = max(1, int(update_every_episodes))
        self.checkpoint_every_steps = max(0, int(checkpoint_every_steps))
        self._episodes_since_update = 0
        self._last_checkpoint_step = 0

    def _on_training_start(self) -> None:
        os.makedirs(os.path.join(self.run_dir, "artifacts"), exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        added = self.episode_collector.on_step_infos(infos, dones)
        self._episodes_since_update += added

        if (
            self.scenario_sampler is not None
            and self._episodes_since_update >= self.update_every_episodes
        ):
            episodes = self.episode_collector.pop_all()
            if episodes:
                self.scenario_sampler.update_weights(episodes)
                if self.verbose:
                    print(
                        f"[TrainingCallbacks] Updated sampler weights "
                        f"with {len(episodes)} episode summaries."
                    )
            self._episodes_since_update = 0

        if (
            self.checkpoint_every_steps > 0
            and (self.num_timesteps - self._last_checkpoint_step) >= self.checkpoint_every_steps
        ):
            ckpt_path = os.path.join(
                self.run_dir, "artifacts", f"checkpoint_{self.num_timesteps}.zip"
            )
            self.model.save(ckpt_path)
            self._last_checkpoint_step = self.num_timesteps
            if self.verbose:
                print(f"[TrainingCallbacks] Saved checkpoint: {ckpt_path}")

        return True

