"""
Phase 3: Reinforcement Learning package.

Modules:
    env_odd_exit    - Gym-style environment wrapping the Phase 2 simulation
    reward          - Reward function (compute_reward)
    ablations       - Ablation flag application (apply_ablations)
    scenario_sampler- Risk-prioritized scenario sampler
    policy          - SB3 PPO model factory (create_model, load_model)
    callbacks       - Training callbacks (EpisodeCollector, TrainingCallbacks)
    train           - Training entry point (train)
    evaluate        - Evaluation entry point (evaluate)
"""
