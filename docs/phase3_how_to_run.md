# Phase 3 — How to Run, What to Check

Everything you need to run Phase 3 RL training, inspect a policy, and understand the results.

---

## Quick reference

| What | Command | Time |
|---|---|---|
| Watch sim UI (no training) | `py -3 -u -m src.main --phase phase2 --config configs/phase2.yaml` | real-time |
| Smoke test (verify it works) | `py -3 -u -m src.main --phase phase3_train --config configs/phase3_train.yaml` with `total_timesteps: 512` | ~90 min |
| Full training (overnight) | same command with `total_timesteps: 5000` | ~14 hours |
| Thesis training | same command with `total_timesteps: 30000` | ~3.5 days |
| **Evaluate a trained policy** | `py -3 -u -m src.main --phase phase3_eval --config configs/phase3_eval.yaml` | ~2–3 hours (100 eps) |
| Inspect a trained policy (quick) | `py -3 scripts/inspect_policy.py --policy logs/runs/.../artifacts/policy.zip --n_episodes 30` | ~10 min |
| Inspect with UI | add `--ui` to inspect command | slower |

---

## 1. Config switches

Everything is controlled from `configs/phase3_train.yaml`.

### UI on/off

```yaml
sim:
  headless: false            # false = show window
  esmini_lib_show_ui: true   # true = open esmini viewer per episode
```

Set both to `true` to watch episodes live. Set both to `false` / `true` → `false` for headless fast training.

> **Note:** UI mode is 2–3× slower. Use it for watching, not for long training runs.

### Training duration

```yaml
training:
  total_timesteps: 512     # smoke test  (~90 min)
  total_timesteps: 5000    # full run    (~14 hours)
  total_timesteps: 30000   # thesis run  (~3.5 days)
  checkpoint_every_steps: 1000   # save checkpoint every N episodes
```

### Checkpoints

With `checkpoint_every_steps: 1000`, you get:
- `artifacts/checkpoint_1000.zip`
- `artifacts/checkpoint_2000.zip`
- …

You can stop training early with **Ctrl+C** — it automatically saves `artifacts/policy_interrupted.zip`.

---

## 2. Running training

```bash
py -3 -u -m src.main --phase phase3_train --config configs/phase3_train.yaml
```

### What you see in the console

```
Starting phase3_train run: phase3_train
Using cpu device
[OddExitEnv] state_source=esmini — esminiLib in-process.
[ODD EXIT] Triggered at sim_time_s=7.05  (trigger_time_s=7.0)
[MRM DECISION] ROAD_SHOULDER_STOP (proposed: ROAD_SHOULDER_STOP) at sim_t=7.05s
[MRC REACHED] at sim_time_s=14.20
...
| total_timesteps | 256 |        ← first PPO update after 256 episodes
| policy_gradient_loss | -0.009 |
...
[Phase3 Train] Done. Policy saved to: logs/runs/.../artifacts/policy.zip
```

### What is saved

```
logs/runs/<timestamp>_phase3_train/
├── episodes.csv                 ← one row per episode (main results table)
├── steps_episode_N.csv          ← sim step data per episode
├── run.log                      ← events (ODD exit, MRM decisions, collisions)
├── episode.json                 ← last episode summary
└── artifacts/
    ├── policy.zip               ← trained model (use this for evaluation)
    ├── checkpoint_1000.zip      ← periodic saves (if checkpoint_every_steps > 0)
    └── training_status.json     ← completed/interrupted + num_timesteps
```

---

## 3. What to check in `episodes.csv`

Open `logs/runs/<timestamp>/episodes.csv`. Each row is one episode.

### Key columns

| Column | What it means |
|---|---|
| `success_mrc` | True = ego braked to full stop (MRC reached) — **the goal** |
| `collision` | True = crash happened |
| `last_sim_time_s` | Sim time when episode ended |
| `primary_mrm` | MRM the agent used: 0=STRAIGHT_STOP, 1=IN_LANE_STOP, 2=ROAD_SHOULDER_STOP |
| `shoulder_available` | True = rear gap was large enough to use shoulder |
| `total_reward` | Total reward for episode |
| `reward_terminal` | Bonus/penalty at episode end (±collision/MRC/shoulder) |

### Identify before-ODD vs post-ODD collisions

**This is critical.** Many episodes have `collision=True` but the crash happened during the approach — *before* the agent even made a decision.

| Condition | Meaning |
|---|---|
| `collision=True` AND `last_sim_time_s < 7.0` (trigger time) | **Before-ODD collision** — ego hit lead car during approach. Agent had NO chance to act. Not the agent's fault. |
| `collision=True` AND `last_sim_time_s >= 7.0` | **Post-ODD collision** — agent chose MRM but it wasn't enough to stop. Agent's fault. |
| `success_mrc=True` AND `last_sim_time_s >= 7.0` | **Success** — agent braked to MRC. |

### The REAL metric (filter before-ODD)

Run this quick check:

```bash
py -3 -c "
import csv
rows = list(csv.DictReader(open('logs/runs/<your_run>/episodes.csv')))
trigger = 7.0
before_odd = [r for r in rows if r['collision']=='True' and float(r['last_sim_time_s'] or 0) < trigger]
valid = [r for r in rows if float(r['last_sim_time_s'] or 0) >= trigger]
v_mrc = sum(1 for r in valid if r['success_mrc']=='True')
print('Before-ODD collisions (not agents fault):', len(before_odd))
print('Valid episodes (agent acted):             ', len(valid))
print('MRC rate on valid episodes:               ', v_mrc, '/', len(valid), '=', 100*v_mrc//len(valid) if valid else 0, '%')
"
```

### What to look for across training

Split episodes into 4 blocks and compare:

```bash
py -3 -c "
import csv
rows = list(csv.DictReader(open('logs/runs/<your_run>/episodes.csv')))
trigger = 7.0
block = len(rows) // 4
def stats(blk, label):
    n = len(blk)
    valid = [r for r in blk if float(r['last_sim_time_s'] or 0) >= trigger]
    v = len(valid)
    mrc = sum(1 for r in valid if r['success_mrc']=='True')
    avg_r = sum(float(r['total_reward']) for r in valid)/v if v else 0
    print('%-25s  valid=%3d  MRC=%3d%%  avg_r=%6.2f  MRM0=%d MRM1=%d MRM2=%d' % (
        label, v, 100*mrc//v if v else 0, avg_r,
        sum(1 for r in blk if r['primary_mrm']=='0'),
        sum(1 for r in blk if r['primary_mrm']=='1'),
        sum(1 for r in blk if r['primary_mrm']=='2')))
stats(rows[0:block],           'ep 1-%d (random)' % block)
stats(rows[block:2*block],     'ep %d-%d (1st update)' % (block, 2*block))
stats(rows[2*block:3*block],   'ep %d-%d (2nd batch)' % (2*block, 3*block))
stats(rows[3*block:],          'ep %d-%d (final)' % (3*block, len(rows)))
"
```

**Policy is learning if:**
- MRC rate on valid episodes goes UP across blocks (e.g., 50% → 70% → 80%)
- Average reward goes UP
- When `shoulder_available=True`, MRM2 (ROAD_SHOULDER_STOP) usage increases

---

## 4. Inspecting the trained policy

After training, run `inspect_policy.py` to see what the policy actually decides:

```bash
# Headless (fast)
py -3 scripts/inspect_policy.py \
    --policy logs/runs/<timestamp>/artifacts/policy.zip \
    --n_episodes 30

# With UI (watch each episode in esmini viewer)
py -3 scripts/inspect_policy.py \
    --policy logs/runs/<timestamp>/artifacts/policy.zip \
    --n_episodes 20 \
    --ui
```

### What the output looks like

```
ep   1/ 30  v=24.3m/s  fg=45.2m  rg=28.1m  risk=0.01  shoulder=True   → ROAD_SHOULDER_STOP     [MRC    ]  r=13.45
ep   2/ 30  v=22.1m/s  fg= 6.8m  rg=41.2m  risk=0.92  shoulder=True   → STRAIGHT_STOP          [COLLISION]  r=-100.51
ep   3/ 30  v=23.0m/s  fg=32.0m  rg=20.5m  risk=0.15  shoulder=False  → IN_LANE_STOP           [MRC    ]  r=9.90
```

Column meanings: `v` = ego speed at ODD exit, `fg` = front gap, `rg` = rear gap, `risk` = risk_total, `shoulder` = was shoulder available, `→` = MRM the policy chose, `[...]` = episode outcome, `r` = total reward.

### Summary section

```
POLICY INSPECTION SUMMARY  (30 episodes)
Before-ODD collision : 8 (26%)  ← approach too tight, agent had no chance
─── episodes where agent actually acted ──────────
Valid episodes       : 22
MRC rate (REAL)      : 17/22 = 77%   ← watch this go up with training
Post-ODD collision   :  4/22 = 18%
Avg reward (valid)   : 8.43

When shoulder_available=True:
  ROAD_SHOULDER_STOP  : 11/14 (78%)   ← ideal: high means policy learned shoulder is best
  IN_LANE_STOP        :  2/14
  STRAIGHT_STOP       :  1/14

When shoulder_available=False:
  IN_LANE_STOP        :  5/ 8 (62%)   ← ideal: don't pull over when shoulder not safe
  STRAIGHT_STOP       :  3/ 8
```

### What a trained policy looks like

| Metric | Random (512 eps) | Trained (5,000+ eps) |
|---|---|---|
| MRC rate (valid eps) | ~50–60% | 75–85% |
| When shoulder=True → MRM2 | ~33% (random) | 70–90% |
| When shoulder=False → MRM0/1 | ~67% (random) | 80–95% |

---

## 5. Reward breakdown — what drives each episode

Each episode in `episodes.csv` has these reward columns:

| Column | What it is | Value |
|---|---|---|
| `total_reward` | Sum of everything | varies |
| `reward_dense` | Dense per-step penalties (risk + step cost) | small negative |
| `reward_terminal` | Bonus/penalty at end | big number |
| `reward_collision_penalty` | −100 if collision | 0 or −100 |
| `reward_success_bonus` | +10 if MRC reached | 0 or +10 |
| `reward_shoulder_bonus` | +5 if shoulder used correctly | 0 or +5 |
| `reward_lane_block_penalty` | −2 if STRAIGHT_STOP when shoulder available | 0 or −2 |

**Ideal episode**: MRC + shoulder → total ≈ **+13 to +15**  
**Worst episode**: collision → total ≈ **−100**

---

## 6. Full run checklist

Before starting overnight training:

- [ ] `configs/phase3_train.yaml` has `headless: true`, `esmini_lib_show_ui: false`
- [ ] `total_timesteps: 5000`
- [ ] `checkpoint_every_steps: 1000`
- [ ] `gap_at_trigger_min_m: 25.0` in `short_front_gap_coverage` (reduces before-ODD collisions)
- [ ] Previous Python processes are dead: `Get-Process py,python | Stop-Process -Force`

Start:
```bash
py -3 -u -m src.main --phase phase3_train --config configs/phase3_train.yaml
```

Stop safely: **Ctrl+C** — saves `policy_interrupted.zip` automatically.

Morning check:
```bash
py -3 scripts/inspect_policy.py \
    --policy logs/runs/<latest>/artifacts/policy.zip \
    --n_episodes 30
```

---

## 8. Evaluating a trained policy (official results)

`inspect_policy.py` is for quick checks. For the **official thesis results** (the number you put in the comparison table) use the evaluation system.

### Step 1 — set the policy path in `configs/phase3_eval.yaml`

```yaml
evaluation:
  policy_path: "logs/runs/2026-03-25_14-49-41_phase3_train/artifacts/policy.zip"
  n_episodes: 100        # 100 gives stable statistics
  deterministic: true    # always pick the best action (not random exploration)
```

To watch episodes in the UI, also set:
```yaml
sim:
  headless: false
  esmini_lib_show_ui: true
```

### Step 2 — run

```bash
py -3 -u -m src.main --phase phase3_eval --config configs/phase3_eval.yaml
```

### What you see in the console

```
[Phase3 Eval] Policy  : logs/runs/.../policy.zip
[Phase3 Eval] Episodes: 100  deterministic=True

  ep   1/100  shoulder=True   → ROAD_SHOULDER_STOP     [MRC           ]  r= 13.45
  ep   2/100  shoulder=False  → IN_LANE_STOP           [MRC           ]  r=  9.90
  ep   3/100  shoulder=True   → STRAIGHT_STOP          [BEFORE-ODD-COLL]  r=-100.51
  ...

==============================================================
  EVALUATION SUMMARY   (100 episodes total, 62 valid)
==============================================================
  Before-ODD collisions  :  38.0%  (agent had no chance)
  ── valid episodes (62) ─────────────────────────────
  MRC rate    (REAL)     :  77.4%  ← main metric
  Post-ODD collision     :  17.7%
  Avg reward (valid)     : 8.54

  MRM distribution (valid episodes):
    STRAIGHT_STOP  (0)   :  14.5%
    IN_LANE_STOP   (1)   :  38.7%
    ROAD_SHOULDER  (2)   :  46.8%

  Shoulder decision (38 ep where shoulder was available):
    Correct (used shoulder)    :  73.7%
    Wrong   (ignored shoulder) :  26.3%

  Avg time from ODD exit to MRC  : 7.24 s
==============================================================
```

### What is saved

```
logs/runs/<timestamp>_phase3_eval/
├── episodes.csv          ← per-episode data (same format as training)
├── steps_episode_N.csv   ← sim steps per episode
├── eval_summary.json     ← all metrics as JSON
└── eval_summary.csv      ← one-row table → paste directly into thesis
```

### Understanding the output

| Value | What it means |
|---|---|
| **Before-ODD collisions** | Ego hit lead car during approach — agent had no chance to act. Not the agent's fault. High value means scenarios are tight (normal). |
| **MRC rate (valid)** | Of episodes where ODD fired and agent acted — how many ended in safe full stop. **This is the key thesis metric.** |
| **Post-ODD collision** | Agent chose a wrong/too-light MRM and crashed. |
| **Shoulder correct rate** | When shoulder was available — did agent pick `ROAD_SHOULDER_STOP`? A trained policy should be ≥70%. |
| **Avg time to MRC** | How many seconds from ODD exit until full stop. Lower = faster braking decision. |

### For the thesis comparison table

Run evaluation 5 times — once for each experiment — and collect `eval_summary.csv` from each:

| Experiment | Config | Policy |
|---|---|---|
| Baseline (Phase 2) | `configs/phase2.yaml` with `phase2_baseline` | N/A (rule-based) |
| RL full system | `phase3_eval.yaml` with full ablations=true | your trained `policy.zip` |
| RL no risk features | same eval config but `use_risk_features: false` | trained without risk obs |
| RL no risk penalty | same eval config but `use_risk_penalty_in_reward: false` | trained without risk reward |
| RL no shield | same eval config but `use_shield: false` | trained without shield |

Each run produces one row in `eval_summary.csv`. Stack them side by side → thesis table.

---

## 9. Full workflow: train → evaluate

```
1.  configs/phase3_train.yaml  → set total_timesteps: 5000
2.  py -3 -u -m src.main --phase phase3_train --config configs/phase3_train.yaml
        (runs overnight)
3.  Check training result:
        py -3 scripts/inspect_policy.py \
            --policy logs/runs/<latest>/artifacts/policy.zip --n_episodes 30
4.  configs/phase3_eval.yaml   → set policy_path to your policy.zip
5.  py -3 -u -m src.main --phase phase3_eval --config configs/phase3_eval.yaml
        (runs ~2 hours for 100 episodes)
6.  Open logs/runs/<latest>_phase3_eval/eval_summary.csv  → thesis table row
```

---

## 7. MRM reference

| ID | Name | What it does | Best when |
|---|---|---|---|
| 0 | `STRAIGHT_STOP` | Full brake, no steer | Emergency, high risk |
| 1 | `IN_LANE_STOP` | Lighter brake, stays in lane | No shoulder space |
| 2 | `ROAD_SHOULDER_STOP` | Moderate brake + steer right | Shoulder available (rear gap > 20 m) |

Reward hierarchy when shoulder available: **MRM2 (+15) > MRM1 (+10) > MRM0 (+8)**
