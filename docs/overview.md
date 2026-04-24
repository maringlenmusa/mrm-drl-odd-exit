# Master Thesis Overview

**Topic:** Decision Logic for ODD Exit — A Reinforcement Learning Approach for MRM Selection

Short, simple summary so anyone (including you) can understand what you will do, how hard it is, and what technical issues you might face.

---

## What You Will Do (Simple Terms)

**In one sentence:**  
When a self-driving car must **exit normal operation** (in this thesis: triggered by a **timer** in simulation), it chooses a **safe way to stop** — the three MRM types from ISO 23793: straight stop, in-lane stop, road shoulder stop — until **MRC** (full stop) is reached. Your thesis is: **build and train a small “brain” (RL) that chooses which of these stopping strategies to use**, using risk and traffic info, and compare it to simple rules. You do this in **esmini only**, with default scenario(s) and 1–2 scenario types (ideal: 2 cars + ego); see `docs/supervisor_answers.md` for scope.

**In steps:**

1. **Phase 0** — Lock scope with your supervisor: which 3 stopping types (MRMs), what “safely stopped” (MRC) means, how you trigger “exit,” what data the sim gives you.
2. **Phase 1** — Get esmini talking to Python over UDP: read car state, send “brake” (or similar), trigger “exit” at a chosen time, log everything. No learning yet.
3. **Phase 2** — Add: risk (DRF + DARA), simple **rule-based** choices (e.g. “if very risky → brake”), a **safety shield** that can block bad choices, run 1–2 scenario types (default esmini, ideal 2 cars + ego), log success/collision/time-to-MRC. Phase 2 also supports a curved road option, ego lane following (and IN_LANE_STOP that stays on the road), and left/right gap logic for adjacent lanes only.
4. **Phase 3** — Turn that into an **RL problem** (Gym env), train an agent (e.g. PPO) to pick the MRM, use risk in the observation and reward, compare **baseline vs RL** and run **ablations** (with/without risk, with/without shield).

So in one sentence: **you will design and implement a risk-aware decision logic for “how to stop safely when leaving the driving domain,” first as rules + shield, then as an RL agent, and evaluate both in esmini.**

---

## How Hard Is It?

- **Medium overall.** You’re not inventing new RL theory; you’re applying known ideas (PPO, Gym env, reward design) to a concrete ODD-exit problem.
- **Hard parts:**  
  - Getting **esmini + UDP + your code** to run reliably (Phase 1).  
  - **Reward design** and **training stability** (Phase 3).  
  - Making sure **risk, baseline, and shield** are consistent and interpretable (Phase 2).
- **Easier parts:**  
  - Risk formulas (DRF/DARA) and rule-based baseline are well specified in your docs.  
  - RL part is standard (discrete actions, PPO, existing papers to follow).

Difficulty is mainly **integration and debugging**, not deep math.

---

## Can You Use Dummy / Fake Data If Something Doesn’t Work?

**Short answer: yes, but only in a controlled way, and you must say so clearly.**

- **Simulator / UDP / esmini:**  
  If the real esmini UDP feed is broken or delayed, you can:
  - Use **recorded or scripted state sequences** (e.g. from a few manual runs or from a simple script that mimics esmini state) to develop and debug your **Python pipeline** (observation builder, risk, baseline, shield, env step logic).  
  - In the thesis, you describe this as a **“simplified/offline test environment”** or **“synthetic state stream for pipeline development”** and then report final results **with the real simulator** (or clearly state the limitation if you couldn’t get it working).

- **Risk (DRF/DARA):**  
  You should implement the real formulas. If you later find bugs, you can temporarily **fix or clamp values** (e.g. cap risk to [0,1]) so training doesn’t explode, but you shouldn’t “fake” risk with random numbers for your main results.

- **RL training / evaluation:**  
  You **must not** fake success rates or rewards. You can:
  - Use **fewer scenarios** or **shorter episodes** for quick experiments.  
  - Use **dummy/simple scenarios** (e.g. one lane, one obstacle) to check that the env and training loop work, then move to real scenarios for the thesis results.  
  - If the full system doesn’t train well, you still write up what you did, what you tried, and what failed (and why). That’s valid thesis content.

**Rule of thumb:**  
- **Fake/dummy OK:** to get the **pipeline running** (state flow, logging, env interface) or to **unit-test** parts of the code.  
- **Not OK:** to invent or hide data in the **final results** (e.g. “we achieved X% success” when that wasn’t measured in your setup).  
- **Always:** document what is real vs synthetic/dummy and what the limitations are.

---

## Technical Difficulties You Might Face

| Area | Difficulty | What can go wrong |
|------|------------|--------------------|
| **esmini + UDP** | High | Wrong port, packet format, or timing; state not updating; control not applied. Your `codec` and timeouts are critical. |
| **Observation / state** | Medium | Sim gives different fields than you expect (e.g. no lane IDs); you need a robust parser and maybe fallbacks (e.g. “unknown lane” → -1). |
| **Risk (DRF/DARA)** | Low–Medium | Formulas are clear; bugs are usually sign errors or wrong units (m vs m/s, etc.). Validate on a few hand-made cases. |
| **Shield / MRM rules** | Medium | Edge cases (e.g. exactly at gap threshold); “downgrade-only” and veto logic must match what you write in the thesis. |
| **RL env** | Medium | `reset`/`step` must close/restart sim cleanly; reward must be aligned with MRC/collision/timeout; done flags must be correct. |
| **Training** | Medium | PPO can be unstable; reward scale and risk penalty weight matter; you may need many runs or seeds. |
| **Scenarios** | Medium | OpenSCENARIO files can be fiddly; you need at least a few that trigger ODD exit and allow different MRMs. |

Biggest single risk: **esmini + UDP not behaving as assumed.** That’s why Phase 1 is so important and why your plan says “only `codec` should change” if the format differs — so you can swap in a **mock UDP feed** (dummy data) until the real one works.

---

## One-Sentence Summary

You will **build a small risk-aware “brain” that chooses how the car should safely stop when it leaves its normal driving domain**, implement it first as rules + safety shield, then as an RL agent, run both in the esmini simulator, and compare them; **you may use dummy or simplified data to get the pipeline working**, but final results and conclusions must be based on what you actually measured and clearly documented.
