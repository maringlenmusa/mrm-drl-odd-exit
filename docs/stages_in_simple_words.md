# What You Will Do at Each Stage (Same Simple Words)

You have a plan: Phase 0, Phase 1, Phase 2, Phase 3, and then writing the thesis. This doc says in plain language **what you do at each stage** and **how that gets you to your master thesis**. Same voice as the papers doc — no fancy words.

---

## The big picture: how you reach the master thesis

Your thesis says: *"When the car must exit (stop being fully autonomous), I built a system that chooses how to stop safely using risk and traffic — first with simple rules, then with RL. I compared them and showed what helps (risk in observation? shield?)."*

You get there by doing **four phases of work**, then **writing it up**:

1. **Phase 0** — You agree with your supervisor what “success” means and what data you have. So you don’t build the wrong thing.
2. **Phase 1** — You get the simulator talking to your code and one fixed “brake” maneuver working. So you have a backbone.
3. **Phase 2** — You add risk, simple rules, and a safety shield, and run several scenarios. So you have a baseline to beat.
4. **Phase 3** — You train an RL agent to choose the maneuver, compare it to the baseline, and run ablations. So you have results for the thesis.
5. **Final** — You organize results, make plots and tables, and write the thesis (intro, method, experiments, results, discussion).

👉 **So:** Each stage gives you one piece. Phase 1 = backbone. Phase 2 = baseline + risk + shield. Phase 3 = RL + comparison + ablations. Final = the written thesis.

---

## Phase 0: Planning & Scope (1–2 weeks)

**What you do there (simple):**

You **don’t code** yet. You **ask your supervisor** the right questions and write the answers down. You need to know: which exact 3 MRM types (e.g. straight stop, in-lane stop, road shoulder stop from ISO 23793), what counts as “MRC reached” (full stop), how ODD exit is triggered (timer only), what simulator is the target (esmini only), what state you can read from the sim (x, y, angle, speed, acceleration from esmini), and whether they want only high-level “which MRM” or also low-level control. *(Answers are in \texttt{docs/supervisor\_answers.md}.)*

You save everything in something like `docs/supervisor_answers.md` and refer back to it when you implement Phase 2 and 3.

**Why it matters:**  
If you skip this, you might implement the wrong MRMs, the wrong “success” condition, or the wrong interface. You waste weeks later.

👉 **So:** Phase 0 = lock what “success” means and what data you have. No code, only clarity.

---

## Phase 1: Basic Simulation Loop (2–3 weeks)

**What you do there (simple):**

You get **one scenario** running in esmini. Your Python code **receives state** from the sim over UDP (e.g. ego speed, position). At a **fixed time** (e.g. 8 seconds) you say “ODD exit now” and send a **fixed maneuver**: e.g. “brake in lane” (brake = 1, throttle = 0). The car brakes, you **log every step** (time, speed, brake, etc.) and at the end you write a short **episode summary** (did it stop? collision? timeout?). No risk, no RL, no choices — just: sim runs, exit triggers, one MRM runs, logs are saved.

You check that: esmini starts, UDP packets arrive, sim_time increases, brake kicks in at the right time, speed goes down, and logs are written.

**Why it matters:**  
Everything later (risk, baseline, RL) sits on top of this loop. If this doesn’t work, nothing else will.

👉 **So:** Phase 1 = one scenario, one fixed “brake” maneuver, and clean logs. Your backbone.

---

## Phase 2: Risk, Baseline, Shield (3–4 weeks)

**What you do there (simple):**

You **add** to the Phase 1 loop:

- **Risk:** You compute DRF using supervisor’s code and the reference paper (ego motion risk); DARA is simplified or out of focus (e.g. simple TTC placeholder). You combine into one **risk** value and log it every step.
- **Observation:** You build a small “picture” of the situation: ego speed, lane, gaps in front/rear/left/right, relative speeds. That’s what the decision logic (and later RL) sees.
- **Baseline:** Simple **rules** choose the MRM: e.g. “if risk is very high → brake”, “if front gap too small → brake”, “if right gap big enough → pull over”, else brake. No learning yet.
- **Shield:** Before you execute the chosen MRM, a **safety check**: e.g. “if the chosen maneuver is lane change but the gap is too small → force brake”. And: once you’re in a safer maneuver, you’re not allowed to go back to a riskier one (downgrade-only).
- **Scenarios:** You use default esmini scenario(s); 1–2 scenario types; ideal: 2 cars + ego. You run the same scenario(s) multiple times with different speed/distance. For each run you log: success (MRC reached?), collision?, time to MRC, max risk, etc. You get a **baseline results table**. 


Just one scenario. Plan the rewrd only wiht the state value. General model that wil lelemenit that will elemtintle the cost of odd exit.


**Why it matters:**  
You now have a **rule-based system** that uses risk and obeys rules. That’s what you’ll **compare** the RL agent to. And you’ve implemented the same risk and shield that the RL will use.

👉 **So:** Phase 2 = risk + observation + rules + shield + multi-scenario logs. Your baseline and your “comparison point” for the thesis.

---

## Phase 3: RL Training & Evaluation (4–5 weeks)

**What you do there (simple):**

You **wrap** the Phase 2 loop into a **Gym-style environment**: `reset()` starts a scenario and returns an observation; `step(action)` means “do this MRM” and returns next observation, reward, and whether the episode is done (MRC reached, collision, or timeout). The **reward** is what you agreed: big minus for collision, plus for reaching MRC, and optionally a **dense** penalty for high risk every step (risk-prioritized). You **train** an agent (e.g. PPO) to choose the MRM. You use **risk-prioritized scenario sampling**: harder/dangerous runs get sampled more so the agent practices difficult situations.

Then you **evaluate**: you run the trained agent on the same scenarios and you compare **baseline (Phase 2) vs RL**. You also run **ablations**: full RL vs RL without risk in observation, vs RL without risk penalty in reward, vs RL without shield. So you get tables and plots: which setup works best, and why (e.g. “shield matters”, “risk in observation helps”).

**Why it matters:**  
This is the core “research” part: you show that RL can be used for MRM selection, and you show **what helps** (risk features, risk penalty, shield) via ablations. That’s your results chapter.

👉 **So:** Phase 3 = RL env + train + evaluate + compare to baseline + ablations. Your thesis results.

---

## Final: Thesis Preparation & Writing

**What you do there (simple):**

You **organize** all runs: baseline logs, Phase 3 training logs, evaluation logs, ablation logs. You **make plots** (e.g. reward over time, success rate per scenario) and **tables** (baseline vs RL, ablations). You **write** the thesis: intro (problem, ODD/MRM), related work (the papers you used), methodology (simulator, state, actions, risk, baseline, RL, reward, shield, scenarios, evaluation), experiments (what you ran), results (tables and plots), discussion (what matters, limitations). You start **at least 8 weeks before** the deadline; your supervisor gets a part at least 5 weeks before and intro/summary at least 3 weeks before (as in your proposal).

**Why it matters:**  
The thesis is the document that proves you did the work and understood it. The stages give you the content; this stage turns it into the story.

👉 **So:** Final = organize results, plots, tables, and write the thesis. How you reach the master thesis on paper.

---

## One table to remember (what you do, stage by stage)

| Stage   | What you do in simple words | What you get at the end |
|--------|-----------------------------|--------------------------|
| **Phase 0** | Ask supervisor, write answers (MRMs, MRC, trigger, sim, state signals). | Clarity: no wrong scope. |
| **Phase 1** | Sim + UDP + one scenario + fixed brake + logs. | Backbone: loop works. |
| **Phase 2** | Risk (DRF+DARA) + observation + rules + shield + several scenarios + logs. | Baseline + risk + shield. |
| **Phase 3** | Gym env + reward + train RL + evaluate + baseline vs RL + ablations. | Results: tables & plots. |
| **Final**  | Organize logs, plots, tables; write intro, method, experiments, results, discussion. | The master thesis document. |

---

## What you can say when someone asks “what do you do in each stage?”

- **Phase 0:** “I lock scope with my supervisor: which MRMs, what counts as success, what data I get. I write it down so I don’t build the wrong thing.”
- **Phase 1:** “I get the simulator talking to my code and one fixed brake maneuver working, with logs. That’s the backbone for everything else.”
- **Phase 2:** “I add risk (DRF+DARA), build the observation, add simple rules and a safety shield, and run several scenarios. I get a baseline and the same risk/shield the RL will use.”
- **Phase 3:** “I turn that into an RL environment, train an agent to choose the MRM, evaluate it, compare to baseline, and run ablations (with/without risk, with/without shield). That’s my results chapter.”
- **Final:** “I organize all results, make plots and tables, and write the thesis: intro, method, experiments, results, discussion.”

So you’re never lost: you know **what you do at each stage** and **how that gets you to the master thesis**.
