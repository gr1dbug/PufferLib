# GbGridEnv Retrospective (2026-02-06)

## Context
This note summarizes what we learned while improving `pufferlib/ocean/gbgridenv/` from an early prototype to a version that now trains materially better.

Recent reported training outcome after updates:
- explained_variance: `0.955`
- entropy: `0.318`
- policy_loss: `0.023`

## Progress Review

### What changed and why it helped
1. Episode semantics were fixed.
- Before: terminal flags were set and then immediately cleared due to reset-in-step behavior.
- After: reset is deferred (`needs_reset`) so learners observe `done=True` transitions.
- Impact: value targets became coherent, improving critic learning.

2. Reward loopholes were removed.
- Before: invalid moves could still get movement reward (truthy negative return), and no-op/blocked patterns could avoid energy costs.
- After: reward distinguishes successful move vs invalid move; energy burns each step.
- Impact: reduced exploitability and encouraged meaningful behavior.

3. A concrete task objective was introduced.
- Before: movement + pickup dense reward allowed high returns without goal completion.
- After: one random `DEPOT` tile per map and reward tied to delivery of carried material.
- Impact: task structure now aligns with intended behavior (collect -> return -> deposit).

4. Observation signal was made task-relevant.
- Before: a near-constant binary observation had low learning value.
- After: `20x20` local patch plus global goal features (`energy`, depot direction, distance proxy, carried load).
- Impact: policy can represent task state and plan toward depot.

5. Spawn randomization was fixed.
- Before: deterministic spawn biased direction features and policy behavior.
- After: spawn is randomized over soil cells.
- Impact: better distributional coverage and more robust policy learning.

## Misunderstandings Corrected (and common RL env traps)

1. "Training" does not imply the task is learnable.
- Early runs had high SPS and nonzero losses but the MDP/reward design was still degenerate.
- Key lesson: first verify environment semantics and incentives, then optimize training throughput.

2. Dense reward is not automatically "better".
- A dense signal helped bootstrap, but if it is not goal-aligned it creates reward hacking.
- Key lesson: reward density is useful only when shaped around the real objective.

3. Metrics must be interpreted together.
- Entropy decreasing alone is ambiguous: it can mean learning or collapse.
- Explained variance rising (especially to ~0.95) with sensible loss behavior is a stronger indicator that value learning is now grounded.

4. Observation richness matters as much as reward design.
- Without state variables tied to objective progress (energy, carrying, goal geometry), PPO can plateau even with a "correct" reward.

5. Terminal/truncation correctness is foundational.
- If done flags are hidden, value targets become inconsistent regardless of reward quality.

## Implementation Challenges We Hit

1. C/Python buffer type synchronization.
- To encode signed/continuous goal features, observation type changed from `uint8` to `float`.
- This required synchronized updates in C struct, Python spaces, and standalone harness.
- Risk class: silent memory/shape bugs if any layer remains stale.

2. Local patch + global feature composition.
- Needed a clear convention for channels and normalization ranges.
- Current channel mix is practical, but it is easy to accidentally leak redundant or unstable signals.

3. Reset/terminal ordering bugs are subtle.
- Easy to accidentally clear terminal info during immediate reset logic.
- Fix required explicit lifecycle state (`needs_reset`).

4. Spawn/depot stochasticity edge cases.
- Map generation can accidentally create pathological placement if fallbacks are not explicit.
- Added fallback logic so there is always one depot.

5. Build flow friction.
- Full `build_ext` can fail in this environment due to unrelated CUDA/Torch extension setup.
- Practical workaround: targeted build command (`build_gbgridenv`).

## Current Risks / What You May Still Be Underestimating

1. Reward scale stability.
- Deposit reward magnitude can dominate all other terms and produce brittle policy updates depending on map size/energy.
- Watch for sudden KL spikes or unstable returns when changing map/task parameters.

2. Overfitting to local map statistics.
- One-depot static regime may produce brittle behaviors if topology varies later.

3. Ambiguity between task success and episode return.
- High return can still arise from partial strategy quality.
- Add explicit success metrics (e.g., deposit events, delivered mass per episode, distance traveled with load).

4. Truncation semantics still minimal.
- You currently rely on terminal behavior, but later curriculum/horizon changes may require explicit truncation handling.

## Recommended Next Steps

1. Add explicit logged diagnostics (high priority).
- `delivered_total_per_episode`
- `num_deposits_per_episode`
- `pickup_total_per_episode`
- `mean_distance_to_depot`
- `episode_length`

2. Harden reward design with lightweight ablations.
- Run short sweeps over:
  - step penalty magnitude
  - invalid move penalty
  - carry normalization constant
- Goal: identify reward terms that stabilize learning without encouraging stalling.

3. Add controlled evaluation scenarios.
- Fixed seeds and scripted maps for:
  - near depot / far depot
  - sparse traversable corridors
  - high vs low resource density
- This helps separate "policy improvement" from "lucky map distribution".

4. Expand goal observability only if needed.
- Current additions are good. If progress stalls, consider:
  - steps remaining channel
  - binary "currently on depot"
  - carrying-by-resource-type channels (if resources later become distinct objectives)

5. Build a small environment test checklist.
- Invariants to assert in tests:
  - exactly one depot after reset
  - terminal is observable at episode end
  - energy monotonically decreases each step until terminal
  - no reward on invalid move from movement term
  - observation shape/range/channel semantics

6. Keep complexity staged.
- Before adding multi-agent or richer chemistry, lock in single-agent reliability and metric instrumentation.

## Bottom Line
The recent metric jump is consistent with real environment quality improvements, not just optimizer luck. The biggest conceptual shift was moving from "activity reward" to "goal completion reward + goal-relevant state" while fixing episode semantics. That is the core RL environment design lesson from this iteration.
