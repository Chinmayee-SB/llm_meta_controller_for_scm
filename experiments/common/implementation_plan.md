# Next Steps: Experiment Plan

Everything below uses the **existing frozen Phase 1 policy** unless explicitly marked as "retrain."

---

## Step 1 — Anchor Numbers (No LLM, ~1 hour)

**Goal:** Establish the baseline performance of your generalist policy so every future experiment has context.

Run the frozen policy on the 20 evaluation demand traces with several fixed α values. Compute **both** your current metrics AND the base paper metrics:

| Run | α (all agents) | Metrics to Compute |
|-----|----------------|-------------------|
| A | 0.1 | Per-period cost, Avg Fluctuation (F), Amplification %, Service Level |
| B | 0.3 | Same |
| C | 0.5 | Same |
| D | 0.625 (base paper value) | Same |
| E | 0.7 | Same |
| F | 0.9 | Same |

**What you learn:**
- Which fixed α is best for YOUR generalist policy (may not be 0.625)
- How far your generalist is from the base paper's specialist (~34 per-period cost)
- The α discrimination fingerprint — is the cost curve steep or flat across {0.1 → 0.9}?
- If the curve is flat → the α lever is weak → LLM meta-control has limited upside regardless of framework

**Script needed:** `evaluate_base_paper_metrics.py` — loads frozen policy, runs 20 traces, computes base paper metrics.

---

## Step 2 — Intra-Episodic Diagnostic (~30 min)

**Goal:** Determine if the existing policy can handle mid-episode α changes.

Run the frozen policy on 5 demand traces with α flipped mid-episode:
- Timesteps 0–99: α = [0.1, 0.1, 0.1, 0.1]
- Timesteps 100–199: α = [0.9, 0.9, 0.9, 0.9]
- No env reset, no GRU reset — just change α in the observation

**What you learn:**
- **Smooth transition within 5–10 timesteps** → Intra-episodic control works with the current Phase 1. No retrain needed. Proceed to Step 3A.
- **Wild oscillation for 30+ timesteps** → Current policy can't handle mid-episode changes. Skip intra-episodic for now. Proceed to Step 3B.

**Script needed:** `diagnostic_alpha_flip.py` — plot per-timestep ordering behavior of all 4 agents across the 200 timesteps, with a vertical line at t=100 showing the flip point.

---

## Step 3A — Intra-Episodic LLM Controller (If Diagnostic Passes)

**Goal:** Build and test the new framework where the LLM intervenes within a single episode.

Build `run_phase2_intra_episodic.py` as a **new, separate script** alongside the existing Phase 2 scripts. It should:

1. Run each of the 20 eval traces as a single 200-timestep episode
2. LLM intervenes every N timesteps (start with N=50, giving 4 LLM calls per episode — same cost as current Phase 2)
3. Compute base paper metrics at the end of each trace
4. Run the same baselines (fixed-α, rule-based, random-α) with the same intervention schedule for fair comparison

**Baselines (all use same intervention interval N):**

| Baseline | α Strategy |
|---|---|
| Fixed α (best from Step 1) | Static, never changes |
| Rule-based | Simple heuristic: if backlog rising → lower α, if inventory heavy → raise α |
| Random α | Random from {0.1, ..., 0.9} every N timesteps |
| **LLM (Ours)** | LLM-guided every N timesteps |

**Then ablate on interval:** Run the LLM version with N = {20, 50, 100} to find the sweet spot.

---

## Step 3B — Fix Current Phase 2 (If Diagnostic Fails)

**Goal:** Improve the existing inter-episodic framework using the cheapest fixes first.

Run in this order, branching from Version 0 each time:

### Version A — Mild Demand (~30 min, no retrain)
- Reduce Merton `lambda` from 15 → 5
- Run existing Phase 2 pipeline
- **If LLM beats rule-based** → demand noise was the problem. Stop here.

### Version D — Conditional Rulebook (~1 hr, no retrain)
- Classify demand blocks by regime (high/normal/low volatility)
- Store rules per regime, inject only relevant rules into prompt
- **If LLM beats rule-based** → memory structure was the problem. Stop here.

### Version E — Discrete Alpha (retrain required, ~4–6 hrs)
- Retrain Phase 1 with α sampled from {0.1, 0.2, ..., 0.9} instead of U(0.1, 0.9)
- Update LLM prompt to output from discrete set
- Re-run Phase 2 evaluation
- **If LLM beats rule-based** → continuous α was too hard for both the policy and the LLM. Stop here.

---

## Step 4 — Final Results Table

After running the winning configuration, produce this table for your paper/thesis:

```
══════════════════════════════════════════════════════════════════
                    RESULTS COMPARISON
══════════════════════════════════════════════════════════════════
Method                  Per-Period   Avg Fluct.  Amplif.   SL
                        Cost         (F)         (%)       (%)
──────────────────────────────────────────────────────────────────
Base Paper HAPPO        ~34          0.758       92.37%    N/A
  (specialist, α=0.625)

Our HAPPO (best α)      ?            ?           ?         ?
  (generalist, static)

Our HAPPO + LLM         ?            ?           ?         ?
  (generalist, adaptive)

Our Rule-Based          ?            ?           ?         ?
  (generalist, heuristic)
══════════════════════════════════════════════════════════════════
```

The narrative:
- Row 1 = theoretical ceiling (specialist)
- Row 2 = the cost of generalization (your policy without LLM)
- Row 3 = the value the LLM adds (your contribution)
- Row 4 = proves LLM is better than a simple heuristic

---

## Decision Tree Summary

```
Step 1: Anchor numbers (fixed-α sweep)
    │
    ├── Is α curve flat? (cost spread < 10%)
    │     → α lever is weak. Consider Version E (discrete) or C (larger network)
    │
    └── Is α curve steep? (cost spread > 15%)
          → α lever works. Proceed to Step 2.
          │
          Step 2: Diagnostic (α flip mid-episode)
              │
              ├── Smooth transition → Step 3A (intra-episodic)
              │
              └── Erratic behavior → Step 3B (fix current Phase 2)
```

## Time Budget

| Step | Time | Requires Retrain? |
|------|------|-------------------|
| Step 1 (anchor numbers) | ~1 hr | No |
| Step 2 (diagnostic) | ~30 min | No |
| Step 3A (intra-episodic) | ~2–3 hrs | No |
| Step 3B, Version A | ~30 min | No |
| Step 3B, Version D | ~1 hr | No |
| Step 3B, Version E | ~4–6 hrs | **Yes** |

**Worst case total: ~8 hours spread across sessions. Best case: 2 hours if Step 3A works immediately.**
