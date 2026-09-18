# Step 5: Intra-Episodic Meta-LLM with Accumulated Memory & Experience

This experiment explores **Cross-Trace Accumulated Episodic Memory** for the Intra-Episodic Meta-LLM Controller on the frozen Phase 1 universal HAPPO MARL policy.

---

## 1. Problem Formulation & Motivation

- **Prior Baseline (Step 3A)**: In Step 3A, the LLM was cold-started on every single 200-timestep trace (`reset_episode` cleared all history). The LLM made decisions with 0 prior experience at $t=50$, resulting in an exploration penalty (cost 93.36 vs Rule-Based 86.34).
- **Hypothesis**: Giving the LLM **accumulated cross-trace memory** (experiencing diverse demand shocks in Traces 1–5, storing operational outcomes, and maintaining a compiled rulebook) will eliminate the cold-start penalty and allow the LLM to make informed, experienced strategic adjustments from $t=50$ onward in evaluation traces (Traces 6–20).

---

## 2. Experimental Design

- **Policy**: Frozen Phase 1 MARL Universal Policy (`MADRL/results/phase1_weights/frozen_policy.pt`). Untouched and unmodified.
- **Demand Environment**: 20 Merton Jump Diffusion Demand Traces ($L=4, h=1, b=1$).
- **Intervention Horizon**: Every $N=50$ timesteps ($t=50, 100, 150$).
- **Warm-Up Phase**: Traces 1–5 (indices 0..4) — LLM actively explores, intervenes, and accumulates memory/rules.
- **Official Evaluation Phase**: Traces 6–20 (indices 5..19) — LLM controls the chain using its rich accumulated knowledge base.
- **Fair Benchmarks**: Fixed $\alpha=0.3$, Rule-Based Heuristic, and Random $\alpha$ evaluated under the exact same warm-up/eval split.

---

## 3. How to Run

```powershell
# Run with Google Gemini Flash (live API calls)
.venv\Scripts\python.exe experiments/step5_llm_accumulated_memory/run_step5.py --model gemini-2.5-flash-lite

# Run offline mock mode for testing
.venv\Scripts\python.exe experiments/step5_llm_accumulated_memory/run_step5.py --mock
```
