# Phase 2 Implementation Plan: Episodic Meta-Control using an LLM

This plan breaks down the implementation of Phase 2 into 6 logical stages to incrementally build and verify the LLM meta-controller.

## User Review Required

> [!IMPORTANT]
> The plan has been revised based on your feedback. We are using OpenAI exclusively, adding the required configuration parameters, and breaking the implementation into the requested 6 stages. We have incorporated the nested `alphas` JSON structure and the `confidence` field, specifically treating the `confidence` field strictly as an interpretability metric (not impacting updates).
> I will NOT write any code until you explicitly say **"approved"** or provide feedback.

---

## 1. LLM Provider Specifications
We will use the official OpenAI Python SDK (`openai>=1.0.0`) exclusively.
- API Key loaded from the `OPENAI_API_KEY` environment variable.
- Model name configurable from `default_config.yaml` (defaulting to `gpt-5`).
- Includes retry logic with exponential backoff.
- Returns raw text response for downstream parsing.

## 2. Phase 2 Configuration Parameters
We will add the following to `MADRL/config/default_config.yaml`:
```yaml
phase2:
  llm_provider: openai
  llm_model: gpt-5

  alpha_smoothing: 0.8
  max_alpha_change: 0.25

  alpha_min: 0.05
  alpha_max: 0.95

  warmup_blocks: 10
  evaluation_blocks: 20
```

## 3. Justification for Alpha Constraints
**Implementation Notes on Constraints:**
- Phase 1 training converged naturally around $\alpha \approx 0.45\text{--}0.55$. The frozen policy has the highest confidence in this region.
- Large alpha jumps may push the policy into poorly explored operating regions.
- `max_alpha_change = 0.25` acts as a stability constraint, preventing abrupt regime switching while still allowing meaningful adaptation within a few blocks.
- `alpha_smoothing = 0.8` prevents the LLM from reacting too aggressively to stochastic KPI fluctuations.

---

## 4. Proposed Staged Execution

### STAGE 1 — LLM Infrastructure
**Purpose:** build the core LLM calling and parsing layer
- **Files to be created or modified:** 
  - `MADRL/llm/llm_client.py` (New)
  - `MADRL/llm/response_parser.py` (New)
- **Key functions/classes to implement:** 
  - `OpenAIClient` (with `query` method utilizing exponential backoff)
  - `parse_llm_response` (extracts reasoning, self_correction, compiled_rule, nested alphas dict, and confidence score)
- **Dependencies:** None.
- **How it will be tested:** Provide a mocked string response (valid JSON, markdown-wrapped JSON, and malformed JSON) to `parse_llm_response` to ensure it falls back gracefully or parses correctly.
- **Estimated complexity:** LOW

### STAGE 2 — KPI Aggregation and Input Builder
**Purpose:** transform raw episode data into structured LLM inputs (A, B, C)
- **Files to be created or modified:** 
  - `MADRL/llm/prompt_builder.py` (New)
- **Key functions/classes to implement:** 
  - `build_input_a_demand_stats`
  - `build_input_b_kpi_summary`
  - `build_input_c_history`
  - `build_full_prompt`
- **Dependencies:** Stage 1 (for output schema knowledge).
- **How it will be tested:** Pass dummy demand arrays and KPI dictionaries through the builders and print the resulting prompt strings to ensure mathematical aggregation (like skewness and kurtosis) is correct.
- **Estimated complexity:** MEDIUM

### STAGE 3 — Alpha Controller
**Purpose:** apply smoothing, capping, and validation to LLM alpha outputs
- **Files to be created or modified:** 
  - `MADRL/llm/alpha_controller.py` (New)
- **Key functions/classes to implement:** 
  - `apply_alpha_update`
- **Dependencies:** None.
- **Handling Confidence:** Ensure `apply_alpha_update` completely ignores the `confidence` field. It must not factor into smoothing, clipping, or updates.
- **How it will be tested:** Write unit tests passing extreme alpha deltas to verify `max_alpha_change` and `alpha_smoothing` bounds behave correctly, and verify `confidence` is unused mathematically.
- **Estimated complexity:** LOW

### STAGE 4 — Phase 2 Runner (Warm-Up Sub-Phase)
**Purpose:** orchestrate 10-block warm-up loop with rulebook accumulation
- **Files to be created or modified:** 
  - `MADRL/config/default_config.yaml` (Modify: inject `phase2` block)
  - `MADRL/llm/rulebook.py` (New)
  - `MADRL/training/run_phase2.py` (New)
- **Key functions/classes to implement:** 
  - `extract_rulebook`
  - `save_rulebook`
  - `run_evaluation_block` (helper)
  - `run_warmup_phase`
- **Dependencies:** Stages 1, 2, 3, and the frozen policy from Phase 1.
- **Extended Reproducibility Logging:** Ensure the loop logs Block ID, KPI snapshot, pre-update alpha vector, raw LLM alpha vector, post-controller alpha vector, and `confidence` score (treated solely as an interpretability metric).
- **How it will be tested:** Execute a dry-run of the warm-up loop using a dummy LLM client that returns random valid alphas to ensure environment steps, alpha injections, and rulebook extraction execute without crashing.
- **Estimated complexity:** HIGH

### STAGE 5 — Phase 2 Runner (Official Evaluation Sub-Phase)
**Purpose:** run 20-trace benchmark with frozen rulebook
- **Files to be created or modified:** 
  - `MADRL/training/run_phase2.py` (Modify)
- **Key functions/classes to implement:** 
  - `run_official_evaluation`
- **Dependencies:** Stage 4.
- **Handling Confidence:** Log `confidence` against actual KPI improvements across the benchmark traces to later measure if the LLM's self-assessment correlates with decision quality.
- **How it will be tested:** Run a few traces of the official evaluation using the saved rulebook and frozen policy to verify metrics are cleanly recorded.
- **Estimated complexity:** MEDIUM

### STAGE 6 — Baseline Comparisons and Results Logging
**Purpose:** run fixed-alpha, rule-based, and random-alpha baselines for comparison
- **Files to be created or modified:** 
  - `MADRL/training/run_phase2.py` (Modify)
- **Key functions/classes to implement:** 
  - `run_baseline_fixed_alpha`
  - `run_baseline_rule_based`
  - `run_baseline_random_alpha`
  - `save_comparative_results`
- **Dependencies:** Stage 5.
- **How it will be tested:** Run the baselines against the exact same 20 traces and ensure their aggregate output JSON cleanly matches the Phase 2 LLM evaluation format for easy paper charting.
- **Estimated complexity:** MEDIUM
