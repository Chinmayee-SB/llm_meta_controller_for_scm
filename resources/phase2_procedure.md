# Phase 2 Procedure: LLM Meta-Controller and Decentralized Execution (Final v2)

---

## Overview

During Phase 2, GRU weights are completely frozen. The LLM operates as an episodic strategic advisor in **two sub-phases**:

1. **LLM Warm-Up (10–50 episodes / 2–10 feedback cycles):** The LLM actively learns domain-specific rules via meta-prompt self-correction. Rules evolve each block.
2. **LLM Evaluation (20-trace benchmark):** Rules are frozen. The LLM assigns α based on its compiled rulebook. Official KPIs are recorded here.

This two-sub-phase design (from NotebookLM analysis of InvAgent framework) ensures the LLM is domain-adapted before official evaluation, analogous to how few-shot learning works for LLM agents.

---

## 1. Evaluation Block Design

### Block size
```yaml
evaluation_block:
  main_experiments: 5            # 5 episodes per block (noise-filtered signal)
  ablation_sizes: [1, 5, 10]    # sensitivity analysis
```

### Why 5 episodes per block
A single Merton JD episode has high variance. Averaging over 5 episodes filters stochastic noise and gives the LLM a stable KPI signal representing true system behavior, not a random fluctuation.

### Phase 2 volume
```yaml
phase2:
  warmup_episodes: 50            # LLM warm-up: 50 episodes = 10 blocks of 5
                                   # LLM refines its rules across 10 feedback cycles
                                   # Based on InvAgent few-shot efficiency analysis
  warmup_blocks: 10              # = warmup_episodes / block_size
  rules_frozen_after_warmup: true  # After 10 blocks, LLM rulebook is locked

  eval_traces: 20                # Official 20-trace benchmark (frozen rules)
  # Total Phase 2 = 50 warm-up episodes + 20 eval traces = 70 episodes
```

### Why 10 feedback cycles is sufficient
LLMs are few-shot learners. Unlike the GRUs which need hundreds of gradient-based episodes, the LLM uses chain-of-thought reasoning to adapt in 2–10 examples. Beyond 50 episodes, the LLM shows diminishing returns in rule refinement. Fixing the rulebook after warm-up and then benchmarking gives a clean, reproducible evaluation.

---

## 2. System Prompt (Static — Set Once Before Phase 2 Begins)

```
You are a strategic advisor for a 4-stage supply chain optimization system.

SUPPLY CHAIN TOPOLOGY (downstream to upstream):
- Stage 1 (Retailer): Faces stochastic customer demand. Bullwhip originates here.
- Stage 2 (Warehouse): Supplies the Retailer. First order amplification point.
- Stage 3 (Distributor): Supplies the Warehouse. Second amplification point.
- Stage 4 (Manufacturer): Most upstream. Produces goods with infinite raw material access.

Orders flow upstream: Customer → Retailer → Warehouse → Distributor → Manufacturer
Goods flow downstream with fixed lead time L=4 at each hop.

THE α PARAMETER:
Each agent has an individual α ∈ [0.05, 0.95] that controls its incentive structure:
- α close to 1.0: agent minimizes its own cost (self-interested, may hoard inventory)
- α close to 0.0: agent minimizes total system cost (cooperative, shares burden)
- Best performance typically requires α between 0.1 and 0.9 per agent

COST STRUCTURE:
- Holding cost: h=1 per unit of inventory per period (uniform across all agents)
- Backlog cost: b=1 per unit of unmet demand per period (uniform across all agents)
- Lead time: L=4 periods at each echelon

YOUR TASK:
At the end of each 5-episode evaluation block, you receive:
- Input A: Statistical summary of Merton Jump Diffusion demand this block
- Input B: Averaged KPI performance metrics for this block
- Input C: Your previous 3 blocks of decisions and their actual outcomes

Analyze the multi-signal tensions, identify conflicts, and set α for each agent.

HARD CONSTRAINTS (enforced programmatically — output will be clipped if violated):
- All α values must be in [0.05, 0.95]
- Maximum α change per block: 0.25 from previous value
- After smoothing (0.8 × prev + 0.2 × your value), changes are gradual

OUTPUT FORMAT — respond ONLY with valid JSON, no other text:
{
  "reasoning": "<2-4 sentences: identify the key tension, explain your tradeoff>",
  "self_correction": "<1 sentence: mistake from previous block, or 'No correction needed'>",
  "compiled_rule": "<1 sentence: domain rule you are adding to your rulebook this block>",
  "alpha_stage_1_retailer": float,
  "alpha_stage_2_warehouse": float,
  "alpha_stage_3_distributor": float,
  "alpha_stage_4_manufacturer": float
}
```

**Note on `compiled_rule`:** This field is new compared to Phase 1 design. During warm-up, the LLM writes one rule per block (e.g., "When bullwhip > 2.5 AND service level > 0.85, reduce upstream α by 0.15"). These rules accumulate in a running rulebook. After warm-up, the rulebook is extracted and injected into the system prompt for the official 20-trace evaluation. The LLM does not revise rules during official evaluation — it only applies them.

---

## 3. Input A — Demand Statistics

```python
def build_input_a(demand_trace_block):
    demands = np.array(demand_trace_block)
    mean = np.mean(demands)
    std  = np.std(demands)
    spikes = [t for t, d in enumerate(demands) if d > mean + 2 * std]
    return {
        "demand_mean":          round(float(mean), 2),
        "demand_variance":      round(float(np.var(demands)), 2),
        "demand_skewness":      round(float(scipy.stats.skew(demands)), 3),
        "demand_kurtosis":      round(float(scipy.stats.kurtosis(demands)), 3),
        "n_demand_spikes":      len(spikes),
        "max_spike_magnitude":  round(float(np.max(demands) / mean), 2) if mean > 0 else 1.0,
        "spike_periods":        spikes[:5]
    }
```

**Prompt injection:**
```
DEMAND ENVIRONMENT (Input A):
Mean: {demand_mean} | Variance: {demand_variance}
Skewness: {demand_skewness} (positive = right-skewed spike events)
Kurtosis: {demand_kurtosis} (>3 = fat tails, sudden jumps typical of Merton JD)
Spike events this block: {n_demand_spikes} periods exceeded 2σ above mean
Maximum spike: {max_spike_magnitude}× normal demand
```

---

## 4. Input B — KPI Summary

```python
def build_input_b(block_metrics_history, current_block_metrics):
    def trend(history, key, threshold=0.05):
        if len(history) < 2: return "insufficient_data"
        recent = [h[key] for h in history[-3:]]
        delta = (recent[-1] - recent[0]) / (abs(recent[0]) + 1e-8)
        if key in ["total_cost", "bullwhip_ratio", "recovery_time", "reward_variance"]:
            if delta < -threshold: return "improving"
            if delta > threshold:  return "degrading"
        else:
            if delta > threshold:  return "improving"
            if delta < -threshold: return "degrading"
        return "stable"

    m = current_block_metrics
    return {
        "mean_total_cost":    round(m["total_cost"], 1),
        "cost_trend":         trend(block_metrics_history, "total_cost"),
        "individual_costs": {
            "stage_1_retailer":     round(m["costs"]["retailer"], 1),
            "stage_2_warehouse":    round(m["costs"]["warehouse"], 1),
            "stage_3_distributor":  round(m["costs"]["distributor"], 1),
            "stage_4_manufacturer": round(m["costs"]["manufacturer"], 1)
        },
        "mean_bullwhip_ratio":          round(m["bullwhip"], 3),
        "bullwhip_trend":               trend(block_metrics_history, "bullwhip"),
        "mean_service_level":           round(m["service_level"], 4),
        "service_level_trend":          trend(block_metrics_history, "service_level"),
        "mean_recovery_time":           round(m["recovery_time"], 1),
        "inter_agent_reward_variance":  round(m["reward_variance"], 4),
        "conflicting_signals":          (m["bullwhip"] > 2.0 and m["service_level"] < 0.85)
    }
```

**Prompt injection:**
```
SYSTEM PERFORMANCE (Input B):
Total cost: {mean_total_cost} ({cost_trend})
Per-agent costs: Retailer={stage_1} | Warehouse={stage_2} | Distributor={stage_3} | Manufacturer={stage_4}
Bullwhip ratio: {mean_bullwhip_ratio} ({bullwhip_trend}) [target: ≈1.0]
Service level: {mean_service_level:.1%} ({service_level_trend}) [target: ≥80%]
Recovery time: {mean_recovery_time} periods
Inter-agent reward variance: {inter_agent_reward_variance}
⚠ Conflicting signals: {conflicting_signals}
```

---

## 5. Input C — LLM Historical Track Record (Last 3 Blocks)

```python
def build_input_c(llm_history):
    recent = llm_history[-3:]
    records = []
    for r in recent:
        records.append({
            "block_id":             r["block_id"],
            "your_reasoning":       r["reasoning"],
            "compiled_rule":        r.get("compiled_rule", "none"),
            "your_alpha_decisions": r["alphas"],
            "actual_total_cost":    round(r["actual_outcome"]["total_cost"], 1),
            "actual_bullwhip":      round(r["actual_outcome"]["bullwhip"], 3),
            "actual_service_level": round(r["actual_outcome"]["service_level"], 4),
            "prediction_correct":   r["prediction_correct"]
        })
    return records
```

---

## 6. α Application with Smoothing and Hard Caps

```python
SMOOTHING    = 0.8
MAX_CHANGE   = 0.25
ALPHA_MIN    = 0.05
ALPHA_MAX    = 0.95

def apply_alpha_update(prev_alphas, llm_alphas):
    applied = {}
    for agent in prev_alphas:
        clipped = np.clip(llm_alphas[agent], ALPHA_MIN, ALPHA_MAX)
        delta   = np.clip(clipped - prev_alphas[agent], -MAX_CHANGE, MAX_CHANGE)
        capped  = prev_alphas[agent] + delta
        applied[agent] = SMOOTHING * prev_alphas[agent] + (1 - SMOOTHING) * capped
    return applied
```

---

## 7. Rulebook Extraction (After Warm-Up)

After 10 warm-up blocks (50 episodes), extract all `compiled_rule` fields from LLM history:

```python
def extract_rulebook(llm_history):
    rules = [h["compiled_rule"] for h in llm_history if h.get("compiled_rule")]
    # Deduplicate and keep only rules that appeared in successful blocks
    # (blocks where total_cost improved vs previous block)
    successful_rules = [
        r for r, h in zip(rules, llm_history)
        if h["actual_outcome"]["total_cost"] < h.get("prev_cost", float("inf"))
    ]
    return successful_rules

# Inject rulebook into system prompt for official 20-trace evaluation:
RULEBOOK_PROMPT = "\n".join([f"RULE {i+1}: {r}" for i, r in enumerate(rulebook)])
```

---

## 8. Phase 2 Main Loop

```python
def run_phase2(env, frozen_actors, llm_client, config):
    current_alphas = {agent: 0.5 for agent in agents}
    llm_history, block_metrics_history = [], []

    # ── WARM-UP PHASE (10 blocks) ──────────────────────────────────────
    for block_id in range(config["warmup_blocks"]):  # 10 blocks × 5 episodes = 50
        block_metrics, block_demands = run_evaluation_block(
            env, frozen_actors, current_alphas, n_episodes=config["block_size"])
        block_metrics_history.append(block_metrics)

        prompt   = build_full_prompt(build_input_a(block_demands),
                                     build_input_b(block_metrics_history, block_metrics),
                                     build_input_c(llm_history))
        response = llm_client.query(prompt)
        llm_alphas, reasoning, rule = parse_llm_response(response, current_alphas)

        llm_history.append({
            "block_id": block_id, "reasoning": reasoning,
            "compiled_rule": rule, "alphas": llm_alphas,
            "actual_outcome": block_metrics,
            "prediction_correct": assess_prediction(reasoning, block_metrics),
            "prev_cost": block_metrics_history[-2]["total_cost"]
                         if len(block_metrics_history) > 1 else float("inf")
        })
        current_alphas = apply_alpha_update(current_alphas, llm_alphas)

    # ── FREEZE RULEBOOK ────────────────────────────────────────────────
    rulebook = extract_rulebook(llm_history)
    save_rulebook(rulebook, "results/phase2/llm_rulebook.json")

    # ── OFFICIAL 20-TRACE EVALUATION ───────────────────────────────────
    eval_results = []
    for trace_id in range(20):
        # Reset to fixed holdout trace seed
        env.set_eval_seed(EVAL_SEEDS[trace_id])
        current_alphas = {agent: 0.5 for agent in agents}  # reset to neutral start
        trace_metrics, trace_demands = run_evaluation_block(
            env, frozen_actors, current_alphas, n_episodes=1)

        # LLM uses frozen rulebook — no rule updates
        prompt   = build_full_prompt_with_rulebook(
            build_input_a(trace_demands),
            build_input_b([trace_metrics], trace_metrics),
            rulebook=rulebook)
        response = llm_client.query(prompt)
        llm_alphas, reasoning, _ = parse_llm_response(response, current_alphas)

        # Execute one episode with LLM-assigned α
        final_metrics, _ = run_evaluation_block(
            env, frozen_actors,
            apply_alpha_update(current_alphas, llm_alphas),
            n_episodes=1)
        eval_results.append(final_metrics)

    save_eval_results(eval_results, "results/phase2/official_eval.json")
    return eval_results, llm_history, rulebook
```

---

## 9. Official Evaluation Protocol

```yaml
evaluation:
  n_eval_traces: 20              # 20 fixed holdout traces
  n_training_repeats: 5          # Run full Phase1+Phase2 pipeline 5 times
                                   # Report mean ± 95% CI
  final_eval_traces: 50          # For final paper numbers: 50 traces
  demand_models: [merton_jd, poisson]
  disruption_scenarios: [none, mild, severe]
```

---

## 10. What Phase 2 Must Show

**Primary empirical claim:** When `conflicting_signals=True` (high bullwhip AND low service level simultaneously), LLM meta-controller outperforms rule-based α controller on at least one metric. This is the scenario where LLM multi-signal reasoning is non-redundant with single-signal rules.

**Qualitative evidence for paper:**
- Sample chain-of-thought traces at key decision points
- α trajectory over blocks per agent
- Cases where `self_correction` led to improved subsequent block
- Evolution of compiled rulebook across warm-up blocks

**Honest reporting if LLM underperforms rule-based:**
> "Block-level gradient-free optimization did not consistently surpass single-signal heuristics in stationary supply chain environments. However, analysis of chain-of-thought traces reveals [insight]. The compiled rulebook demonstrates [domain adaptation]. This motivates future gradient-informed LLM integration."
