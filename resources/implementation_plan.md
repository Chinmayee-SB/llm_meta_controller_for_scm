# Implementation Plan: Episodic Meta LLM Demonstrations

This document outlines the implementation plan for the two interactive simulation demonstrations and their shared Streamlit UI.

## 1. Frozen Policy and Telemetry Strategy

> [!IMPORTANT]
> **Frozen Policy Loader:** We will reuse the exact same loader used in the main pipeline: `MADRL.training.run_phase2.load_frozen_policy`. The policy will be loaded from `"MADRL/results/phase1_weights/frozen_policy.pt"`.
> 
> **Telemetry Collection:** Additional telemetry (inventory, backlog, order quantities, shipments) **can be collected externally**. After each `env.step()`, the simulation scripts will directly inspect `env.inventory`, `env.backlog`, `env.action_history`, and `env.order` to record the per-timestep state. This satisfies the strict requirement of **no changes to the `serial.py` logic**.

## 2. Demo 1: Disruption Resilience Simulation (`MADRL/simulations/demo1_disruption.py`)

**What it does:** Runs a single 200-timestep episode with the frozen Phase 1 policy and injects a 20-step manufacturer shutdown at $t=80$. 

**Key Functions/Classes:**
- `run_disrupted_episode(env, frozen_actors, config)`: Main simulation loop.
- Disruption Injection: Instead of modifying `serial.py`, we will externally override the shipment pipeline. Immediately after `env.step()`, if $80 \le t < 100$, we will zero out the goods entering the distributor's pipeline (`env.order[2][-1] = 0`) and refund the manufacturer's inventory to perfectly simulate a failure to ship.
- `compute_recovery_metrics(trajectory)`: Analyzes the service level trace to identify the exact onset timestep, the trough (lowest service level), and the recovery timestep (when SL returns to $\ge 90\%$ of pre-disruption levels).

**JSON Schema (`MADRL/results/simulations/demo1_trace.json`):**
```json
{
  "metadata": {
    "disruption_start": 80,
    "disruption_end": 100,
    "affected_agent": "manufacturer",
    "disruption_type": "zero_shipment"
  },
  "recovery_metrics": {
    "disruption_onset_timestep": 80,
    "lowest_service_level_reached": 0.45,
    "recovery_timestep": 115,
    "recovery_time": 35
  },
  "trajectory": [
    {
      "timestep": 0,
      "inventory": [10, 10, 10, 10],
      "backlog": [0, 0, 0, 0],
      "order_quantity": [5, 5, 5, 5],
      "shipment_quantity": [5, 5, 5, 5],
      "service_level": 1.0,
      "cumulative_cost": 40.0
    }
  ]
}
```

## 3. Demo 2: Alpha Impact Simulation (`MADRL/simulations/demo2_alpha_impact.py`)

**What it does:** Runs 5 evaluation blocks (5 episodes each), consulting the real Anthropic LLM via `llm_client.py` after each block to adapt per-agent alphas.

**Key Functions/Classes:**
- `run_alpha_impact_demo(env, frozen_actors, llm_client, config)`: Loops 5 times. Reuses `build_input_a`, `build_input_b`, `build_input_c`, and `build_full_prompt` from `MADRL.llm.prompt_builder`, exactly as Phase 2 does.
- Adapts `run_evaluation_block` to aggregate the KPIs and consults `AlphaController` to apply updates and track the alpha trajectory.

**JSON Schema (`MADRL/results/simulations/demo2_trace.json`):**
```json
{
  "blocks": [
    {
      "block_id": 1,
      "alphas": {"retailer": 0.5, "warehouse": 0.5, "distributor": 0.5, "manufacturer": 0.5},
      "cost_shares": {"retailer": 0.3, "warehouse": 0.3, "distributor": 0.2, "manufacturer": 0.2},
      "total_cost": 4500.0,
      "service_level": 0.92,
      "bullwhip": 1.15,
      "llm_output": {
        "reasoning": "Distributor cost is disproportionately high...",
        "self_correction": "No correction needed",
        "compiled_rule": "Reduce alpha for middle echelons when..."
      }
    }
  ]
}
```

## 4. Shared UI (`MADRL/simulations/ui.py`)

**What it does:** A Streamlit dashboard that reads the generated JSON files (strict decoupling) and visualizes the experiments in two tabs.

**Dependencies:** `streamlit`, `pandas`, `plotly`, `numpy`

**Visualizations & Features:**
- **Tab 1: Disruption Resilience (Demo 1)**
  - Timeline chart plotting Service Level and Cumulative Cost with shaded regions for the disruption window and recovery period.
  - Interactive heatmaps (`plotly.express.imshow`) for Inventory and Backlog across all 4 agents and 200 timesteps.
  - Recovery metrics scorecard (Lowest SL, Recovery Time).
- **Tab 2: Alpha Impact (Demo 2)**
  - Multi-line chart showing the Alpha trajectory for each of the 4 agents across blocks.
  - Stacked bar chart showing Per-Agent Cost Share evolution.
  - KPI evolution charts (Total Cost, Bullwhip, Service Level).
  - An expandable text section detailing the LLM's causal reasoning, self-correction, and compiled rules for each block.
- **Causal Chain Explanation:** The UI will include Markdown text dynamically populated with LLM reasoning to clearly map how *Alpha changes $\rightarrow$ Inventory buffering $\rightarrow$ KPI improvements*.

## User Review Required
Please review the JSON schemas and the strategy for non-invasively injecting disruptions (modifying `env.order` mid-loop) to ensure they perfectly align with your requirements. Let me know if you approve this plan to begin implementation.
