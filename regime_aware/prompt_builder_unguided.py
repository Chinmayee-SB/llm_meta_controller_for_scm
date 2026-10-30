"""
prompt_builder_unguided.py

Defines the system prompt and dynamic prompt construction functions for the
Intra-Episodic Meta-LLM Strategic Supervisor (Unguided Variant) controlling the
Tri-State MARL 4-echelon supply chain model.

Based on .legacy_backup/experiments/tristate_parallel_threads/prompt_builder_tristate.py,
with all numeric switching threshold rules removed (demand >= 14, demand < 13,
backlog > 40, amplification > 150%).
"""

import numpy as np
from typing import List, Dict, Any, Optional

SYSTEM_PROMPT_UNGUIDED = """You are the Intra-Episodic Meta-LLM Strategic Supervisor for an autonomous 4-stage supply chain optimization system.

================================================================================
1. ENVIRONMENT TOPOLOGY & PHYSICAL GROUND TRUTH
================================================================================
You govern a linear, 4-echelon stochastic multi-echelon supply chain structured as:
  Stage 1: Retailer (most downstream) — directly serves external customer demand.
  Stage 2: Wholesaler — fulfills orders placed by the Retailer.
  Stage 3: Distributor — fulfills orders placed by the Wholesaler.
  Stage 4: Factory / Manufacturer (most upstream) — fulfills orders placed by the Distributor, with infinite raw material access.

PHYSICAL DYNAMICS & PIPELINE DELAYS:
- Transportation & Production Lead Time: L = 4 periods between each tier.
  When Stage i places an replenishment order O_t^i at period t, the goods enter a 4-period in-transit pipeline (P_4, P_3, P_2, P_1) and arrive at Stage i's on-hand inventory exactly at period t + 4.
- Unmet Demand & Backlog: If incoming orders exceed available on-hand stock plus arriving shipments, unmet demand is backlogged (B_t^i) and carries over to subsequent periods.
- Economic Cost Structure (Undiscounted):
  * Holding cost coefficient: h = 1 per unit per period on positive on-hand inventory (I_t^i).
  * Backlog penalty coefficient: b = 1 per unit per period on unsatisfied orders (B_t^i).
  * Total Supply Chain Cost per period: C_t^total = sum_{i=1}^4 (I_t^i + B_t^i).
  * The baseline benchmark target established in literature is ~34.00 total cost per period.
- Demand Environment: Customer demand follows a Merton Jump Diffusion (MJD) stochastic process (a continuous Brownian motion baseline punctuated by random, sudden Poisson jump shocks where demand spikes 2x-3x above baseline).

THE BULLWHIP PHENOMENON & LEAD TIME LAG:
- Order variability naturally amplifies as orders travel upstream from Retailer to Factory (Bullwhip Effect).
- CRITICAL PHYSICAL LAW: Because of the L = 4 pipeline delay, panic-ordering or setting high selfish priorities during sudden demand shocks DOES NOT deliver instant inventory. It merely causes upstream order variance explosion, starving upstream stages and inflating total chain costs.

================================================================================
2. THE MULTI-AGENT RL ACTOR SUBSTRATUM
================================================================================
Beneath your strategic supervision, each of the 4 supply chain stages is actuated by a trained Multi-Agent Reinforcement Learning (MARL) neural network actor.
- Decentralized Observations: Each actor observes its local state [I, B, incoming downstream order, 4 in-transit pipeline slots] plus an operational cooperation parameter alpha.
- Action Space: Each actor selects discrete order quantities from {0, 1, ..., 20}.
- Centralized Training: The actors were trained using Centralized Training with Decentralized Execution (CTDE) under a tri-state curriculum.

================================================================================
3. THE DISCRETE TRI-STATE OPERATIONAL SPECTRUM
================================================================================
The underlying MARL actors are explicitly trained to operate across three distinct operational states, determined by the parameter alpha:

[State 1] CRUISER (alpha = 0.375) — The Steady-State Anchor (70% training prior)
  * Reward Weighting: 37.5% local echelon cost / 62.5% global system cost.
  * Optimal operational equilibrium reproducing the benchmark (~34 cost per period).
  * WHEN TO USE: Nominal, stationary demand; balanced inventories across stages; low backlogs.
  * BEHAVIOR: Smooth replenishment, minimal order fluctuation, optimal long-run cost efficiency.

[State 2] SHOCK_ABSORBER (alpha = 0.10) — Volatility Suppression (15% training prior)
  * Reward Weighting: 10% local echelon cost / 90% global system cost.
  * Maximum cooperative alignment across all echelons.
  * WHEN TO USE: ONLY when upstream bullwhip amplification is severely out of control AND upstream inventory is dangerously overflowing.
  * BEHAVIOR: Upstream agents prioritize chain-wide stability over local inventory hoarding, dampening order variance and preventing bullwhip cascades.
  * CAUTION: DO NOT use SHOCK_ABSORBER during quiet periods or mild demand. Clamping alpha to 0.10 starves the supply chain of healthy buffers, leaving it highly vulnerable to the next sudden demand spike.

[State 3] BACKLOG_CLEARER (alpha = 0.65) — Rapid Pipeline Recovery (15% training prior)
  * Reward Weighting: 65% local echelon cost / 35% global system cost.
  * Local stock replenishment priority.
  * WHEN TO USE: Immediately after a demand spike subsides when customer backlogs are elevated and downstream stages are stock-starved (I=0, B>0).
  * BEHAVIOR: Accelerates order placement to rapidly replenish exhausted inventory buffers and clear customer backlogs.
  * CAUTION: Do NOT sustain BACKLOG_CLEARER for more than 1 consecutive window if the 4-period pipeline is already filled with incoming stock, or excessive holding costs will occur.

================================================================================
4. YOUR INTERVENTION PROTOCOL (EVERY 20 TIMESTEPS)
================================================================================
The simulation spans T = 200 timesteps per episode.
You intervene LIVE at the boundary of every 20-step window (t = 0, 20, 40, 60, 80, 100, 120, 140, 160, 180).
Your assigned alpha state dictates agent behavior for the upcoming 20-step execution window.
A 20-period interval is optimal because it aligns with the 16-period physical end-to-end conveyor lag (4 tiers x 4 lead time), allowing replenishment waves to land before re-evaluating.

CRITICAL ASYMMETRIC RECOVERY RULES:
1. Moderate Downstream Backlog (Retailer backlogged, Warehouse inventory healthy):
   Assign TARGETED_HYBRID:
   * Retailer = 0.65 (BACKLOG_CLEARER: pulls inventory to clear customer demand).
   * Wholesaler, Distributor, Factory = 0.10 (SHOCK_ABSORBER: 90% cooperative, flushes stock downstream without hoarding).

2. Severe Multi-Tier Backlog & High Demand Saturation (Retailer and Warehouse backlogs are severe, or demand is heavily saturated):
   Assign MULTI_TIER_RECOVERY (Push-Pull Saturation Defense):
   * Retailer = 0.65 (Pulls urgently from Warehouse).
   * Warehouse = 0.65 (Pulls urgently from Distributor to resolve intermediate starvation).
   * Distributor = 0.10 (SHOCK_ABSORBER: Acts as a 90% cooperative frictionless pipe).
   * Manufacturer / Factory = 0.65 (FACTORY SATURATION PUSH: Forces the factory to produce at max capacity to flood the 16-period pipeline with stock. NEVER leave Factory at 0.10 during high demand saturation, as 0.10 dampens production and permanently starves the chain!).

OPERATIONAL TRANSITION HEURISTICS:
1. Cruising Normal (DEFAULT STRICT ANCHOR): If backlogs are zero (or near zero) and demand is moderate/stable, you MUST default to CRUISER (alpha = 0.375 across all tiers). This maintains healthy protective buffers to weather future spikes.
2. High Demand Saturation / Multi-Tier Crisis: If demand is heavily saturated or combined Retailer+Warehouse backlog is in crisis -> Switch to MULTI_TIER_RECOVERY (Retailer=0.65, Warehouse=0.65, Distributor=0.10, Manufacturer=0.65).
3. Downstream-Only Recovery: If demand has normalized but Retailer has lingering backlogs -> Switch to TARGETED_HYBRID (Retailer=0.65, others=0.10).
4. Extreme Bullwhip Overflow: ONLY if backlogs are zero AND upstream bullwhip amplification is severely elevated AND upstream inventories are overflowing -> Switch to SHOCK_ABSORBER (alpha = 0.10 across all tiers). Never use this just because demand is quiet.

================================================================================
5. OUTPUT FORMAT — STRICT JSON ONLY
================================================================================
Respond ONLY with a valid JSON object. No preamble, no markdown outside the JSON block:
{
  "window_diagnosis": "<2-3 sentences diagnosing recent demand volatility, echelon inventory/backlog bottlenecks, and bullwhip amplification>",
  "target_mode": "<CRUISER | SHOCK_ABSORBER | BACKLOG_CLEARER | TARGETED_HYBRID | MULTI_TIER_RECOVERY>",
  "assigned_alphas": {
    "retailer": <0.375 | 0.10 | 0.65>,
    "warehouse": <0.375 | 0.10 | 0.65>,
    "distributor": <0.375 | 0.10 | 0.65>,
    "manufacturer": <0.375 | 0.10 | 0.65>
  },
  "operational_rationale": "<1-2 concise sentences explaining why this state assignment optimizes cost and stability for the next 20 periods>"
}"""


def build_tristate_window_demand_features(window_demands: List[float]) -> Dict[str, Any]:
    """Extracts statistical demand metrics over the window."""
    demands = np.array(window_demands, dtype=np.float64)
    if len(demands) == 0:
        return {
            "demand_mean": 0.0,
            "demand_std": 0.0,
            "demand_variance": 0.0,
            "demand_min": 0.0,
            "demand_max": 0.0,
            "n_demand_spikes": 0,
            "max_spike_ratio": 1.0,
        }

    mean_d = float(np.mean(demands))
    std_d = float(np.std(demands))
    var_d = float(np.var(demands))
    min_d = float(np.min(demands))
    max_d = float(np.max(demands))

    threshold = mean_d + 1.8 * std_d if std_d > 1e-4 else mean_d + 2.0
    spikes = [float(d) for d in demands if d > threshold]
    max_ratio = float(max_d / (mean_d + 1e-4))

    return {
        "demand_mean": round(mean_d, 2),
        "demand_std": round(std_d, 2),
        "demand_variance": round(var_d, 2),
        "demand_min": round(min_d, 2),
        "demand_max": round(max_d, 2),
        "n_demand_spikes": len(spikes),
        "max_spike_ratio": round(max_ratio, 2)
    }


def build_tristate_window_echelon_features(
    window_orders: List[List[float]],
    window_inventories: List[List[float]],
    window_backlogs: List[List[float]],
    window_demands: List[float],
    current_alphas: List[float]
) -> Dict[str, Any]:
    """Extracts physical inventory, backlog, cost, and bullwhip metrics over the window."""
    agent_names = ["retailer", "warehouse", "distributor", "manufacturer"]
    num_agents = 4
    window_len = max(len(window_demands), 1)

    echelon_stats = {}
    total_system_cost = 0.0
    fluctuations = []

    for i in range(num_agents):
        inv_seq = np.array(window_inventories[i], dtype=np.float64) if len(window_inventories[i]) > 0 else np.array([0.0])
        backlog_seq = np.array(window_backlogs[i], dtype=np.float64) if len(window_backlogs[i]) > 0 else np.array([0.0])
        ord_seq = np.array(window_orders[i], dtype=np.float64) if len(window_orders[i]) > 0 else np.array([0.0])

        avg_inv = float(np.mean(inv_seq))
        avg_b = float(np.mean(backlog_seq))
        current_inv = float(inv_seq[-1])
        current_b = float(backlog_seq[-1])

        holding_cost = float(np.sum(inv_seq))
        backlog_cost = float(np.sum(backlog_seq))
        stage_cost = holding_cost + backlog_cost
        total_system_cost += stage_cost

        mean_ord = float(np.mean(ord_seq))
        std_ord = float(np.std(ord_seq))
        f_i = float(std_ord / (mean_ord + 1e-4))
        fluctuations.append(f_i)

        if avg_b > 2.0 and avg_inv < 1.0:
            status = "STOCK_STARVED (Persistent Backlog)"
        elif avg_inv > 15.0 and avg_b == 0.0:
            status = "INVENTORY_HEAVY (Excess Holding Buffer)"
        elif avg_b > 0.0:
            status = "TRANSIENT_BACKLOG"
        else:
            status = "HEALTHY_BALANCED"

        echelon_stats[agent_names[i]] = {
            "current_alpha": round(current_alphas[i], 3),
            "avg_inventory": round(avg_inv, 2),
            "current_inventory": round(current_inv, 2),
            "avg_backlog": round(avg_b, 2),
            "current_backlog": round(current_b, 2),
            "status": status,
            "avg_order": round(mean_ord, 2),
            "order_fluctuation_F": round(f_i, 3),
            "stage_window_cost": round(stage_cost, 1)
        }

    per_period_cost = total_system_cost / float(window_len)
    avg_chain_fluctuation = float(np.mean(fluctuations))

    f_downstream = fluctuations[0]
    f_upstream = fluctuations[3]
    bullwhip_amp_pct = float(((f_upstream / (f_downstream + 1e-4)) - 1.0) * 100.0)

    demands_arr = np.array(window_demands, dtype=np.float64)
    backlogs_0 = np.array(window_backlogs[0], dtype=np.float64) if len(window_backlogs[0]) > 0 else np.zeros_like(demands_arr)
    unfulfilled = np.sum(np.maximum(backlogs_0, 0.0))
    total_demanded = np.sum(demands_arr)
    service_level = max(0.0, float(1.0 - (unfulfilled / (total_demanded + 1e-4))))

    return {
        "per_period_cost": round(per_period_cost, 2),
        "total_window_cost": round(total_system_cost, 2),
        "service_level_pct": round(service_level * 100.0, 1),
        "avg_chain_fluctuation_F": round(avg_chain_fluctuation, 3),
        "bullwhip_amplification_pct": round(bullwhip_amp_pct, 2),
        "echelons": echelon_stats
    }


def build_tristate_user_prompt(
    timestep: int,
    window_interval: int,
    demand_features: Dict[str, Any],
    echelon_features: Dict[str, Any],
    current_alphas: List[float]
) -> str:
    """Builds the structured user prompt presented to the LLM supervisor at each interval."""
    agent_names = ["retailer", "warehouse", "distributor", "manufacturer"]
    current_mode = "CRUISER (0.375)" if np.isclose(current_alphas[0], 0.375) else (
        "SHOCK_ABSORBER (0.10)" if np.isclose(current_alphas[0], 0.10) else (
            "BACKLOG_CLEARER (0.65)" if np.isclose(current_alphas[0], 0.65) else "CUSTOM"
        )
    )

    prompt = f"""[INTERVENTION CHECKPOINT: TIMESTEP t = {timestep} / 200]
Recent Execution Window: Periods [t = {max(0, timestep - window_interval)} to t = {timestep}]
Current Active Operational State: {current_mode} (Alphas = {current_alphas})

--------------------------------------------------------------------------------
1. DEMAND ENVIRONMENT TELEMETRY (Last {window_interval} periods):
--------------------------------------------------------------------------------
- Mean Customer Demand: {demand_features['demand_mean']} units/period
- Demand Std Dev: {demand_features['demand_std']} (Variance: {demand_features['demand_variance']})
- Demand Range: [{demand_features['demand_min']}, {demand_features['demand_max']}]
- Jump Spikes Detected: {demand_features['n_demand_spikes']} (Max Spike Ratio: {demand_features['max_spike_ratio']}x mean)

--------------------------------------------------------------------------------
2. SUPPLY CHAIN PERFORMANCE & HEALTH:
--------------------------------------------------------------------------------
- Undiscounted Per-Period Cost: {echelon_features['per_period_cost']} (Benchmark Target: ~34.00)
- Customer Service Level: {echelon_features['service_level_pct']}%
- Average Chain Fluctuation (F): {echelon_features['avg_chain_fluctuation_F']} (Benchmark Target: 0.758)
- Upstream Bullwhip Amplification: {echelon_features['bullwhip_amplification_pct']}% (Benchmark Target: 92.37%)

--------------------------------------------------------------------------------
3. TIER-BY-TIER ECHELON BREAKDOWN:
--------------------------------------------------------------------------------
"""
    for name in agent_names:
        stats = echelon_features["echelons"][name]
        prompt += (
            f"• {name.upper()}: Status = {stats['status']} | "
            f"Avg Inv = {stats['avg_inventory']} (Curr = {stats['current_inventory']}) | "
            f"Avg Backlog = {stats['avg_backlog']} (Curr = {stats['current_backlog']}) | "
            f"Order Fluctuation F = {stats['order_fluctuation_F']} | "
            f"Window Cost = {stats['stage_window_cost']}\n"
        )

    prompt += f"""
--------------------------------------------------------------------------------
DECISION REQUIRED:
--------------------------------------------------------------------------------
Analyze whether the supply chain should remain in {current_mode} or transition to another state
for the next {window_interval}-period window [t = {timestep} to t = {min(200, timestep + window_interval)}].

Available Operational Modes:
- CRUISER (alpha = 0.375 across all tiers): Nominal steady state (~34 cost, low fluctuation).
- SHOCK_ABSORBER (alpha = 0.10 across all tiers): Dampens bullwhip during spikes and volatility surges.
- TARGETED_HYBRID (Retailer = 0.65, Wholesaler/Distributor/Factory = 0.10): Downstream backlog recovery when upstream is healthy.
- MULTI_TIER_RECOVERY (Retailer = 0.65, Warehouse = 0.65, Distributor = 0.10, Factory = 0.65): Multi-tier crisis and high-demand saturation (Factory pushes at max capacity).
- BACKLOG_CLEARER (alpha = 0.65 across all tiers): Widespread backlog clearance (use with caution).

Respond ONLY with the requested JSON schema.
"""
    return prompt

