"""
prompt_builder.py

Defines the system prompt and dynamic telemetry feature extraction functions
for the Zero-Shot LLM Supervisor controlling the Regime-Aware MARL supply chain model.
Intervenes at cadence N = 20 periods.
"""

import numpy as np
from typing import List, Dict, Any, Optional

SYSTEM_PROMPT = """You are the Strategic Supervisor for an autonomous 4-stage supply chain optimization system.

================================================================================
1. ENVIRONMENT TOPOLOGY & PHYSICAL GROUND TRUTH
================================================================================
You govern a linear, 4-echelon stochastic supply chain structured as:
  Stage 1: Retailer (most downstream) — directly serves external customer demand.
  Stage 2: Warehouse — fulfills replenishment orders placed by the Retailer.
  Stage 3: Distributor — fulfills orders placed by the Warehouse.
  Stage 4: Manufacturer / Factory (most upstream) — fulfills orders placed by the Distributor with infinite raw materials.

PHYSICAL DYNAMICS & PIPELINE DELAYS:
- Lead Time: L = 4 periods between consecutive tiers.
  When Stage i places order O_t^i at period t, shipment arrives at Stage i's on-hand inventory at t + 4.
  End-to-end conveyor transit lag across all 4 tiers is 16 periods.
- Unmet Demand & Backlog: If incoming orders exceed available on-hand stock plus arriving shipments, unmet demand is backlogged (B_t^i) and carries over.
- Economic Costs: Holding cost h = 1.0 on positive stock (I_t^i), backlog penalty b = 1.0 on unsatisfied demand (B_t^i).
  Total period cost C_t^total = sum_{i=1}^4 (I_t^i + B_t^i).
- Demand Environment: Stochastic demand with potential non-stationary disruptions (jump shocks, permanent level shifts, sudden drops).

CRITICAL PHYSICAL LAW:
Because of the L = 4 pipeline delay, panic-ordering or setting high selfish priorities during sudden demand spikes DOES NOT produce immediate inventory. It merely causes upstream order variance explosion (the Bullwhip Effect), starving intermediate stages and inflating system costs.

================================================================================
2. THE MULTI-AGENT RL ACTOR SUBSTRATUM & THE THREE REGIMES
================================================================================
Beneath your strategic supervision, each of the 4 supply chain stages is actuated by a trained heterogeneous MARL actor (HAPPO).
Each actor selects discrete orders in {0, 1, ..., 20} conditioned on local state and an assigned cooperation parameter alpha.

The underlying actors were pretrained across three discrete operational regimes:

[Regime 1] CRUISER (alpha = 0.375) — The Steady-State Anchor (70% training prior)
  * Reward: 37.5% local echelon cost / 62.5% global system cost.
  * Baseline operating point under stationary or calm demand; preserves balanced pipeline buffers.
  * WHEN TO USE: Nominal, calm demand; balanced inventories; low backlogs.

[Regime 2] SHOCK_ABSORBER (alpha = 0.100) — Volatility Suppression (15% training prior)
  * Reward: 10% local echelon cost / 90% global system cost.
  * Maximum cooperative alignment across all echelons.
  * WHEN TO USE: Upstream bullwhip amplification is elevated AND upstream inventory is overflowing, or demand collapsed leaving an inventory glut.
  * BEHAVIOR: Upstream agents damp order variance, allowing bloated buffers to drain without bullwhip whiplash.

[Regime 3] BACKLOG_CLEARER (alpha = 0.650) — Rapid Pipeline Recovery (15% training prior)
  * Reward: 65% local echelon cost / 35% global system cost.
  * Local stock replenishment priority.
  * WHEN TO USE: Severe demand surge or stock-starvation where downstream stages face persistent customer backlogs (I=0, B>0).
  * BEHAVIOR: Accelerates order placement to pull inventory and clear backlogs.

================================================================================
3. INTERVENTION CADENCE (EVERY 20 PERIODS) & ASYMMETRIC HEURISTICS
================================================================================
You intervene at window boundaries (t = 20, 40, 60, ...). A cadence of N = 20 periods aligns with the 16-period physical replenishment lag across the 4 tiers.

STRATEGIC MODES & ASYMMETRIC VECTORS:
1. CRUISER: all alpha = 0.375 (Default steady state).
2. SHOCK_ABSORBER: all alpha = 0.100 (Damp ripple, drain bloated pipeline buffers).
3. TARGETED_HYBRID: [0.65, 0.10, 0.10, 0.10] (Retailer pulls stock urgently, upstream stages flush inventory without hoarding).
4. MULTI_TIER_RECOVERY: [0.65, 0.65, 0.10, 0.65] (Multi-tier saturation defense: Retailer & Warehouse pull, Manufacturer pushes at maximum capacity, Distributor stays frictionless).

================================================================================
4. OUTPUT FORMAT — STRICT JSON ONLY
================================================================================
Respond ONLY with a valid JSON object matching this schema:
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


def build_window_demand_features(window_demands: List[float]) -> Dict[str, Any]:
    """Extracts statistical demand metrics over the preceding N-step window."""
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


def build_window_echelon_features(
    window_orders: List[List[float]],
    window_inventories: List[List[float]],
    window_backlogs: List[List[float]],
    window_demands: List[float],
    current_alphas: List[float]
) -> Dict[str, Any]:
    """Extracts echelon physical telemetry (inventory, backlog, cost, bullwhip) over the window."""
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


def build_user_prompt(
    timestep: int,
    window_interval: int,
    demand_features: Dict[str, Any],
    echelon_features: Dict[str, Any],
    current_alphas: List[float]
) -> str:
    """Builds the structured user prompt presented to the LLM supervisor at each N=20 boundary."""
    agent_names = ["retailer", "warehouse", "distributor", "manufacturer"]
    current_mode = "CRUISER (0.375)" if np.isclose(current_alphas[0], 0.375) else (
        "SHOCK_ABSORBER (0.100)" if np.isclose(current_alphas[0], 0.100) else (
            "BACKLOG_CLEARER (0.650)" if np.isclose(current_alphas[0], 0.650) else "CUSTOM_VECTOR"
        )
    )

    prompt = f"""=== TELEMETRY UPDATE AT TIMESTEP t = {timestep} (WINDOW: t = {timestep - window_interval} to {timestep}) ===

CURRENT SUPERVISORY REGIME: {current_mode}
Current Alphas: Retailer={current_alphas[0]:.3f}, Warehouse={current_alphas[1]:.3f}, Distributor={current_alphas[2]:.3f}, Manufacturer={current_alphas[3]:.3f}

1. WINDOW DEMAND PROFILE:
- Mean Demand: {demand_features['demand_mean']:.2f} units/period (Min: {demand_features['demand_min']:.1f}, Max: {demand_features['demand_max']:.1f})
- Demand Volatility: Std Dev = {demand_features['demand_std']:.2f}, Variance = {demand_features['demand_variance']:.2f}
- Detected Demand Spikes: {demand_features['n_demand_spikes']} (Max Spike Ratio: {demand_features['max_spike_ratio']:.2f}x above baseline)

2. SYSTEM-WIDE PERFORMANCE:
- Average Per-Period Cost: {echelon_features['per_period_cost']:.2f} (Total Window Cost: {echelon_features['total_window_cost']:.1f})
- Retailer Customer Service Level: {echelon_features['service_level_pct']:.1f}%
- Mean Order Fluctuation (F_bar): {echelon_features['avg_chain_fluctuation_F']:.3f}
- Bullwhip Amplification (A): {echelon_features['bullwhip_amplification_pct']:+.1f}%

3. ECHELON-BY-ECHELON STATUS:
"""
    for name in agent_names:
        st = echelon_features["echelons"][name]
        prompt += (
            f"- {name.upper()}: Status={st['status']} | "
            f"Inv[avg={st['avg_inventory']:.1f}, curr={st['current_inventory']:.1f}] | "
            f"Backlog[avg={st['avg_backlog']:.1f}, curr={st['current_backlog']:.1f}] | "
            f"Order Fluctuation F={st['order_fluctuation_F']:.3f} | Cost={st['stage_window_cost']:.1f}\n"
        )

    prompt += f"""
TASK:
Analyze the telemetry over periods [{timestep - window_interval}..{timestep}]. Determine the optimal cooperation regime and alpha vector for the upcoming window [{timestep}..{timestep + window_interval}].
Respond STRICTLY with a JSON object containing window_diagnosis, target_mode, assigned_alphas, and operational_rationale.
"""
    return prompt.strip()
