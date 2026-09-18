import numpy as np
import scipy.stats
from typing import List, Dict, Any, Optional

SYSTEM_PROMPT_INTRA = """You are an adaptive intra-episodic strategic advisor for a 4-stage supply chain optimization system.

SUPPLY CHAIN TOPOLOGY (downstream to upstream):
- Stage 1 (Retailer): Directly faces stochastic customer demand.
- Stage 2 (Warehouse): Supplies the Retailer.
- Stage 3 (Distributor): Supplies the Warehouse.
- Stage 4 (Manufacturer): Supplies the Distributor with infinite raw material access.

PHYSICAL DYNAMICS & LEAD TIME LAG:
- There is a physical lead time of L=4 periods between each echelon (orders placed at t arrive at t+4).
- Holding cost h=1 and backlog cost b=1 per unit/period.
- CRITICAL DYNAMICS NOTE: Because of the 4-period pipeline delay, panic-ordering or setting high selfish alphas during demand shocks or backlogs DOES NOT deliver instant inventory. Instead, erratic spike orders severely amplify upstream variance (Bullwhip Effect), causing upstream stockouts and escalating total chain costs.

THE COOPERATION PARAMETER alpha in [0.05, 0.95]:
Each agent observes an individual alpha parameter across the full range [0.05, 0.95] that controls its objective function:
- Low alpha (0.05 - 0.35, COOPERATIVE): Agent prioritizes total supply chain cost. Upstream agents with low alpha smooth their replenishment orders, reducing the Bullwhip Effect and ensuring steady downstream inventory flow during demand jumps.
- High alpha (0.65 - 0.95, SELFISH): Agent prioritizes its own local inventory buffer. This is effective when an echelon is heavily overstocked with excess holding inventory and needs to throttle incoming orders. Setting high alpha during backlogs is counterproductive due to the L=4 pipeline delay.
- Medium alpha (0.35 - 0.65, BALANCED): Standard intermediate balancing between local and system objectives.

INTRA-EPISODIC CONTEXT:
You are intervening LIVE within an active 200-timestep supply chain execution at regular intervals (e.g., every 50 timesteps).
Your decisions take immediate effect for the next execution window.

YOUR TASK:
Analyze the demand profile, agent bottlenecks (stock-starved vs inventory-heavy), and system-wide bullwhip amplification from the recent window. Then assign targeted alpha values [alpha_retailer, alpha_warehouse, alpha_distributor, alpha_manufacturer] in [0.05, 0.95] to minimize total supply chain cost and maintain high service levels while controlling order variance.

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown outside code fences:
{
  "reasoning": "<2-3 sentences explaining the key bottleneck observed in this window and how your alpha adjustments account for lead time lag and bullwhip control>",
  "compiled_rule": "<1 concise operational rule derived from this decision>",
  "alphas": {
    "retailer": <float between 0.05 and 0.95>,
    "warehouse": <float between 0.05 and 0.95>,
    "distributor": <float between 0.05 and 0.95>,
    "manufacturer": <float between 0.05 and 0.95>
  }
}"""


def build_window_input_a(window_demands: List[float]) -> Dict[str, Any]:
    """Extracts statistical demand characteristics from the recent time window."""
    demands = np.array(window_demands, dtype=np.float64)
    if len(demands) == 0:
        return {
            "demand_mean": 0.0,
            "demand_variance": 0.0,
            "demand_skewness": 0.0,
            "demand_kurtosis": 0.0,
            "n_demand_spikes": 0,
            "max_spike_magnitude": 1.0,
        }
    
    mean = float(np.mean(demands))
    variance = float(np.var(demands))
    std = float(np.std(demands))
    
    threshold = mean + 2.0 * std if std > 1e-4 else mean + 1.0
    spikes = [int(d) for d in demands if d > threshold]
    
    skewness = float(scipy.stats.skew(demands)) if len(demands) > 2 and std > 1e-4 else 0.0
    kurtosis = float(scipy.stats.kurtosis(demands)) if len(demands) > 3 and std > 1e-4 else 0.0
    max_spike_mag = float(np.max(demands) / mean) if mean > 1e-4 else 1.0
    
    return {
        "demand_mean": round(mean, 2),
        "demand_variance": round(variance, 2),
        "demand_skewness": round(skewness, 3),
        "demand_kurtosis": round(kurtosis, 3),
        "n_demand_spikes": len(spikes),
        "max_spike_magnitude": round(max_spike_mag, 2),
    }


def build_window_input_b(
    window_orders: List[List[float]],
    window_inventories: List[List[float]],
    window_backlogs: List[List[float]],
    window_demands: List[float],
    window_fulfilled: List[float],
    current_alphas: List[float]
) -> Dict[str, Any]:
    """Computes physical and financial metrics for each agent over the window."""
    agent_names = ["retailer", "warehouse", "distributor", "manufacturer"]
    num_agents = 4
    window_len = len(window_demands)
    
    per_agent_stats = {}
    total_system_cost = 0.0
    agent_costs = []
    fluctuations = []
    
    for i in range(num_agents):
        inv_i = np.array(window_inventories[i], dtype=np.float64)
        backlog_i = np.array(window_backlogs[i], dtype=np.float64)
        orders_i = np.array(window_orders[i], dtype=np.float64)
        
        avg_inv = float(np.mean(inv_i)) if len(inv_i) > 0 else 0.0
        avg_backlog = float(np.mean(backlog_i)) if len(backlog_i) > 0 else 0.0
        
        # Cost: h=1, b=1
        holding_cost = float(np.sum(inv_i))
        backlog_cost = float(np.sum(backlog_i))
        agent_cost = holding_cost + backlog_cost
        agent_costs.append(agent_cost)
        total_system_cost += agent_cost
        
        # Order statistics
        mean_ord = float(np.mean(orders_i)) if len(orders_i) > 0 else 0.0
        std_ord = float(np.std(orders_i)) if len(orders_i) > 0 else 0.0
        var_ord = float(np.var(orders_i)) if len(orders_i) > 0 else 0.0
        f_i = float(std_ord / (mean_ord + 1e-4))
        fluctuations.append(f_i)
        
        # Bottleneck classification
        ratio = avg_inv / (avg_backlog + 1e-4)
        if ratio < 0.2:
            bottleneck = "stock_starved (severe backlog)"
            recommended_action = "lower upstream alpha for steady replenishment flow; avoid downstream panic ordering"
        elif ratio > 5.0:
            bottleneck = "inventory_heavy (excess holding)"
            recommended_action = "raise local alpha to throttle replenishment orders and drain excess stock"
        else:
            bottleneck = "balanced"
            recommended_action = "maintain cooperative balance"
            
        per_agent_stats[agent_names[i]] = {
            "current_alpha": round(current_alphas[i], 3),
            "avg_inventory": round(avg_inv, 2),
            "avg_backlog": round(avg_backlog, 2),
            "inv_backlog_ratio": round(ratio, 2),
            "bottleneck_type": bottleneck,
            "recommended_action": recommended_action,
            "avg_order_qty": round(mean_ord, 2),
            "order_variance": round(var_ord, 2),
            "fluctuation_F": round(f_i, 3),
            "window_cost": round(agent_cost, 1)
        }
        
    # Cost shares
    for i, name in enumerate(agent_names):
        share = (agent_costs[i] / total_system_cost * 100.0) if total_system_cost > 0 else 25.0
        per_agent_stats[name]["cost_share_pct"] = round(share, 1)
        
    # Bullwhip amplification over window (F4 / F1 - 1)
    f1 = fluctuations[0]
    f4 = fluctuations[3]
    bullwhip_amp = ((f4 / (f1 + 1e-6)) - 1.0) * 100.0
    
    # Service level at Retailer
    total_demanded = float(np.sum(window_demands))
    total_fulfilled = float(np.sum(window_fulfilled))
    service_level = (total_fulfilled / total_demanded) if total_demanded > 0 else 1.0
    
    per_period_cost = total_system_cost / float(window_len) if window_len > 0 else 0.0
    
    return {
        "window_len": window_len,
        "per_period_cost": round(per_period_cost, 2),
        "total_window_cost": round(total_system_cost, 1),
        "service_level_pct": round(service_level * 100.0, 2),
        "bullwhip_amplification_pct": round(bullwhip_amp, 2),
        "retailer_fluctuation_F1": round(f1, 3),
        "manufacturer_fluctuation_F4": round(f4, 3),
        "agents": per_agent_stats
    }


def build_intra_prompt(
    timestep: int,
    window_interval: int,
    input_a: Dict[str, Any],
    input_b: Dict[str, Any],
    decision_history: List[Dict[str, Any]],
    current_alphas: List[float]
) -> str:
    """Builds the comprehensive user prompt for intra-episodic LLM intervention."""
    window_start = timestep - window_interval
    window_end = timestep
    
    prompt = f"=== LIVE INTRA-EPISODIC STATUS (Timestep {timestep} / 200, Window [{window_start} .. {window_end}]) ===\n\n"
    
    prompt += "--- INPUT A: DEMAND ENVIRONMENT (Recent Window) ---\n"
    prompt += f"Demand Mean: {input_a.get('demand_mean')} | Variance: {input_a.get('demand_variance')}\n"
    prompt += f"Skewness: {input_a.get('demand_skewness')} | Kurtosis: {input_a.get('demand_kurtosis')}\n"
    prompt += f"Spike Events: {input_a.get('n_demand_spikes')} periods (> 2 std) | Max Spike Ratio: {input_a.get('max_spike_magnitude')}x\n\n"
    
    prompt += "--- INPUT B: SYSTEM & AGENT PERFORMANCE (Recent Window) ---\n"
    prompt += f"Window Per-Period Cost: {input_b.get('per_period_cost')} | Service Level: {input_b.get('service_level_pct')}%\n"
    prompt += f"Bullwhip Amplification (F4/F1 - 1): {input_b.get('bullwhip_amplification_pct')}%\n"
    prompt += f"Order Fluctuation: F1(Retailer)={input_b.get('retailer_fluctuation_F1')} -> F4(Manufacturer)={input_b.get('manufacturer_fluctuation_F4')}\n\n"
    
    prompt += "Agent Breakdown:\n"
    for agent, stats in input_b.get("agents", {}).items():
        prompt += f"- {agent.upper()}:\n"
        prompt += f"    Current alpha: {stats['current_alpha']} | Cost Share: {stats['cost_share_pct']}%\n"
        prompt += f"    Avg Inv: {stats['avg_inventory']} | Avg Backlog: {stats['avg_backlog']} -> Bottleneck: {stats['bottleneck_type']}\n"
        prompt += f"    Avg Order: {stats['avg_order_qty']} (Variance: {stats['order_variance']}, F: {stats['fluctuation_F']})\n"
        prompt += f"    Guidance: {stats['recommended_action']}\n"
        
    prompt += "\n--- INPUT C: INTRA-EPISODIC DECISION HISTORY (Prior Windows in This Episode) ---\n"
    if not decision_history:
        prompt += "This is the first intervention in this 200-timestep episode. Initial alphas were: " + str(current_alphas) + "\n"
    else:
        for item in decision_history[-3:]:
            t_prev = item.get("timestep", 0)
            prompt += f"Timestep t={t_prev}:\n"
            prompt += f"  Alphas Set: {item.get('alphas')}\n"
            prompt += f"  Reasoning: {item.get('reasoning')}\n"
            prompt += f"  Subsequent Window Per-Period Cost: {item.get('observed_cost', 'N/A')} | Service Level: {item.get('observed_service_level', 'N/A')}%\n"
            
    prompt += "\nINSTRUCTIONS:\n"
    prompt += "1. Diagnose whether the primary issue right now is Backlog/Service Level vs Excess Inventory vs Upstream Bullwhip Amplification.\n"
    prompt += "2. Determine the optimal alpha for each of the 4 stages for the next execution window.\n"
    prompt += "3. Respond ONLY with valid JSON containing 'reasoning', 'compiled_rule', and 'alphas'."
    
    return prompt
