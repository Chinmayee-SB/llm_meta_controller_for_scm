import numpy as np
import scipy.stats

SYSTEM_PROMPT = """You are a strategic advisor for a 4-stage supply chain optimization system.

SUPPLY CHAIN TOPOLOGY (downstream to upstream):
- Stage 1 (Retailer): Faces stochastic customer demand. Bullwhip originates here.
- Stage 2 (Warehouse): Supplies the Retailer. First order amplification point.
- Stage 3 (Distributor): Supplies the Warehouse. Second amplification point.
- Stage 4 (Manufacturer): Most upstream. Produces goods with infinite raw material access.

Orders flow upstream: Customer -> Retailer -> Warehouse -> Distributor -> Manufacturer
Goods flow downstream with fixed lead time L=4 at each hop.

THE alpha PARAMETER:
Each agent has an individual alpha in [0.05, 0.95] that controls its incentive structure:
- alpha close to 1.0: agent minimizes its own cost (self-interested, may hoard inventory)
- alpha close to 0.0: agent minimizes total system cost (cooperative, shares burden)
- Best performance typically requires alpha between 0.1 and 0.9 per agent

AGENT-SPECIFIC ANALYSIS REQUIRED:
You must reason about each agent individually, not just the global system.
For each agent, consider:
  - Are they expensive because they hoard inventory (inventory_heavy, high avg_inventory)?
  - Or are they expensive because they are starved of stock (stock_starved, high backlog)?
  - Are their backlogs or inventories rapidly increasing/decreasing (check trends)?
  - Is their ordering behavior erratic (high order_variance) contributing to the bullwhip?
  - Use cost share, inventory level, backlog level, bottleneck type, and ordering behavior to make your decision instead of cost share alone.

A good response assigns DIFFERENT alpha values to different agents 
based on their individual cost share and role.
Example of targeted reasoning:
  'Distributor has 32% of total cost despite being a middle echelon.
   This suggests it is over-stocking. Reduce its alpha to 0.1 to make
   it more cooperative and share inventory burden downstream.
   Retailer has only 17% of cost — maintain its alpha to preserve 
   service level.'

COST STRUCTURE:
- Holding cost: h=1 per unit of inventory per period (uniform across all agents)
- Backlog cost: b=1 per unit of unmet demand per period (uniform across all agents)
- Lead time: L=4 periods at each echelon

YOUR TASK:
At the end of each 5-episode evaluation block, you receive:
- Input A: Statistical summary of Merton Jump Diffusion demand this block
- Input B: Averaged KPI performance metrics for this block
- Input C: Your previous 3 blocks of decisions and their actual outcomes

Analyze the multi-signal tensions, identify conflicts, and set alpha for each agent.

HARD CONSTRAINTS (enforced programmatically — output will be clipped if violated):
- All alpha values must be in [0.05, 0.95]
- Maximum alpha change per block: 0.25 from previous value
- After smoothing (0.8 * prev + 0.2 * your value), changes are gradual

OUTPUT FORMAT — respond ONLY with valid JSON, no other text:
{
  "reasoning": "<2-4 sentences: identify the key tension, explain your tradeoff>",
  "self_correction": "<1 sentence: mistake from previous block, or 'No correction needed'>",
  "compiled_rule": "<1 sentence: domain rule you are adding to your rulebook this block>",
  "confidence": <float between 0.0 and 1.0 representing your confidence in this decision>,
  "alphas": {
      "stage_1_retailer": <float>,
      "stage_2_warehouse": <float>,
      "stage_3_distributor": <float>,
      "stage_4_manufacturer": <float>
  }
}"""

def build_input_a(demand_trace: list) -> dict:
    demands = np.array(demand_trace)
    if len(demands) == 0:
        return {
            "demand_mean": 0.0,
            "demand_variance": 0.0,
            "demand_skewness": 0.0,
            "demand_kurtosis": 0.0,
            "n_demand_spikes": 0,
            "max_spike_magnitude": 1.0,
            "spike_periods": []
        }
        
    mean = np.mean(demands)
    std = np.std(demands)
    
    threshold = mean + 2 * std
    spikes = [t for t, d in enumerate(demands) if d > threshold]
    
    return {
        "demand_mean": round(float(mean), 2),
        "demand_variance": round(float(np.var(demands)), 2),
        "demand_skewness": round(float(scipy.stats.skew(demands)), 3) if len(demands) > 2 else 0.0,
        "demand_kurtosis": round(float(scipy.stats.kurtosis(demands)), 3) if len(demands) > 3 else 0.0,
        "n_demand_spikes": len(spikes),
        "max_spike_magnitude": round(float(np.max(demands) / mean), 2) if mean > 0 else 1.0,
        "spike_periods": spikes[:5]
    }

def build_input_b(block_metrics_history: list, current_block_metrics: dict) -> dict:
    def get_trend(history, key, threshold=0.05, lower_is_better=True):
        if len(history) < 2:
            return "insufficient_data"
            
        recent = [h.get(key, 0.0) for h in history[-3:]]
        if len(recent) < 2:
            return "insufficient_data"
            
        first = recent[0]
        last = recent[-1]
        
        delta = (last - first) / (abs(first) + 1e-8)
        
        if lower_is_better:
            if delta < -threshold: return "improving"
            if delta > threshold: return "degrading"
        else:
            if delta > threshold: return "improving"
            if delta < -threshold: return "degrading"
            
        return "stable"

    def get_direction(history, agent, key, threshold=0.05):
        if len(history) < 2:
            return "N/A"
        recent = []
        for h in history[-3:]:
            if "agent_behavior_raw" in h and agent in h["agent_behavior_raw"]:
                recent.append(h["agent_behavior_raw"][agent].get(key, 0.0))
        if len(recent) < 2:
            return "N/A"
        delta = (recent[-1] - recent[0]) / (abs(recent[0]) + 1e-8)
        if delta > threshold: return "increasing"
        if delta < -threshold: return "decreasing"
        return "stable"
    
    m = current_block_metrics.copy()
    
    per_agent_costs = m.get("per_agent_costs", {})
    if per_agent_costs:
        total = sum(per_agent_costs.values())
        if total > 0:
            m["cost_shares"] = {
                agent: f"{cost/total:.1%}" 
                for agent, cost in per_agent_costs.items()
            }
        else:
            m["cost_shares"] = {agent: "0.0%" for agent in per_agent_costs.keys()}
    else:
        m["cost_shares"] = {}
    
    m["total_cost_trend"] = get_trend(block_metrics_history, "total_cost", lower_is_better=True)
    m["bullwhip_ratio_trend"] = get_trend(block_metrics_history, "bullwhip_ratio", lower_is_better=True)
    m["service_level_trend"] = get_trend(block_metrics_history, "service_level", lower_is_better=False)
    m["recovery_time_trend"] = get_trend(block_metrics_history, "recovery_time", lower_is_better=True)
    m["reward_variance_trend"] = get_trend(block_metrics_history, "reward_variance", lower_is_better=True)
    
    bw = m.get("bullwhip_ratio", 1.0)
    sl = m.get("service_level", 1.0)
    
    m["conflicting_signals"] = (bw > 2.0 and sl < 0.85)

    m["agent_state"] = {}
    m["agent_behavior"] = {}
    raw = m.pop("agent_behavior_raw", {})
    for agent, stats in raw.items():
        ratio = stats["inventory_backlog_ratio"]
        if ratio < 0.2:
            bottleneck = "stock_starved"
            recommended_alpha_bias = "consider_higher_alpha"
        elif ratio > 5.0:
            bottleneck = "inventory_heavy"
            recommended_alpha_bias = "consider_lower_alpha"
        else:
            bottleneck = "balanced"
            recommended_alpha_bias = "maintain_or_small_adjustment"

        m["agent_state"][agent] = {
            "avg_inventory": round(stats["avg_inventory"], 1),
            "avg_backlog": round(stats["avg_backlog"], 1),
            "inventory_backlog_ratio": round(ratio, 2),
            "bottleneck_type": bottleneck,
            "recommended_alpha_bias": recommended_alpha_bias,
            "inventory_trend": get_direction(block_metrics_history, agent, "avg_inventory"),
            "backlog_trend": get_direction(block_metrics_history, agent, "avg_backlog")
        }
        m["agent_behavior"][agent] = {
            "avg_order_qty": round(stats["avg_order_qty"], 1),
            "order_variance": round(stats["order_variance"], 1)
        }
    
    return m

def build_input_c(llm_history: list) -> list:
    recent = llm_history[-3:]
    records = []
    for r in recent:
        # Simple heuristic for prediction_correct
        reasoning = r.get("reasoning", "").lower()
        actual_cost = r.get("actual_total_cost", 0.0)
        prev_cost = r.get("prev_cost", float('inf'))
        
        # Determine if outcome matched expectation
        if "cost will decrease" in reasoning or "decrease cost" in reasoning:
            prediction_correct = (actual_cost < prev_cost)
        elif "service level will increase" in reasoning or "improve service" in reasoning:
            prediction_correct = (r.get("actual_service_level", 0.0) > r.get("prev_sl", 1.0))
        else:
            prediction_correct = r.get("prediction_correct", False)
            
        records.append({
            "block_id": r.get("block_id", -1),
            "reasoning": r.get("reasoning", ""),
            "compiled_rule": r.get("compiled_rule", "none"),
            "alphas": r.get("alphas", {}),
            "actual_total_cost": round(actual_cost, 1),
            "actual_bullwhip": round(r.get("actual_bullwhip", 1.0), 3),
            "actual_service_level": round(r.get("actual_service_level", 1.0), 4),
            "prediction_correct": prediction_correct
        })
    return records

def build_full_prompt(input_a: dict, input_b: dict, input_c: list, system_prompt: str, rulebook=None) -> str:
    prompt = "--- DEMAND ENVIRONMENT (Input A) ---\n"
    prompt += f"Mean: {input_a.get('demand_mean')} | Variance: {input_a.get('demand_variance')}\n"
    prompt += f"Skewness: {input_a.get('demand_skewness')} (positive = right-skewed spike events)\n"
    prompt += f"Kurtosis: {input_a.get('demand_kurtosis')} (>3 = fat tails, sudden jumps typical of Merton JD)\n"
    prompt += f"Spike events this block: {input_a.get('n_demand_spikes')} periods exceeded 2 std above mean\n"
    prompt += f"Maximum spike: {input_a.get('max_spike_magnitude')}x normal demand\n\n"
    
    prompt += "--- SYSTEM PERFORMANCE (Input B) ---\n"
    prompt += f"Total cost: {input_b.get('total_cost')} ({input_b.get('total_cost_trend')})\n"
    
    per_agent = input_b.get('per_agent_costs', {})
    prompt += f"Per-agent costs: Retailer={per_agent.get('retailer')} | Warehouse={per_agent.get('warehouse')} | Distributor={per_agent.get('distributor')} | Manufacturer={per_agent.get('manufacturer')}\n"
    shares = input_b.get('cost_shares', {})
    if shares:
        prompt += f"Cost share: R={shares.get('retailer')} W={shares.get('warehouse')} D={shares.get('distributor')} M={shares.get('manufacturer')}\n"
    prompt += f"Bullwhip ratio: {input_b.get('bullwhip_ratio')} ({input_b.get('bullwhip_ratio_trend')}) [target: approx 1.0]\n"
    prompt += f"Service level: {input_b.get('service_level')} ({input_b.get('service_level_trend')}) [target: >=0.80]\n"
    prompt += f"Recovery time: {input_b.get('recovery_time', 'N/A')} periods\n"
    prompt += f"Inter-agent reward variance: {input_b.get('reward_variance')}\n"
    prompt += f"WARNING Conflicting signals: {input_b.get('conflicting_signals')}\n\n"

    prompt += "--- AGENT PHYSICAL STATES ---\n"
    for agent, state in input_b.get("agent_state", {}).items():
        prompt += f"{agent.capitalize()} ({state['bottleneck_type']}):\n"
        prompt += f"  Inventory: {state['avg_inventory']} ({state['inventory_trend']}) | Backlog: {state['avg_backlog']} ({state['backlog_trend']})\n"
        prompt += f"  Inv/Backlog Ratio: {state['inventory_backlog_ratio']} -> Bias: {state['recommended_alpha_bias']}\n"

    prompt += "\n--- AGENT ORDERING BEHAVIOR ---\n"
    for agent, beh in input_b.get("agent_behavior", {}).items():
        prompt += f"{agent.capitalize()}: Avg Order Qty: {beh['avg_order_qty']} | Order Variance: {beh['order_variance']}\n"
    
    prompt += "\n--- YOUR HISTORICAL TRACK RECORD (Input C) ---\n"
    if not input_c:
        prompt += "No history available yet. This is the first block.\n"
    else:
        for record in input_c:
            prompt += f"Block {record['block_id']}:\n"
            prompt += f"  Your reasoning: {record['reasoning']}\n"
            prompt += f"  Compiled rule: {record['compiled_rule']}\n"
            prompt += f"  Your alpha decisions: {record['alphas']}\n"
            prompt += f"  Actual total cost: {record['actual_total_cost']}\n"
            prompt += f"  Actual bullwhip: {record['actual_bullwhip']}\n"
            prompt += f"  Actual service level: {record['actual_service_level']}\n"
            prompt += f"  Prediction correct: {record['prediction_correct']}\n\n"
            
    if rulebook:
        prompt += "--- CURRENT COMPILED RULEBOOK ---\n"
        for i, rule in enumerate(rulebook):
            prompt += f"RULE {i+1}: {rule}\n"
        prompt += "\nYou must apply the above rules to determine alpha values for this block.\n"
        
    return prompt
