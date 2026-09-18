def build_discrete_regime_prompt(
    trace_idx,
    recent_kpis=None,
    recent_demand_summary=None,
    memory_context=None
):
    """
    Builds a prompt instructing the LLM to select from 9 discrete operational regimes:
    alpha in {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9} for each echelon.
    """
    regimes_menu = """
### DISCRETE OPERATING REGIMES MENU:
Select exactly one discrete alpha value from {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9} for each echelon:
- 0.1 (Regime 1 - Extreme Bullwhip Suppression): Maximum order smoothing, highly cooperative, heavily penalizes upstream order spikes.
- 0.2 (Regime 2 - Heavy Upstream Damping): Strong smoothing, prevents cascading demand shock waves.
- 0.3 (Regime 3 - Moderate Stability Mode): Prioritizes network coordination while allowing responsive adjustments.
- 0.4 (Regime 4 - Balanced Conservative): Dampens bullwhip while maintaining inventory safety stock.
- 0.5 (Regime 5 - Neutral Equal Weight): 50% local cost minimization, 50% global supply chain penalty.
- 0.6 (Regime 6 - Cost-Conscious Baseline): Closer to base paper optimal trade-off point.
- 0.7 (Regime 7 - Lean High Efficiency): Aggressive reduction of inventory holding and backlog costs.
- 0.8 (Regime 8 - High Agility Lean): Rapid fulfillment, minimum buffer inventory.
- 0.9 (Regime 9 - Extreme Cost Minimization): 90% local cost minimization, strictly minimizes local holding/backlog.
"""

    context_str = ""
    if memory_context:
        context_str += f"""
### ACCUMULATED STRATEGIC MEMORY:
{memory_context}
"""

    if recent_kpis:
        context_str += f"""
### PERFORMANCE ON RECENT EPISODE:
- Total Cost: {recent_kpis.get('total_cost', 0):.2f}
- Service Level: {recent_kpis.get('service_level', 0):.3f}
- Bullwhip Ratio: {recent_kpis.get('bullwhip_ratio', 0):.2f}
- Per-Agent Costs: Retailer={recent_kpis.get('per_agent_costs', {}).get('retailer', 0):.1f}, Warehouse={recent_kpis.get('per_agent_costs', {}).get('warehouse', 0):.1f}, Distributor={recent_kpis.get('per_agent_costs', {}).get('distributor', 0):.1f}, Manufacturer={recent_kpis.get('per_agent_costs', {}).get('manufacturer', 0):.1f}
- Active Alphas: {recent_kpis.get('active_alphas', [0.5]*4)}
"""

    if recent_demand_summary:
        context_str += f"""
### DEMAND CHARACTERISTICS FOR UPCOMING TRACE {trace_idx}:
- Mean Customer Demand: {recent_demand_summary.get('mean', 10.0):.2f}
- Demand Std Dev: {recent_demand_summary.get('std', 3.0):.2f}
- Spike / Jump Count: {recent_demand_summary.get('spike_count', 0)}
- Volatility Regime: {recent_demand_summary.get('regime', 'Normal')}
"""

    prompt = f"""You are the Chief Strategic Supply Chain Meta-Controller managing a 4-echelon serial supply chain (Retailer -> Warehouse -> Distributor -> Manufacturer).

A high-capacity MARL execution policy (256-unit GRU) has been trained across 9 discrete operating postures.
Your role is to select the optimal discrete regime ID for each echelon for the upcoming episode.

{regimes_menu}
{context_str}

### STRATEGIC GUIDELINES:
1. **Downstream vs Upstream Coordination:** Downstream agents (Retailer/Warehouse) often benefit from leaner, more responsive regimes (0.6 - 0.8) to meet customer demand, whereas upstream agents (Distributor/Manufacturer) require more stabilizing regimes (0.4 - 0.6) to avoid catastrophic bullwhip magnification.
2. **Volatile / High Spike Traces:** When customer demand has high jump variance, lower alphas (0.3 - 0.5) prevent upstream factory over-ordering.
3. **Calm / Predictable Traces:** Higher alphas (0.7 - 0.8) strip out wasteful safety stock and minimize total holding cost.

### REQUIRED OUTPUT FORMAT:
Respond with ONLY a valid JSON object matching this schema:
```json
{{
  "retailer_alpha": 0.7,
  "warehouse_alpha": 0.6,
  "distributor_alpha": 0.5,
  "manufacturer_alpha": 0.4,
  "reasoning": "<1-2 sentences explaining why this regime configuration was selected>"
}}
```
"""
    return prompt
