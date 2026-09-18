import numpy as np
from typing import List, Dict, Any

def compute_trace_metrics(
    orders: List[List[float]],          # shape: [M, T] where M=4 echelons, T=200 steps
    inventories: List[List[float]],     # shape: [M, T]
    backlogs: List[List[float]],        # shape: [M, T]
    demands: List[float],               # shape: [T]
    holding_costs: List[float] = None,  # default [1, 1, 1, 1]
    backlog_costs: List[float] = None,  # default [1, 1, 1, 1]
    fulfilled_demands: List[float] = None # optional, for exact SL
) -> Dict[str, Any]:
    """
    Computes all standard SCM KPIs and Base Paper benchmark metrics for a single 200-timestep trace.
    
    Echelon indexing:
      0: Retailer (Echelon 1, downstream)
      1: Warehouse (Echelon 2)
      2: Distributor (Echelon 3)
      3: Manufacturer (Echelon 4, upstream)
    """
    M = len(orders)
    T = len(demands)
    
    if holding_costs is None:
        holding_costs = [1.0] * M
    if backlog_costs is None:
        backlog_costs = [1.0] * M
        
    orders_arr = np.array(orders, dtype=np.float64)        # [M, T]
    inv_arr = np.array(inventories, dtype=np.float64)       # [M, T]
    backlog_arr = np.array(backlogs, dtype=np.float64)      # [M, T]
    demands_arr = np.array(demands, dtype=np.float64)       # [T]
    
    # 1. Costs
    echelon_total_costs = []
    echelon_holding_costs = []
    echelon_backlog_costs = []
    for i in range(M):
        h_cost = np.sum(inv_arr[i] * holding_costs[i])
        b_cost = np.sum(backlog_arr[i] * backlog_costs[i])
        echelon_holding_costs.append(float(h_cost))
        echelon_backlog_costs.append(float(b_cost))
        echelon_total_costs.append(float(h_cost + b_cost))
        
    total_system_cost = float(np.sum(echelon_total_costs))
    per_period_cost = float(total_system_cost / T)
    echelon_per_period_costs = [float(c / T) for c in echelon_total_costs]
    
    # 2. Demand fluctuation metric F_i = std(O_i) / mean(O_i) (Coefficient of Variation)
    echelon_fluctuations = []
    for i in range(M):
        mean_o = np.mean(orders_arr[i])
        std_o = np.std(orders_arr[i])
        if mean_o > 1e-6:
            f_i = float(std_o / mean_o)
        else:
            f_i = 0.0
        echelon_fluctuations.append(f_i)
        
    # Average fluctuation F = (F_1 + F_2 + ... + F_M) / M
    avg_fluctuation = float(np.mean(echelon_fluctuations))
    
    # Customer demand CV
    mean_d = np.mean(demands_arr)
    std_d = np.std(demands_arr)
    cv_demand = float(std_d / max(mean_d, 1e-6))
    
    # Bullwhip Amplification = (F_M / F_1) - 1
    # Note: F_1 is Retailer (downstream), F_M is Manufacturer (upstream)
    f_1 = echelon_fluctuations[0]
    f_M = echelon_fluctuations[-1]
    
    if f_1 > 1e-6:
        bw_amplification_ratio = float((f_M / f_1) - 1.0)
        bw_amplification_pct = float(bw_amplification_ratio * 100.0)
    else:
        bw_amplification_ratio = 0.0
        bw_amplification_pct = 0.0
        
    # Standard Bullwhip ratio = F_M / CV_demand
    if cv_demand > 1e-6:
        bw_ratio = float(f_M / cv_demand)
    else:
        bw_ratio = 1.0
        
    # 3. Service Level at Retailer
    if fulfilled_demands is not None:
        total_fulfilled = float(np.sum(fulfilled_demands))
        total_demanded = float(np.sum(demands_arr))
        service_level = float(total_fulfilled / max(total_demanded, 1.0))
    else:
        # Approximate from retailer backlog if fulfilled series not provided
        # fulfilled_t = max(0, demand_t - backlog_t)
        total_demanded = float(np.sum(demands_arr))
        total_backlog_r = float(np.sum(backlog_arr[0]))
        service_level = float(max(0.0, 1.0 - (total_backlog_r / max(total_demanded, 1.0))))
        
    return {
        "total_cost": total_system_cost,
        "per_period_cost": per_period_cost,
        "echelon_per_period_costs": {
            "retailer": echelon_per_period_costs[0],
            "warehouse": echelon_per_period_costs[1],
            "distributor": echelon_per_period_costs[2],
            "manufacturer": echelon_per_period_costs[3]
        },
        "echelon_fluctuations": {
            "F1_retailer": echelon_fluctuations[0],
            "F2_warehouse": echelon_fluctuations[1],
            "F3_distributor": echelon_fluctuations[2],
            "F4_manufacturer": echelon_fluctuations[3]
        },
        "avg_fluctuation": avg_fluctuation,
        "bullwhip_amplification_pct": bw_amplification_pct,
        "bullwhip_amplification_ratio": bw_amplification_ratio,
        "cv_demand": cv_demand,
        "bullwhip_ratio": bw_ratio,
        "service_level": service_level
    }

def aggregate_trace_metrics(trace_metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes mean and std of all metrics across 20 evaluation demand traces.
    """
    N = len(trace_metrics_list)
    if N == 0:
        return {}
        
    per_period_costs = [m["per_period_cost"] for m in trace_metrics_list]
    total_costs = [m["total_cost"] for m in trace_metrics_list]
    avg_fluctuations = [m["avg_fluctuation"] for m in trace_metrics_list]
    amplifications_pct = [m["bullwhip_amplification_pct"] for m in trace_metrics_list]
    amplifications_ratio = [m["bullwhip_amplification_ratio"] for m in trace_metrics_list]
    service_levels = [m["service_level"] for m in trace_metrics_list]
    bw_ratios = [m["bullwhip_ratio"] for m in trace_metrics_list]
    cv_demands = [m["cv_demand"] for m in trace_metrics_list]
    
    # Echelon fluctuations
    f1s = [m["echelon_fluctuations"]["F1_retailer"] for m in trace_metrics_list]
    f2s = [m["echelon_fluctuations"]["F2_warehouse"] for m in trace_metrics_list]
    f3s = [m["echelon_fluctuations"]["F3_distributor"] for m in trace_metrics_list]
    f4s = [m["echelon_fluctuations"]["F4_manufacturer"] for m in trace_metrics_list]
    
    # Echelon costs
    c_r = [m["echelon_per_period_costs"]["retailer"] for m in trace_metrics_list]
    c_w = [m["echelon_per_period_costs"]["warehouse"] for m in trace_metrics_list]
    c_d = [m["echelon_per_period_costs"]["distributor"] for m in trace_metrics_list]
    c_m = [m["echelon_per_period_costs"]["manufacturer"] for m in trace_metrics_list]
    
    return {
        "n_traces": N,
        "per_period_cost": {
            "mean": float(np.mean(per_period_costs)),
            "std": float(np.std(per_period_costs))
        },
        "total_cost": {
            "mean": float(np.mean(total_costs)),
            "std": float(np.std(total_costs))
        },
        "avg_fluctuation": {
            "mean": float(np.mean(avg_fluctuations)),
            "std": float(np.std(avg_fluctuations))
        },
        "bullwhip_amplification_pct": {
            "mean": float(np.mean(amplifications_pct)),
            "std": float(np.std(amplifications_pct))
        },
        "bullwhip_amplification_ratio": {
            "mean": float(np.mean(amplifications_ratio)),
            "std": float(np.std(amplifications_ratio))
        },
        "service_level": {
            "mean": float(np.mean(service_levels)),
            "std": float(np.std(service_levels))
        },
        "bullwhip_ratio": {
            "mean": float(np.mean(bw_ratios)),
            "std": float(np.std(bw_ratios))
        },
        "cv_demand": {
            "mean": float(np.mean(cv_demands)),
            "std": float(np.std(cv_demands))
        },
        "echelon_fluctuations_mean": {
            "F1_retailer": float(np.mean(f1s)),
            "F2_warehouse": float(np.mean(f2s)),
            "F3_distributor": float(np.mean(f3s)),
            "F4_manufacturer": float(np.mean(f4s))
        },
        "echelon_per_period_costs_mean": {
            "retailer": float(np.mean(c_r)),
            "warehouse": float(np.mean(c_w)),
            "distributor": float(np.mean(c_d)),
            "manufacturer": float(np.mean(c_m))
        }
    }
