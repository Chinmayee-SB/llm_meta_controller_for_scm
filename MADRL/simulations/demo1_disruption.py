import os
import sys
import json
import numpy as np
import torch
import yaml
from collections import deque

# Add MADRL to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from MADRL.envs.serial import Env as SerialEnv
from MADRL.training.run_phase2 import load_frozen_policy

AGENT_NAMES = ["retailer", "warehouse", "distributor", "manufacturer"]

def run_demo1():
    current_alphas = {
        "retailer": 0.80,
        "warehouse": 0.25,
        "distributor": 0.25,
        "manufacturer": 0.50
    }
    with open("MADRL/config/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    class DummyArgs:
        pass
    env_args = DummyArgs()
    env_args.num_agents = 4
    env_args.obs_dim = 8
    env = SerialEnv(env_args)
    
    # Fix the missing _kpi_step_rewards initialization bug in serial.py if any
    if not hasattr(env, '_kpi_step_rewards'):
        env._kpi_step_rewards = []
    
    # Redirect stdout to suppress print from load_frozen_policy
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    frozen_actors = load_frozen_policy("MADRL/results/phase1_weights/frozen_policy.pt", config)
    sys.stdout.close()
    sys.stdout = old_stdout
    
    alpha_list = [current_alphas[k] for k in AGENT_NAMES]
    env.set_alphas(alpha_list)
    obs = env.reset(train=False, normalize=True)
    
    hidden_size = getattr(frozen_actors[0], 'hidden_size', 64)
    recurrent_N = getattr(frozen_actors[0], '_recurrent_N', 1)
    rnn_states = [torch.zeros((1, recurrent_N, hidden_size), dtype=torch.float32) for _ in range(4)]
    masks = torch.ones((1, 1), dtype=torch.float32)
    
    timesteps_data = []
    
    rolling_demands = deque(maxlen=10)
    rolling_fulfilled = deque(maxlen=10)
    
    prev_kpi_demanded = 0.0
    prev_kpi_fulfilled = 0.0
    
    for t in range(200):
        actions = []
        with torch.no_grad():
            for i in range(4):
                obs_tensor = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0)
                if i < len(frozen_actors):
                    action, _, next_rnn_state = frozen_actors[i](
                        obs_tensor, 
                        rnn_states[i], 
                        masks, 
                        deterministic=True
                    )
                    rnn_states[i] = next_rnn_state
                    act_int = int(action.item())
                    one_hot = np.eye(21)[act_int]
                    actions.append(one_hot)
                else:
                    one_hot = np.zeros(21)
                    one_hot[10] = 1.0
                    actions.append(one_hot)
            
            disruption_active = (80 <= t <= 99)
            if disruption_active:
                actions[3] = np.eye(21)[0]
        
        # Step the environment
        obs, rewards, done, info = env.step(actions)
                
        # Calculate Rolling SL
        step_demanded = max(env._kpi_demanded - prev_kpi_demanded, 0)
        step_fulfilled = max(env._kpi_fulfilled - prev_kpi_fulfilled, 0)
        
        if step_demanded == 0 and t > 0:
            step_demanded = 1.0
            step_fulfilled = 1.0
            
        rolling_demands.append(step_demanded)
        rolling_fulfilled.append(step_fulfilled)
        
        sl_rolling = sum(rolling_fulfilled) / max(sum(rolling_demands), 1.0)
        
        prev_kpi_demanded = env._kpi_demanded
        prev_kpi_fulfilled = env._kpi_fulfilled
        
        order_qty = [int(np.argmax(a)) for a in actions]
        shipment = [float(env.order[i][-1]) for i in range(4)]
        cost_step = [-float(r[0]) for r in rewards]
        
        demand_t = env.demand_list[t] if t < len(env.demand_list) else env.demand_list[-1]
        
        step_data = {
            "t": int(t),
            "disruption_active": bool(disruption_active),
            "inventory": {k: float(v) for k, v in zip(AGENT_NAMES, env.inventory)},
            "backlog": {k: float(v) for k, v in zip(AGENT_NAMES, env.backlog)},
            "order_qty": {k: int(v) for k, v in zip(AGENT_NAMES, order_qty)},
            "shipment": {k: float(v) for k, v in zip(AGENT_NAMES, shipment)},
            "cost_step": {k: float(v) for k, v in zip(AGENT_NAMES, cost_step)},
            "service_level_rolling": float(sl_rolling),
            "cumulative_cost": float(env._kpi_total_cost),
            "demand": float(demand_t),
            
            "manufacturer_inventory": float(env.inventory[3]),
            "manufacturer_backlog": float(env.backlog[3]),
            "retailer_backlog": float(env.backlog[0]),
            # recovery_threshold_used will be populated later
        }
        timesteps_data.append(step_data)
        
        if all(done) if isinstance(done, list) else done:
            break
            
    # Post-episode metrics
    pre_disruption_sls = [d["service_level_rolling"] for d in timesteps_data[60:80]]
    pre_disruption_SL = np.mean(pre_disruption_sls) if pre_disruption_sls else 1.0
    
    pre_disruption_costs = [sum(d["cost_step"].values()) for d in timesteps_data[60:80]]
    pre_disruption_cost = np.mean(pre_disruption_costs) if pre_disruption_costs else 0.0
    
    disruption_costs = [sum(d["cost_step"].values()) for d in timesteps_data[80:100]]
    disruption_cost_total = sum(disruption_costs)
    
    target_sl = 0.9 * pre_disruption_SL
    
    for d in timesteps_data:
        d["recovery_threshold_used"] = float(target_sl)
        
    disruption_onward = timesteps_data[80:]
    min_sl_obj = min(disruption_onward, key=lambda x: x["service_level_rolling"])
    minimum_service_level = min_sl_obj["service_level_rolling"]
    timestep_of_minimum_service_level = min_sl_obj["t"]
    area_under_sl_curve = sum(d["service_level_rolling"] for d in disruption_onward)
    
    recovery_time = 100
    recovery_cost_total = 0.0
    recovery_service_level = target_sl
    
    consecutive_recovered = 0
    first_recovered_t = -1
    
    for t in range(100, len(timesteps_data)):
        d = timesteps_data[t]
        recovery_cost_total += sum(d["cost_step"].values())
        if d["service_level_rolling"] >= target_sl:
            if consecutive_recovered == 0:
                first_recovered_t = t
            consecutive_recovered += 1
            if consecutive_recovered >= 5:
                recovery_time = first_recovered_t - 100
                recovery_service_level = timesteps_data[first_recovered_t]["service_level_rolling"]
                break
        else:
            consecutive_recovered = 0
            
    slla_end_t = first_recovered_t if (recovery_time != 100 and first_recovered_t != -1) else len(timesteps_data) - 1
    service_level_loss_area = sum(max(0.0, pre_disruption_SL - timesteps_data[t]["service_level_rolling"]) for t in range(80, slla_end_t + 1))
            
    peak_backlog = {agent: float(max(d["backlog"][agent] for d in timesteps_data)) for agent in AGENT_NAMES}
    min_inventory = {agent: float(min(d["inventory"][agent] for d in timesteps_data)) for agent in AGENT_NAMES}
    
    summary = {
        "disruption_start": 80,
        "disruption_end": 99,
        "pre_disruption_SL": float(pre_disruption_SL),
        "pre_disruption_cost": float(pre_disruption_cost),
        "recovery_time": int(recovery_time),
        "disruption_cost_total": float(disruption_cost_total),
        "recovery_cost_total": float(recovery_cost_total),
        "peak_backlog_per_agent": peak_backlog,
        
        "recovery_threshold": float(target_sl),
        "recovery_service_level": float(recovery_service_level),
        "minimum_service_level": float(minimum_service_level),
        "timestep_of_minimum_service_level": int(timestep_of_minimum_service_level),
        "area_under_sl_curve_post_80": float(area_under_sl_curve),
        "manufacturer_peak_backlog": float(peak_backlog["manufacturer"]),
        "manufacturer_min_inventory": float(min_inventory["manufacturer"]),
        "retailer_peak_backlog": float(peak_backlog["retailer"]),
        "service_level_loss_area": float(service_level_loss_area),
        
        "alpha_used": current_alphas
    }
    
    output = {
        "metadata": {
            "alpha_mode": "neutral",
            "alpha_values": summary["alpha_used"],
            "disruption_start": 80,
            "disruption_end": 99,
            "disruption_type": "manufacturer_production_shutdown"
        },
        "timesteps": timesteps_data,
        "recovery_metrics": {
            "recovery_time": int(recovery_time),
            "minimum_service_level": float(minimum_service_level),
            "timestep_of_minimum_service_level": int(timestep_of_minimum_service_level),
            "area_under_sl_curve_post_80": float(area_under_sl_curve)
        },
        "summary": summary
    }
    
    os.makedirs("MADRL/simulations/output", exist_ok=True)
    out_path = f"MADRL/simulations/output/demo1_disruption.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
        
    return summary

if __name__ == "__main__":
    print("Running Demo 1 (Disruption Resilience)...")
    summary = run_demo1()
    
    print("\n  DISRUPTION RESILIENCE SUMMARY")
    print("  ----------------------------------------")
    print(f"  {'Pre-disruption Service Level':<35} {summary['pre_disruption_SL']:.3f}")
    print(f"  {'Minimum Service Level':<35} {summary['minimum_service_level']:.3f}")
    print(f"  {'Recovery Threshold':<35} {summary['recovery_threshold']:.3f}")
    print(f"  {'Recovery Time':<35} {summary['recovery_time']}")
    # print(f"  {'Service Level Loss Area':<35} {summary['service_level_loss_area']:.3f}")
    print(f"  {'Manufacturer Peak Backlog':<35} {summary['manufacturer_peak_backlog']:.0f}")
    print(f"  {'Manufacturer Minimum Inventory':<35} {summary['manufacturer_min_inventory']:.0f}")
    print(f"  {'Retailer Peak Backlog':<35} {summary['retailer_peak_backlog']:.0f}")
    print(f"  {'Disruption Period Cost':<35} {summary['disruption_cost_total']:.0f}")
    print("  ----------------------------------------\n")
