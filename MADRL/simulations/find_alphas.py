import sys; import os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np; import torch
from envs.serial import Env as SerialEnv
from agents.happo.policy import HAPPOPolicy

def run_sim(alphas):
    env = SerialEnv()
    policy = HAPPOPolicy(env.observation_space[0], env.action_space[0], num_agents=env.num_agents)
    checkpoint = torch.load('MADRL/training/weights/happo_best.pt', map_location='cpu')
    policy.actor.load_state_dict(checkpoint['actor'])
    policy.actor.eval()
    
    # Force exact same demand stream
    np.random.seed(42)
    obs = env.reset()
    
    total_cost = 0
    min_sl = 1.0
    final_sl = 0
    
    alpha_array = np.array([alphas['retailer'], alphas['warehouse'], alphas['distributor'], alphas['manufacturer']])
    
    for t in range(200):
        actions = []
        for i in range(env.num_agents):
            state_tensor = torch.FloatTensor(obs[i]).unsqueeze(0)
            with torch.no_grad():
                a, _ = policy.get_action(state_tensor, deterministic=True)
                actions.append(a.numpy()[0])
        actions = np.array(actions)
        
        # Inject Disruption
        if 80 <= t <= 99:
            actions[3] = 0.0
            
        next_obs, rewards, done, info = env.step(actions, current_alphas=alpha_array)
        obs = next_obs
        total_cost += info['total_cost']
        
        if t > 50:
            sl = info.get('service_level_rolling', info['service_level'])
            if sl < min_sl: min_sl = sl
            final_sl = sl
            
    return total_cost, min_sl, final_sl

cands = [
    {'retailer': 0.5, 'warehouse': 0.5, 'distributor': 0.5, 'manufacturer': 0.5},
    {'retailer': 0.8, 'warehouse': 0.2, 'distributor': 0.2, 'manufacturer': 0.8},
    {'retailer': 1.0, 'warehouse': 0.0, 'distributor': 0.0, 'manufacturer': 1.0},
    {'retailer': 0.2, 'warehouse': 0.8, 'distributor': 0.8, 'manufacturer': 0.2},
    {'retailer': 0.8, 'warehouse': 0.5, 'distributor': 0.5, 'manufacturer': 0.8},
    {'retailer': 1.0, 'warehouse': 1.0, 'distributor': 1.0, 'manufacturer': 1.0},
    {'retailer': 0.0, 'warehouse': 0.0, 'distributor': 0.0, 'manufacturer': 0.0},
    {'retailer': 1.0, 'warehouse': 0.25, 'distributor': 0.25, 'manufacturer': 0.5}
]

for a in cands:
    c, ms, fs = run_sim(a)
    print(f"R:{a['retailer']} W:{a['warehouse']} D:{a['distributor']} M:{a['manufacturer']} -> Cost:{c:.0f} MinSL:{ms:.2f} FinSL:{fs:.2f}")
