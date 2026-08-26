import os
import sys
import torch
import torch.nn as nn
import numpy as np
import random
from sumo_rl import SumoEnvironment

# KEEP THIS: Crucial for Linux virtual environment stability
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.insert(0, tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# Model Def (Must exactly match Zhang training architecture)
class DQNetwork(nn.Module):
    def __init__(self, stateDim, actionDim):
        super(DQNetwork, self).__init__()
        # Extract the raw integer out of the tuple safely to prevent nn.Linear failures
        if isinstance(stateDim, tuple):
            input_features = stateDim[0]
        else:
            input_features = stateDim
            
        self.fullyConnected = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),           
            nn.Linear(64, actionDim)
        )
    def forward(self, x):
        return self.fullyConnected(x)

class ZhangPartialObservabilityWrapper:
    """Emulates partial vehicle detection and implements Zhang et al. reward function during evaluation."""
    def __init__(self, env, detection_rate=0.6, v_max=13.89):
        self.env = env
        self.detection_rate = detection_rate
        self.v_max = v_max
        
    def _apply_partial_observability(self, raw_obs):
        masked_obs = np.array(raw_obs, dtype=np.float32)
        return masked_obs * self.detection_rate

    def _calculate_zhang_reward(self):
        ts_id = self.env.ts_ids[0] if isinstance(self.env.ts_ids, list) else self.env.ts_ids
        lanes = self.env.traffic_signals[ts_id].lanes
        vehicle_speeds = []
        
        for lane in lanes:
            veh_list = self.env.sumo.lane.getLastStepVehicleIDs(lane)
            for veh in veh_list:
                if random.random() <= self.detection_rate:
                    vehicle_speeds.append(self.env.sumo.vehicle.getSpeed(veh))
                    
        if len(vehicle_speeds) == 0:
            v_S = self.v_max
        else:
            v_S = np.mean(vehicle_speeds)
            
        v_S = min(v_S, self.v_max)
        reward = (-1.0 / self.v_max) * (self.v_max - v_S)
        return float(reward)

    def reset(self):
        raw_obs, info = self.env.reset()
        return self._apply_partial_observability(raw_obs), info

    def step(self, action):
        next_raw_obs, _, terminated, truncated, info = self.env.step(action)
        masked_next_obs = self._apply_partial_observability(next_raw_obs)
        custom_reward = self._calculate_zhang_reward()
        return masked_next_obs, custom_reward, terminated, truncated, info


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, ".."))    
    net_path = os.path.join(base_dir, "dataset", "nets", "single-intersection", "single-intersection.net.xml")
    route_path = os.path.join(base_dir, "dataset", "nets", "single-intersection", "single-intersection.rou.xml")

    weights_path = "zhang_dqn_partial_weights.pt"

    if not os.path.exists(weights_path):
        sys.exit(f"❌ Error: Missing weights file at {weights_path}! Please run tco_SingleAgent_TorchTrain.py first.")

    print("Initializing Visual SUMO GUI Environment (Zhang Partial Observability Mode)...")
    base_env = SumoEnvironment(
        net_file=net_path, 
        route_file=route_path, 
        use_gui=True, 
        num_seconds=3600,  
        max_depart_delay=1, 
        delta_time=5, 
        single_agent=True
    )
    
    # Force evaluation wrapper injection
    env = ZhangPartialObservabilityWrapper(base_env, detection_rate=0.6, v_max=13.89)

    stateDim = base_env.observation_space.shape
    actionDim = base_env.action_space.n

    policyNet = DQNetwork(stateDim, actionDim)
    policyNet.load_state_dict(torch.load(weights_path))
    policyNet.eval()  
    print(f"💾 Successfully loaded trained Zhang DQN policy weights from {weights_path}!")

    try:
        print("\n▶ Press the green 'Play' button in the SUMO GUI window to start traffic flow...")
        obs, info = env.reset()
        done = False        
        total_eval_reward = 0.0
        
        while not done:
            state_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(policyNet(state_t)).item()
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_eval_reward += reward
            obs = next_obs
            
            done = terminated or truncated or (base_env.sim_step >= 3600)
            
        print(f"\n✔ Zhang Visual evaluation finished cleanly!")
        print(f"➡ Total Evaluation Zhang Reward: {total_eval_reward:.4f}")

    except Exception as e:
        print(f"\n[EVALUATION INTERRUPTED] Loop stopped due to: {e}")
    finally:
        if 'base_env' in locals() and hasattr(base_env, 'sumo') and base_env.sumo is not None:
            base_env.close()      

if __name__ == "__main__":
    main()
