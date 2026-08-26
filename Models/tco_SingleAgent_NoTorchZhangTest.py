import os
import sys
import random
import numpy as np
from sumo_rl import SumoEnvironment
from DeepQNetwork import DQN

# KEEP THIS: Crucial for Linux virtual environment stability
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.insert(0, tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


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
        """Computes r_t = (-1/v_max) * [v_max - v_S(t)] based on active lane tracks."""
        # FIXED: Explicitly grab index 0 from the list to completely bypass unhashable type list errors
        if isinstance(self.env.ts_ids, list):
            ts_id = self.env.ts_ids[0]
        else:
            ts_id = self.env.ts_ids
            
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
    
    # Target standard baseline single-intersection network configurations
    net_path = os.path.join(base_dir, "dataset", "nets", "single-intersection", "single-intersection.net.xml")
    route_path = os.path.join(base_dir, "dataset", "nets", "single-intersection", "single-intersection.rou.xml")

    weights_path = "noTorch_zhang_scratch_weights.npz"

    if not os.path.exists(weights_path):
        sys.exit(f"❌ Error: Missing trained NumPy weights file at {weights_path}! Please run tco_SingleAgent_NoTorchTrainZhang.py first.")

    print("Initializing Visual SUMO GUI Environment (Custom NumPy Zhang Mode)...")
    base_env = SumoEnvironment(
        net_file=net_path, 
        route_file=route_path, 
        use_gui=True, 
        num_seconds=3600,  
        max_depart_delay=1, 
        delta_time=5, 
        single_agent=True
    )
    
    # Inject evaluation wrapper
    env = ZhangPartialObservabilityWrapper(base_env, detection_rate=0.6, v_max=13.89)

    # FIXED: Extract clean integer dimensions out of the shape tuple by explicitly grabbing index 0
    shape_val = base_env.observation_space.shape
    state_dim = shape_val[0] if isinstance(shape_val, tuple) else shape_val
    action_dim = base_env.action_space.n

    # Reconstruct custom network model
    net = DQN(state_dim, action_dim)
    
    # Load raw matrix properties directly into layer properties
    print("Loading raw NumPy parameter block weights...")
    loaded_data = np.load(weights_path)
    net.layer1.W = loaded_data['W1']
    net.layer1.b = loaded_data['b1']
    net.layer2.W = loaded_data['W2']
    net.layer2.b = loaded_data['b2']
    print("💾 Successfully loaded trained custom matrix array weights!")

    try:
        print("\n▶ Press the green 'Play' button in the SUMO GUI window to start traffic flow...")
        obs, info = env.reset()
        done = False        
        total_eval_reward = 0.0
        
        while not done:
            # Process state array through standard manual matrix forward steps
            q_values = net.forward(obs.reshape(1, -1))
            action = np.argmax(q_values)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_eval_reward += reward
            obs = next_obs
            
            done = terminated or truncated or (base_env.sim_step >= 3600)
            
        print(f"\n✔ Custom NumPy Zhang Evaluation finished cleanly!")
        print(f"➡ Total Evaluation Zhang Reward: {total_eval_reward:.4f}")

    except Exception as e:
        print(f"\n[EVALUATION INTERRUPTED] Loop stopped due to: {e}")
    finally:
        if 'base_env' in locals() and hasattr(base_env, 'sumo') and base_env.sumo is not None:
            base_env.close()      

if __name__ == "__main__":
    main()
