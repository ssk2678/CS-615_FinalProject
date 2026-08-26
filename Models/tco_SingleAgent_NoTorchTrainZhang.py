import os
import sys
import csv
import random
import numpy as np
from sumo_rl import SumoEnvironment
from DeepQNAgent import Agent

# KEEP THIS: Crucial for Linux virtual environment stability
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.insert(0, tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


class ZhangPartialObservabilityWrapper:
    """Emulates partial vehicle detection and implements Zhang et al. reward function using NumPy."""
    def __init__(self, env, detection_rate=0.6, v_max=13.89):
        self.env = env
        self.detection_rate = detection_rate
        self.v_max = v_max  # Default ~50 km/h (13.89 m/s)
        
    def _apply_partial_observability(self, raw_obs):
        """Simulates partial observation by masking out a fraction of vehicle data counts."""
        masked_obs = np.array(raw_obs, dtype=np.float32)
        return masked_obs * self.detection_rate

    def _calculate_zhang_reward(self):
        """Computes r_t = (-1/v_max) * [v_max - v_S(t)] based on active lane tracks."""
        # Explicitly isolate the string ID out of the list to prevent unhashable dict keys crashes
        ts_id = self.env.ts_ids[0] if isinstance(self.env.ts_ids, list) else self.env.ts_ids
        lanes = self.env.traffic_signals[ts_id].lanes
        vehicle_speeds = []
        
        for lane in lanes:
            veh_list = self.env.sumo.lane.getLastStepVehicleIDs(lane)
            for veh in veh_list:
                # Stochastic sample drop based on detection visibility floor
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
        
        # Mutate trajectory vectors to match the paper's criteria
        masked_next_obs = self._apply_partial_observability(next_raw_obs)
        custom_reward = self._calculate_zhang_reward()
        
        return masked_next_obs, custom_reward, terminated, truncated, info


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, ".."))
    
    # Targeting the single-intersection baseline files as requested
    net_path = os.path.join(base_dir, "dataset", "nets", "single-intersection", "single-intersection.net.xml")
    route_path = os.path.join(base_dir, "dataset", "nets", "single-intersection", "single-intersection.rou.xml")

    weight_save_path = "noTorch_zhang_scratch_weights.npz"
    log_file_path = "noTorch_zhang_training_rewards.csv"

    print("Initializing Base SUMO Engine...")
    base_env = SumoEnvironment(
        net_file=net_path, 
        route_file=route_path, 
        use_gui=True, 
        num_seconds=3600, 
        max_depart_delay=1, 
        delta_time=5, 
        single_agent=True
    )
    
    # Inject our partial observability wrapper layer
    env = ZhangPartialObservabilityWrapper(base_env, detection_rate=0.6, v_max=13.89)
    
    # Extract dimensions correctly for single agent
    shape_val = base_env.observation_space.shape
    state_dim = shape_val[0] if isinstance(shape_val, tuple) else shape_val
    action_dim = base_env.action_space.n
    
    # Initialize custom Agent model
    agent = Agent(state_dim, action_dim)
    
    # CSV Logger Setup
    with open(log_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Total_Reward", "Final_Epsilon"])

    NUM_EPISODES = 3  # Matches the validated 3-episode stability curve

    try:
        for episode in range(1, NUM_EPISODES + 1):
            print(f"\n--- Starting Custom NumPy Zhang Episode {episode}/{NUM_EPISODES} ---")
            obs, info = env.reset()
            print("Success! Environment masked and initialized cleanly.")
            done = False
            step_count = 0
            episode_reward = 0.0
            
            while not done:
                # Select action using pure matrix forwarding math
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                step_count += 1

                episode_reward += reward
                
                # Push transitions to ReplayBuffer
                agent.memory.push(obs, action, reward, next_obs, terminated)
                    
                # Run Adam backpropagation matrix transformations
                agent.train_step(32)
                obs = next_obs
                
                done = terminated or truncated or (base_env.sim_step >= 3600)
                
                if step_count % 100 == 0:
                    agent.target_net.copy_weights_from(agent.policy_net)
                if step_count % 100 == 0:
                    print(f"NumPy Zhang Ep {episode} | Step {step_count} | Step Reward: {reward:.4f} | Epsilon: {agent.epsilon:.3f}")
            
            agent.target_net.copy_weights_from(agent.policy_net)
            print(f"✔ Custom NumPy Zhang Episode {episode} Finished! Total Reward: {episode_reward:.2f}")
            
            with open(log_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([episode, episode_reward, agent.epsilon])

            # Export weights directly to manual archive layout
            np.savez(weight_save_path, 
                     W1=agent.policy_net.layer1.W, b1=agent.policy_net.layer1.b, 
                     W2=agent.policy_net.layer2.W, b2=agent.policy_net.layer2.b)
            print(f"💾 Saved layer matrices directly to:\n➡ {weight_save_path}")

    except Exception as e:
        print(f"\n[CRASH DETECTED] Script failed due to: {e}")
    finally:
        print(f"\nCustom NumPy Zhang training complete. Logs saved to {log_file_path}")
        if 'base_env' in locals() and hasattr(base_env, 'sumo') and base_env.sumo is not None:
            base_env.close()

if __name__ == "__main__":
    main()
