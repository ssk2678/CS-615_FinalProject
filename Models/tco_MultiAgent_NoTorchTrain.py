import os
import sys
import csv
import numpy as np
# =====================================================================
# 🔥 CRITICAL BUGFIX: Resolve the PettingZoo agent_selector module conflict
# =====================================================================
try:
    import pettingzoo.utils.agent_selector as az_mod
    import pettingzoo.utils
    if hasattr(az_mod, 'agent_selector'):
        pettingzoo.utils.agent_selector = az_mod.agent_selector
except ImportError:
    pass
# =====================================================================

import sumo_rl  # FIXED: Required for parallel_env multi-agent processing
from DeepQNAgent import Agent

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.insert(0, tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, ".."))
    
    # FIXED: Re-targeted network paths to look into the 2x2/3x3 grid configurations
    net_path = os.path.join(base_dir, "dataset", "nets", "3x3grid", "3x3Grid2lanes.net.xml")
    route_path = os.path.join(base_dir, "dataset", "nets", "3x3grid", "routes14000.rou.xml")

    print("Initializing Layered Multi-Agent NumPy Environment...")
    # FIXED: Replaced SumoEnvironment with parallel_env interface for MARL routing
    env = sumo_rl.parallel_env(
        net_file=net_path, 
        route_file=route_path, 
        use_gui=True, 
        num_seconds=3600, 
        max_depart_delay=1, 
        delta_time=5
    )
    
    # Run a brief baseline reset call to configure internal tracking properties
    env.reset()
    
    # DYNAMIC CONFIGURATION: Spawn independent scratch frameworks for each intersection node
    agents = {}
    for tl_id in env.possible_agents:
        state_dim = env.observation_space(tl_id).shape[0]
        action_dim = env.action_space(tl_id).n
        agents[tl_id] = Agent(state_dim, action_dim)
    
    # CSV Logging Setup
    log_file_path = "noTorch_marl_training_rewards.csv"
    with open(log_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Total_Reward", "Mean_Epsilon"])

    NUM_EPISODES = 5

    try:
        for episode in range(1, NUM_EPISODES + 1):
            print(f"\n--- Starting Layered NumPy Multi-Agent Episode {episode}/{NUM_EPISODES} ---")
            obs_dict, _ = env.reset() # PettingZoo wrapper returns (obs, info) on reset
            print(f"Success! 3x3 Grid initialized with {len(obs_dict)} active NumPy agents.")
            done = False
            step_count = 0
            episode_reward = 0.0
            
            while not done:
                # FIXED: Maps action evaluations sequentially across distinct brain memory records
                action_dict = {tl_id: agents[tl_id].select_action(obs) for tl_id, obs in obs_dict.items()}
                
                # Step execution returns 5 dictionary objects
                next_obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = env.step(action_dict)
                step_count += 1

                episode_reward += sum(reward_dict.values())
                
                # Train and push transition matrices selectively per active intersection
                for tl_id in obs_dict.keys():
                    is_done = terminated_dict[tl_id] or truncated_dict[tl_id]
                    
                    # Store experience directly to that specific agent ID buffer pool
                    agents[tl_id].memory.push(
                        obs_dict[tl_id], 
                        action_dict[tl_id], 
                        reward_dict[tl_id], 
                        next_obs_dict[tl_id], 
                        is_done
                    )
                    
                    # Compute handwritten gradient descent backpropagation steps for this agent node
                    agents[tl_id].train_step(32)
                    
                obs_dict = next_obs_dict                
                
                # FIXED: Verifies step clock mathematically, bypassing internal wrapper bugs completely
                done = all(terminated_dict.values()) or all(truncated_dict.values()) or (step_count * 5 >= 3600)
                
                # Periodically sync parameters and print live metric feedback indicators
                if step_count % 100 == 0:
                    for tl_id in agents.keys():
                        agents[tl_id].target_net.copy_weights_from(agents[tl_id].policy_net)
                        
                if step_count % 100 == 0:
                    avg_r = np.mean(list(reward_dict.values()))
                    mean_eps = np.mean([a.epsilon for a in agents.values()])
                    print(f"NumPy MARL Ep {episode} | Step {step_count} | Grid Avg Reward: {avg_r:.2f} | Mean Epsilon: {mean_eps:.3f}")
            
            # Synchronize target weight arrays at episode completion boundaries
            for tl_id in agents.keys():
                agents[tl_id].target_net.copy_weights_from(agents[tl_id].policy_net)
                
            mean_final_eps = np.mean([a.epsilon for a in agents.values()])
            print(f"✔ Layered NumPy Episode {episode} Finished! Total Combined Grid Reward: {episode_reward:.2f}")
            
            with open(log_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([episode, episode_reward, mean_final_eps])

            # FIXED: Archive all 4 distinct sets of weights under tagged index headers to prevent overwriting
            weight_save_path = "noTorch_marl_3x3_scratch_weights.npz"
            archive_data = {}
            for tl_id, agent_node in agents.items():
                archive_data[f"{tl_id}_W1"] = agent_node.policy_net.layer1.W
                archive_data[f"{tl_id}_b1"] = agent_node.policy_net.layer1.b
                archive_data[f"{tl_id}_W2"] = agent_node.policy_net.layer2.W
                archive_data[f"{tl_id}_b2"] = agent_node.policy_net.layer2.b
                
            np.savez(weight_save_path, **archive_data)
            print(f"💾 Saved comprehensive 3x3 multi-agent parameter matrices to:\n➡ {weight_save_path}")

    except Exception as e:
        print(f"\n[CRASH DETECTED] Script failed due to: {e}")
    finally:
        print(f"\nLayered NumPy training complete. Logs saved to {log_file_path}")
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    main()
