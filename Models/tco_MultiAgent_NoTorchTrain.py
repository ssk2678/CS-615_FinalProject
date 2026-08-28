import os
import sys
import argparse
import csv
import numpy as np
# =====================================================================
# 🔥 Resolve the PettingZoo agent_selector module conflict
# =====================================================================
try:
    import pettingzoo.utils.agent_selector as az_mod
    import pettingzoo.utils
    if hasattr(az_mod, 'agent_selector'):
        pettingzoo.utils.agent_selector = az_mod.agent_selector
except ImportError:
    pass
# =====================================================================

import sumo_rl  # Required for parallel_env multi-agent processing
from DeepQNAgent import Agent
from multiagent_zhang import mask_observations, zhang_reward

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.insert(0, tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

def main():
    parser = argparse.ArgumentParser(description="Train a NumPy multi-agent DQN.")
    parser.add_argument("--scenario", choices=("2x2", "4x4"), required=True)
    parser.add_argument("--zhang", action="store_true", help="Use partial observations and the Zhang reward.")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, ".."))

    if args.scenario == "2x2":
        scenario_dir, net_file, route_file = "2x2grid", "2x2.net.xml", "2x2.rou.xml"
    else:
        scenario_dir, net_file, route_file = "4x4-Lucas", "4x4.net.xml", "4x4c1.rou.xml"

    net_path = os.path.join(base_dir, "dataset", "nets", scenario_dir, net_file)
    route_path = os.path.join(base_dir, "dataset", "nets", scenario_dir, route_file)

    print("Initializing Layered Multi-Agent NumPy Environment...")
    env = sumo_rl.parallel_env(
        net_file=net_path, 
        route_file=route_path, 
        use_gui=True, 
        num_seconds=3600, 
        max_depart_delay=1, 
        delta_time=5,
        reward_fn=zhang_reward if args.zhang else "diff-waiting-time"
    )
    
    # Run a brief baseline reset call to configure internal tracking properties
    env.reset()
    
    # DYNAMIC CONFIGURATION: Spawn independent scratch frameworks for each intersection node
    agents = {}
    for tl_id in env.possible_agents:
        # Peel off the first index scalar to pass an integer rather than a tuple object
        shape_val = env.observation_space(tl_id).shape
        state_dim = shape_val[0] if isinstance(shape_val, tuple) else shape_val
        action_dim = env.action_space(tl_id).n
        agents[tl_id] = Agent(state_dim, action_dim)
    
    # CSV Logging Setup
    mode_suffix = "_zhang" if args.zhang else ""
    log_file_path = f"noTorch_marl_{args.scenario}{mode_suffix}_training_rewards.csv"
    step_log_file_path = log_file_path.replace("_rewards.csv", "_step_metrics.csv")
    step_records = []
    with open(log_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Total_Reward", "Mean_Epsilon"])

    NUM_EPISODES = 3

    try:
        for episode in range(1, NUM_EPISODES + 1):
            print(f"\n--- Starting Layered NumPy Multi-Agent Episode {episode}/{NUM_EPISODES} ---")
            obs_dict, _ = env.reset() # PettingZoo wrapper returns (obs, info) on reset
            if args.zhang:
                obs_dict = mask_observations(obs_dict, env)
            print(f"Success! {args.scenario} Grid initialized with {len(obs_dict)} active NumPy agents.")
            done = False
            step_count = 0
            episode_reward = 0.0
            
            while not done:
                # Maps action evaluations sequentially across distinct brain memory records
                action_dict = {tl_id: agents[tl_id].select_action(obs) for tl_id, obs in obs_dict.items()}
                
                # Step execution returns 5 dictionary objects
                next_obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = env.step(action_dict)
                if args.zhang:
                    next_obs_dict = mask_observations(next_obs_dict, env)
                step_count += 1

                total_step_reward = sum(reward_dict.values())
                episode_reward += total_step_reward
                
                # Train and push transition matrices selectively per active intersection
                step_losses = {}
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
                    step_losses[tl_id] = agents[tl_id].train_step(32)

                mean_step_epsilon = np.mean([a.epsilon for a in agents.values()])
                for tl_id in obs_dict.keys():
                    step_records.append([
                        episode, step_count, tl_id, action_dict[tl_id], reward_dict[tl_id],
                        total_step_reward, episode_reward, agents[tl_id].epsilon,
                        mean_step_epsilon, step_losses[tl_id],
                        terminated_dict[tl_id], truncated_dict[tl_id]
                    ])
                    
                obs_dict = next_obs_dict                
                
                # Restored strict snake_case dictionary references for termination tracking
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

            # Archive each agent's weights under its traffic-light ID.
            weight_save_path = f"noTorch_marl_{args.scenario}{mode_suffix}_scratch_weights.npz"
            archive_data = {}
            for tl_id, agent_node in agents.items():
                archive_data[f"{tl_id}_W1"] = agent_node.policy_net.layer1.W
                archive_data[f"{tl_id}_b1"] = agent_node.policy_net.layer1.b
                archive_data[f"{tl_id}_W2"] = agent_node.policy_net.layer2.W
                archive_data[f"{tl_id}_b2"] = agent_node.policy_net.layer2.b
                
            np.savez(weight_save_path, **archive_data)
            print(f"💾 Saved comprehensive {args.scenario} multi-agent parameter matrices to:\n➡ {weight_save_path}")

    except Exception as e:
        print(f"\n[CRASH DETECTED] Script failed due to: {e}")
    finally:
        with open(step_log_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Episode", "Step", "Agent_ID", "Action", "Agent_Reward",
                "Total_Step_Reward", "Episode_Total_Reward", "Agent_Epsilon",
                "Mean_Epsilon", "Loss", "Terminated", "Truncated"
            ])
            writer.writerows(step_records)
        print(f"\nLayered NumPy training complete. Logs saved to {log_file_path} and {step_log_file_path}")
        if 'env' in locals() and hasattr(env, 'close'):
            env.close()

if __name__ == "__main__":
    main()
