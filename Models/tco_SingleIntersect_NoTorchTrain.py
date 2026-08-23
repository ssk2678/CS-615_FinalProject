import os
import sys
import csv
import numpy as np
from sumo_rl import SumoEnvironment
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
    
    net_path = os.path.join(base_dir, "dataset", "nets", "single-intersection", "single-intersection.net.xml")
    route_path = os.path.join(base_dir, "dataset", "nets", "single-intersection", "single-intersection.rou.xml")

    print("Initializing Layered Object NumPy Environment...")
    env = SumoEnvironment(
        net_file=net_path, route_file=route_path, use_gui=True, 
        num_seconds=3600, max_depart_delay=1, delta_time=5, single_agent=False
    )
    
    
    sample_tl = env.ts_ids[0]    
    
    state_dim = env.observation_spaces(sample_tl).shape[0]
    action_dim = env.action_spaces(sample_tl).n
    
    log_file_path = "noTorch_training_rewards.csv"
    with open(log_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Total_Reward", "Final_Epsilon"])

    NUM_EPISODES = 5
    agent = Agent(state_dim, action_dim)

    try:
        for episode in range(1, NUM_EPISODES + 1):
            print(f"\n--- Starting Layered NumPy Episode {episode}/{NUM_EPISODES} ---")
            obs_dict = env.reset()
            print("Success! Environment reset cleanly.")
            done = False
            step_count = 0
            episode_reward = 0.0
            
            while not done:
                action_dict = {tl_id: agent.select_action(obs) for tl_id, obs in obs_dict.items()}
                next_obs_dict, reward_dict, terminated_dict, info_dict = env.step(action_dict)
                episode_reward += sum(reward_dict.values())
                
                for tl_id in obs_dict.keys():
                    is_done = terminated_dict[tl_id]
                    agent.memory.push(obs_dict[tl_id], action_dict[tl_id], reward_dict[tl_id], next_obs_dict[tl_id], is_done)
                    
                agent.train_step(32)
                obs_dict = next_obs_dict
                done = all(terminated_dict.values()) or (env.sim_step >= 3600)
                step_count += 1
                
                if step_count % 100 == 0:
                    agent.target_net.copy_weights_from(agent.policy_net)
                if step_count % 100 == 0:
                    avg_r = np.mean(list(reward_dict.values()))
                    print(f"Layered NumPy Ep {episode} | Step {step_count} | Avg Reward: {avg_r:.2f} | Epsilon: {agent.epsilon:.3f}")
            
            agent.target_net.copy_weights_from(agent.policy_net)
            print(f"✔ Layered NumPy Episode {episode} Finished! Total Reward: {episode_reward:.2f}")
            
            with open(log_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([episode, episode_reward, agent.epsilon])

            # Export weights updated to fit the layered properties object structures
            weight_save_path = "noTorch_scratch_weights.npz"
            np.savez(weight_save_path, 
                     W1=agent.policy_net.layer1.W, b1=agent.policy_net.layer1.b, 
                     W2=agent.policy_net.layer2.W, b2=agent.policy_net.layer2.b)
            print(f"💾 Saved layer arrays directly to:\n➡ {weight_save_path}")

    except Exception as e:
        print(f"\n[CRASH DETECTED] Script failed due to: {e}")
    finally:
        print(f"\nLayered NumPy training complete. Logs saved to {log_file_path}")
        if 'env' in locals() and hasattr(env, 'sumo') and env.sumo is not None:
            env.close()

if __name__ == "__main__":
    main()
