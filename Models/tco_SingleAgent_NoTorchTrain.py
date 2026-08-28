import os
import sys
import argparse
import csv
import numpy as np
from sumo_rl import SumoEnvironment
from DeepQNAgent import Agent
from multiagent_zhang import mask_observation, zhang_reward

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.insert(0, tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

def main():
    parser = argparse.ArgumentParser(description="Train a single-agent NumPy DQN.")
    parser.add_argument("--zhang", action="store_true", help="Use partial observations and the Zhang reward.")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, ".."))
    
    net_path = os.path.join(base_dir, "dataset", "nets", "single-intersection", "single-intersection.net.xml")
    route_path = os.path.join(base_dir, "dataset", "nets", "single-intersection", "single-intersection.rou.xml")

    print("Initializing Layered Object NumPy Environment...")
    env = SumoEnvironment(
        net_file=net_path, route_file=route_path, use_gui=True, 
        num_seconds=3600, max_depart_delay=1, delta_time=5, single_agent=True,
        reward_fn=zhang_reward if args.zhang else "diff-waiting-time"
    )
    
    # Clean attribute extractions mapped to single-agent gymnasium configurations
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    log_file_path = "noTorch_zhang_training_rewards.csv" if args.zhang else "noTorch_training_rewards.csv"
    step_log_file_path = log_file_path.replace("_rewards.csv", "_step_metrics.csv")
    step_records = []
    with open(log_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Total_Reward", "Final_Epsilon"])

    NUM_EPISODES = 3
    agent = Agent(state_dim, action_dim)

    try:
        for episode in range(1, NUM_EPISODES + 1):
            print(f"\n--- Starting Layered NumPy Episode {episode}/{NUM_EPISODES} ---")
            
            # Properly unpack single-agent Gym tuple arrays
            obs, info = env.reset()
            if args.zhang:
                obs = mask_observation(obs, action_dim)
            print("Success! Environment reset cleanly.")
            done = False
            step_count = 0
            episode_reward = 0.0
            
            while not done:
                # Stripped dictionary comprehension; directly call the raw observation array
                action = agent.select_action(obs)
                
                # Directly unpack standard scalar return metrics
                next_obs, reward, terminated, truncated, info = env.step(action)
                if args.zhang:
                    next_obs = mask_observation(next_obs, action_dim)
                step_count += 1

                episode_reward += reward
                
                # Push clean flat step data pairs straight to memory buffer
                agent.memory.push(obs, action, reward, next_obs, terminated)
                    
                loss = agent.train_step(32)
                obs = next_obs

                step_records.append([
                    episode, step_count, action, reward, episode_reward,
                    agent.epsilon, loss, terminated, truncated
                ])
                
                # Aligned single agent termination validation metrics
                done = terminated or truncated or (env.sim_step >= 3600)
                
                if step_count % 100 == 0:
                    agent.target_net.copy_weights_from(agent.policy_net)
                if step_count % 100 == 0:
                    print(f"Layered NumPy Ep {episode} | Step {step_count} | Step Reward: {reward:.2f} | Epsilon: {agent.epsilon:.3f}")
            
            agent.target_net.copy_weights_from(agent.policy_net)
            print(f"✔ Layered NumPy Episode {episode} Finished! Total Reward: {episode_reward:.2f}")
            
            with open(log_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([episode, episode_reward, agent.epsilon])

            # Export weights updated to fit the layered properties object structures
            weight_save_path = "noTorch_zhang_scratch_weights.npz" if args.zhang else "noTorch_scratch_weights.npz"
            np.savez(weight_save_path, 
                     W1=agent.policy_net.layer1.W, b1=agent.policy_net.layer1.b, 
                     W2=agent.policy_net.layer2.W, b2=agent.policy_net.layer2.b)
            print(f"💾 Saved layer arrays directly to:\n➡ {weight_save_path}")

    except Exception as e:
        print(f"\n[CRASH DETECTED] Script failed due to: {e}")
    finally:
        with open(step_log_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Episode", "Step", "Action", "Reward", "Episode_Total_Reward",
                "Epsilon", "Loss", "Terminated", "Truncated"
            ])
            writer.writerows(step_records)
        print(f"\nLayered NumPy training complete. Logs saved to {log_file_path} and {step_log_file_path}")
        if 'env' in locals() and hasattr(env, 'sumo') and env.sumo is not None:
            env.close()

if __name__ == "__main__":
    main()
