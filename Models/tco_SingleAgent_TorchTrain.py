import os
import sys
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import csv  
from sumo_rl import SumoEnvironment
from multiagent_zhang import mask_observation, zhang_reward

# KEEP THIS: Crucial for Linux virtual environment stability
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.insert(0, tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


# Model Def
class DQNetwork(nn.Module):
    def __init__(self, stateDim, actionDim):
        super(DQNetwork, self).__init__()
        self.fullyConnected = nn.Sequential(
            nn.Linear(stateDim, 64),
            nn.ReLU(),            
            nn.Linear(64, actionDim)
        )
    def forward(self, x):
        return self.fullyConnected(x)

class PyTorchReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, nextState, done):
        self.buffer.append((state, action, reward, nextState, done))
    def sample(self, batch_size):
        state, action, reward, nextState, done = zip(*random.sample(self.buffer, batch_size))
        return (torch.FloatTensor(np.array(state)),
                torch.LongTensor(action),
                torch.FloatTensor(reward),
                torch.FloatTensor(np.array(nextState)),
                torch.FloatTensor(done))
    def __len__(self):
        return len(self.buffer)

class PyTorchAgent:
    def __init__(self, stateDim, actionDim):
        self.actionDim = actionDim
        self.policyNet = DQNetwork(stateDim, actionDim)
        self.targetNet = DQNetwork(stateDim, actionDim)
        self.targetNet.load_state_dict(self.policyNet.state_dict())
        
        self.optimizer = optim.Adam(self.policyNet.parameters(), lr=1e-3)
        self.memory = PyTorchReplayBuffer(10000)
        self.gamma, self.epsilon, self.epsilon_min, self.epsilon_decay = 0.99, 1.0, 0.05, 0.998

    def selectAction(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionDim - 1)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.policyNet(state_t)).item()

    def trainStep(self, batchSize):
        if len(self.memory) < batchSize: 
            return None
        states, actions, rewards, nextStates, dones = self.memory.sample(batchSize)
        currentQ = self.policyNet(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            maxNextQ = self.targetNet(nextStates).max(1)[0]
            targetQ = rewards + (1 - dones) * self.gamma * maxNextQ
        
        loss = nn.MSELoss()(currentQ, targetQ)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min: 
            self.epsilon *= self.epsilon_decay
        return loss.item()

def main():
    parser = argparse.ArgumentParser(description="Train a single-agent PyTorch DQN.")
    parser.add_argument("--zhang", action="store_true", help="Use partial observations and the Zhang reward.")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, ".."))
        
    net_path = os.path.join(base_dir, "dataset", "nets", "single-intersection", "single-intersection.net.xml")
    route_path = os.path.join(base_dir, "dataset", "nets", "single-intersection", "single-intersection.rou.xml")

    model_save_path = "zhang_dqn_partial_weights.pt" if args.zhang else "dqn_traffic_weights.pt"
    

    print("Initializing SUMO-RL Environment...")
    env = SumoEnvironment(
        net_file=net_path, 
        route_file=route_path, 
        use_gui=True, 
        num_seconds=3600,  
        max_depart_delay=1, 
        delta_time=5, 
        single_agent=True,
        reward_fn=zhang_reward if args.zhang else "diff-waiting-time"
    )
    
    # Clean extraction for single agent
    stateDim = env.observation_space.shape[0]
    actionDim = env.action_space.n

    # CSV Logging Setup
    log_file_path = "zhang_training_rewards.csv" if args.zhang else "training_rewards.csv"
    step_log_file_path = log_file_path.replace("_rewards.csv", "_step_metrics.csv")
    step_records = []
    with open(log_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Total_Reward", "Final_Epsilon"])

    NUM_EPISODES = 3
    agent = PyTorchAgent(stateDim, actionDim)

    try:
        for episode in range(1, NUM_EPISODES + 1):
            print(f"\n--- Starting Episode {episode}/{NUM_EPISODES} ---")
            
            # Unpack Gymnasium single-agent tuple properly
            obs, info = env.reset()
            if args.zhang:
                obs = mask_observation(obs, actionDim)
            print("Success! Environment reset cleanly.")
            done = False        
            step_count = 0
            episode_reward = 0.0
            
            while not done:
                # 'obs' variable is now correctly populated from env.reset()
                action = agent.selectAction(obs)            
                next_obs, reward, terminated, truncated, info = env.step(action)
                if args.zhang:
                    next_obs = mask_observation(next_obs, actionDim)
                step_count += 1

                episode_reward += reward
                
                # Directly push the single-agent step data
                agent.memory.push(obs, action, reward, next_obs, terminated)

                loss = agent.trainStep(32)
                obs = next_obs

                step_records.append([
                    episode, step_count, action, reward, episode_reward,
                    agent.epsilon, loss, terminated, truncated
                ])
                
                # SUMO-RL single agent handles termination naturally
                done = terminated or truncated or (env.sim_step >= 3600)
                
                if step_count % 100 == 0:
                    agent.targetNet.load_state_dict(agent.policyNet.state_dict())
                    
                if step_count % 100 == 0:  
                    print(f"Episode {episode} | Step {step_count} | Current Step Reward: {reward:.2f} | Epsilon: {agent.epsilon:.3f}")
                    
            agent.targetNet.load_state_dict(agent.policyNet.state_dict())
            print(f"✔ Episode {episode} Finished! Total Accumulated Reward: {episode_reward:.2f}")
            
            with open(log_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([episode, episode_reward, agent.epsilon])

            # Save model weights before environment teardown            
            torch.save(agent.policyNet.state_dict(), model_save_path)
            print(f"💾 Success! Best policy network weights saved to:\n➡ {model_save_path}")

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
        print(f"\nTraffic training loop completed! Metrics saved to {log_file_path} and {step_log_file_path}")
        if 'env' in locals() and hasattr(env, 'sumo') and env.sumo is not None:
            env.close()      

if __name__ == "__main__":
    main()
