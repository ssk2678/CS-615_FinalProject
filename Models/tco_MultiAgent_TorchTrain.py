import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import csv

# =====================================================================
# 🔥 Resolve the PettingZoo agent_selector module conflict
# =====================================================================
try:
    import pettingzoo.utils.agent_selector as az_mod
    import pettingzoo.utils
    # Check if the module is masking the class function
    if hasattr(az_mod, 'agent_selector'):
        pettingzoo.utils.agent_selector = az_mod.agent_selector
except ImportError:
    pass
# =====================================================================  
import sumo_rl  # Required for parallel_env MARL mapping

# KEEP THIS: Crucial for Linux virtual environment stability
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.insert(0, tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


# Model Def (Perfectly matches 2-layer custom framework architecture)
class DQNetwork(nn.Module):
    def __init__(self, stateDim, actionDim):
        super(DQNetwork, self).__init__()
        # Robustly handle both raw integers and tuple structures to maintain test synchronization
        input_features = stateDim[0] if isinstance(stateDim, tuple) else stateDim
        self.fullyConnected = nn.Sequential(
            nn.Linear(input_features, 64),
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
            return
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

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, ".."))
        
    net_path = os.path.join(base_dir, "dataset", "nets", "3x3grid", "3x3Grid2lanes.net.xml")
    route_path = os.path.join(base_dir, "dataset", "nets", "3x3grid", "routes14000.rou.xml")

    model_save_path = "marl_3x3_traffic_weights.pt"       
    
    print("Initializing Multi-Agent SUMO-RL Environment...")
    env = sumo_rl.parallel_env(
        net_file=net_path, 
        route_file=route_path, 
        use_gui=True, 
        num_seconds=3600,  
        max_depart_delay=1, 
        delta_time=5
    )
    
    # Run a brief reset call to initialize internal agent tracking spaces
    env.reset()

    # DYNAMIC CONFIGURATION: Spawn independent brain network layers for every intersection
    agents = {}
    for tl_id in env.possible_agents:
        # Pass the clean shape matrix layout uniformly to align with evaluation files
        stateDim = env.observation_space(tl_id).shape
        actionDim = env.action_space(tl_id).n
        agents[tl_id] = PyTorchAgent(stateDim, actionDim)

    # CSV Logging Setup
    log_file_path = "marl_training_rewards.csv"
    with open(log_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Total_Reward", "Mean_Epsilon"])

    NUM_EPISODES = 3

    try:
        for episode in range(1, NUM_EPISODES + 1):
            print(f"\n--- Starting Multi-Agent Episode {episode}/{NUM_EPISODES} ---")
            
            obsDict, _ = env.reset()
            print(f"Success! 3x3 Grid initialized with {len(obsDict)} active agents.")
            done = False        
            step_count = 0
            episode_reward = 0.0
            
            while not done:
                # Maps actions sequentially across distinct brain instances using ID keys
                actionDict = {tl_id: agents[tl_id].selectAction(obs) for tl_id, obs in obsDict.items()}            
                
                # Step parallel execution returns 5 dictionary objects
                nextObsDict, rewardDict, terminated_dict, truncated_dict, infoDict = env.step(actionDict)
                step_count += 1

                episode_reward += sum(rewardDict.values())
                
                # Train and push memories selectively per active agent
                for tl_id in obsDict.keys():
                    is_done = terminated_dict[tl_id] or truncated_dict[tl_id]
                    
                    # Store to corresponding agent replay memory pool
                    agents[tl_id].memory.push(
                        obsDict[tl_id], 
                        actionDict[tl_id], 
                        rewardDict[tl_id], 
                        nextObsDict[tl_id], 
                        is_done
                    )
                    
                    # Update parameters for this specific traffic signal node
                    agents[tl_id].trainStep(32)

                obsDict = nextObsDict
                                
                # Check execution end conditions across multi-agent nodes
                done = all(terminated_dict.values()) or all(truncated_dict.values()) or (step_count * 5 >= 3600)
                
                # Periodically update target networks and report logging metrics
                if step_count % 100 == 0:
                    for tl_id in agents.keys():
                        agents[tl_id].targetNet.load_state_dict(agents[tl_id].policyNet.state_dict())
                    
                    avg_r = np.mean(list(rewardDict.values()))
                    mean_eps = np.mean([a.epsilon for a in agents.values()])
                    print(f"E{episode} | Step {step_count} | Grid Step Avg Reward: {avg_r:.2f} | Mean Epsilon: {mean_eps:.3f}")
                    
            # Synchronize target network state locks at episode boundaries
            for tl_id in agents.keys():
                agents[tl_id].targetNet.load_state_dict(agents[tl_id].policyNet.state_dict())
                
            mean_final_eps = np.mean([a.epsilon for a in agents.values()])
            print(f"✔ Episode {episode} Finished! Total Combined Grid Reward: {episode_reward:.2f}")
            
            with open(log_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([episode, episode_reward, mean_final_eps])

            # Seamlessly patch, pack, and export structural weights dictionary parameters
            torch.save({tl_id: agent.policyNet.state_dict() for tl_id, agent in agents.items()}, model_save_path)
            print(f"💾 Success! Multi-Agent network weights package saved to:\n➡ {model_save_path}")

    except Exception as e:
        print(f"\n[CRASH DETECTED] Script failed due to: {e}")

    finally:
        print(f"\nMulti-Agent training loop completed! Metrics saved to {log_file_path}")
        if 'env' in locals() and hasattr(env, 'close'):
            env.close()      

if __name__ == "__main__":
    main()
