import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import csv  
from sumo_rl import SumoEnvironment

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
        self.gamma, self.epsilon, self.epsilon_min, self.epsilon_decay = 0.99, 1.0, 0.05, 0.995

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
        
    net_path = os.path.join(base_dir, "dataset", "nets", "2way-single-intersection", "single-intersection.net.xml")
    route_path = os.path.join(base_dir, "dataset", "nets", "2way-single-intersection", "single-intersection-gen.rou.xml")

    model_save_path = "dqn_traffic_weights.pt"       
    

    print("Initializing SUMO-RL Environment...")
    # FIXED: Restored 'num_seconds' and added 'max_depart_delay' to cleanly govern flow
    env = SumoEnvironment(
        net_file=net_path, 
        route_file=route_path, 
        use_gui=True, 
        num_seconds=3600,  
        max_depart_delay=1, 
        delta_time=5, 
        single_agent=False
    )
    
    sample_tl = env.ts_ids[0]
    stateDim = env.observation_spaces(sample_tl).shape[0]
    actionDim = env.action_spaces(sample_tl).n

    # CSV Logging Setup
    log_file_path = "training_rewards.csv"
    with open(log_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Total_Reward", "Final_Epsilon"])

    NUM_EPISODES = 5
    agent = PyTorchAgent(stateDim, actionDim)

    try:
        for episode in range(1, NUM_EPISODES + 1):
            print(f"\n--- Starting Episode {episode}/{NUM_EPISODES} ---")
            
            obsDict = env.reset()
            print("Success! Environment reset cleanly.")
            done = False        
            step_count = 0
            episode_reward = 0.0
            
            while not done:
                actionDict = {tl_id: agent.selectAction(obs) for tl_id, obs in obsDict.items()}            
                nextObsDict, rewardDict, terminatedDict, infoDict = env.step(actionDict)

                episode_reward += sum(rewardDict.values())
                
                for tl_id in obsDict.keys():
                    is_done = terminatedDict[tl_id]
                    agent.memory.push(obsDict[tl_id], actionDict[tl_id], rewardDict[tl_id], nextObsDict[tl_id], is_done)

                agent.trainStep(32)
                obsDict = nextObsDict
                
                # FIXED: Check natural termination OR force stop when internal clock ticks past limit
                done = all(terminatedDict.values()) or (env.sim_step >= 3600)
                
                step_count += 1
                
                if step_count % 100 == 0:
                    agent.targetNet.load_state_dict(agent.policyNet.state_dict())
                    
                if step_count % 100 == 0:  
                    avg_r = np.mean(list(rewardDict.values()))
                    print(f"Episode {episode} | Step {step_count} | Current Step Avg Reward: {avg_r:.2f} | Epsilon: {agent.epsilon:.3f}")
                    
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
        print(f"\nTraffic training loop completed! Metrics saved to {log_file_path}")
        # FIXED: Protective gate avoids calling close() if __init__ failed before instantiation finished
        if 'env' in locals() and hasattr(env, 'sumo') and env.sumo is not None:
            env.close()      

if __name__ == "__main__":
    main()
