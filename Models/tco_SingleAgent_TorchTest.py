import os
import sys
import torch
import torch.nn as nn
import numpy as np
from sumo_rl import SumoEnvironment

# KEEP THIS: Crucial for Linux virtual environment stability
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.insert(0, tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# Model Def (Must exactly match training architecture)
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

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, ".."))    
    net_path = os.path.join(base_dir, "dataset", "nets", "2way-single-intersection", "single-intersection.net.xml")
    route_path = os.path.join(base_dir, "dataset", "nets", "2way-single-intersection", "single-intersection-vhvh.rou.xml")

    weights_path = "dqn_traffic_weights.pt"

    if not os.path.exists(weights_path):
        sys.exit(f"Missing weights file at {weights_path}! Please run the training script first.")

    print("Initializing Visual SUMO GUI Environment...")
    # NOTE: use_gui=True enables the native SUMO visual interface window
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

    # Reconstruct network and load saved training parameters
    policyNet = DQNetwork(stateDim, actionDim)
    policyNet.load_state_dict(torch.load(weights_path))
    policyNet.eval()  # Disables gradient tracking overhead
    print("💾 Successfully loaded trained DQN policy weights!")

    try:
        print("\n▶ Press the green 'Play' button in the SUMO GUI window to start traffic flow...")
        obsDict = env.reset()
        done = False        
        total_eval_reward = 0.0
        
        while not done:
            actionDict = {}
            for tl_id, obs in obsDict.items():
                state_t = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    # Pure exploitation: always select the absolute highest mathematical Q-value
                    actionDict[tl_id] = torch.argmax(policyNet(state_t)).item()
            
            nextObsDict, rewardDict, terminatedDict, infoDict = env.step(actionDict)
            total_eval_reward += sum(rewardDict.values())
            obsDict = nextObsDict
            done = all(terminatedDict.values()) or (env.sim_step >= 3600)
            
        print(f"\n✔ Visual evaluation finished cleanly! Total Evaluation Reward: {total_eval_reward:.2f}")

    except Exception as e:
        print(f"\n[EVALUATION INTERRUPTED] Loop stopped due to: {e}")
    finally:
        env.close()      

if __name__ == "__main__":
    main()
