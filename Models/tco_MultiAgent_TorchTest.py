import os
import sys
import torch
import torch.nn as nn
import numpy as np
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
import sumo_rl  # Required for parallel_env multi-agent execution wrapper

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
        # Extract the raw integer out of the tuple safely to prevent nn.Linear failures
        if isinstance(stateDim, tuple):
            input_features = stateDim[0]
        else:
            input_features = stateDim
            
        self.fullyConnected = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),           
            nn.Linear(64, actionDim)
        )
    def forward(self, x):
        return self.fullyConnected(x)

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, ".."))    
    
    # Re-targeted network paths to look into the 2x2/3x3 grid configurations
    net_path = os.path.join(base_dir, "dataset", "nets", "2x2grid", "2x2.net.xml")
    route_path = os.path.join(base_dir, "dataset", "nets", "2x2grid", "2x2.rou.xml")

    weights_path = "marl_2x2_traffic_weights.pt"

    if not os.path.exists(weights_path):
        sys.exit(f"Missing weights file at {weights_path}! Please run the training script first.")

    print("Initializing Visual Multi-Agent SUMO GUI Environment...")
    env = sumo_rl.parallel_env(
        net_file=net_path, 
        route_file=route_path, 
        use_gui=True, 
        num_seconds=3600,  
        max_depart_delay=1, 
        delta_time=5
    )
    
    # Trigger a baseline environment initialization step to pull topology dimensions
    env.reset()

    # Load the archived multi-agent weights dictionary
    saved_weights_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    
    # DYNAMIC MAPPING: Build an independent network instance for each unique traffic light node
    policy_networks = {}
    for tl_id in env.possible_agents:
        # Pulled shape attribute cleanly via robust dimension extraction to bypass version conflicts
        stateDim = env.observation_space(tl_id).shape
        actionDim = env.action_space(tl_id).n
        
        # Instantiate 2-layer network framework for this specific junction node
        net = DQNetwork(stateDim, actionDim)
        
        # Pull the specific weight sub-dictionary for this exact traffic light ID
        net.load_state_dict(saved_weights_dict[tl_id])
        net.eval()  # Disables gradient tracking overhead
        policy_networks[tl_id] = net

    print("💾 Successfully loaded and mapped all trained 2x2 MARL policy weights!")

    try:
        print("\n▶ Press the green 'Play' button in the SUMO GUI window to start grid traffic flow...")
        obsDict, _ = env.reset() # PettingZoo wrapper returns (obs, info) on reset
        done = False        
        total_eval_reward = 0.0
        step_count = 0
        
        while not done:
            actionDict = {}
            for tl_id, obs in obsDict.items():
                state_t = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    # Consult the SPECIFIC network mapped to this active traffic light key
                    active_net = policy_networks[tl_id]
                    actionDict[tl_id] = torch.argmax(active_net(state_t)).item()
            
            # Step execution returns 5 dictionary objects
            nextObsDict, rewardDict, terminatedDict, truncatedDict, infoDict = env.step(actionDict)
            
            total_eval_reward += sum(rewardDict.values())
            obsDict = nextObsDict
            step_count += 1  
            
            # Bypasses library version wrapper bugs entirely using step calculations
            done = all(terminatedDict.values()) or all(truncatedDict.values()) or (step_count * 5 >= 3600)
            
        print(f"\n✔ Visual MARL evaluation finished cleanly! Total Evaluation Grid Reward: {total_eval_reward:.2f}")

    except Exception as e:
        print(f"\n[EVALUATION INTERRUPTED] Loop stopped due to: {e}")
    finally:
        if 'env' in locals() and hasattr(env, 'close'):
            env.close()      

if __name__ == "__main__":
    main()
