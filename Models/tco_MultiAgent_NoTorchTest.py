import os
import sys
import numpy as np

# =====================================================================
# 🔥 CRITICAL BUGFIX: Resolve the PettingZoo agent_selector module conflict
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

import sumo_rl  # FIXED: Required for parallel_env multi-agent execution wrapper

# Clean local module imports from the layer library layout
from DeepQNetwork import DQN

# KEEP THIS: Crucial for Linux virtual environment stability
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.insert(0, tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, ".."))
    
    net_path = os.path.join(base_dir, "dataset", "nets", "3x3grid", "3x3Grid2lanes.net.xml")
    route_path = os.path.join(base_dir, "dataset", "nets", "3x3grid", "routes14000.rou.xml")
    
    # FIXED: Direct targeting to the correct multi-agent 2x2/3x3 weights archive
    weights_path = os.path.join(current_dir, "noTorch_marl_3x3_scratch_weights.npz")

    # Protective file path assertions
    if not os.path.exists(weights_path):
        sys.exit(f"❌ Error: Missing trained multi-agent matrix arrays at {weights_path}.\nRun the main NumPy MARL training pipeline first!")

    print("Initializing Multi-Agent Evaluation Environment...")
    # FIXED: Swapped out SumoEnvironment for parallel_env wrapper interface
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

    print("Loading raw NumPy parameter block weights...")
    loaded_data = np.load(weights_path)
    
    # FIXED: DYNAMIC MAPPING - Reconstruct independent networks for each individual intersection
    policy_networks = {}
    for tl_id in env.possible_agents:
        state_dim = env.observation_space(tl_id).shape[0]
        action_dim = env.action_space(tl_id).n
        
        # Instantiate 2-layer network framework for this specific junction node
        net = DQN(state_dim, action_dim)
        
        # Pull specific weight sub-arrays using tagged prefix string keys
        net.layer1.W = loaded_data[f"{tl_id}_W1"]
        net.layer1.b = loaded_data[f"{tl_id}_b1"]
        net.layer2.W = loaded_data[f"{tl_id}_W2"]
        net.layer2.b = loaded_data[f"{tl_id}_b2"]
        
        policy_networks[tl_id] = net
        
    print("💾 Successfully restored all layer nodes! Commencing test verification curve...")

    try:
        # PettingZoo parallel environment reset structure returns (obs, info) tuple
        obs_dict, _ = env.reset()
        done = False        
        total_test_reward = 0.0
        step_count = 0
        
        while not done:
            action_dict = {}
            for tl_id, obs in obs_dict.items():
                # FIXED: Consult the SPECIFIC network dictionary entry mapped to this active traffic light key
                active_net = policy_networks[tl_id]
                q_values = active_net.forward(obs.reshape(1, -1))
                action_dict[tl_id] = np.argmax(q_values)
            
            # FIXED: Step execution now returns 5 dictionary objects
            next_obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = env.step(action_dict)
            
            total_test_reward += sum(reward_dict.values())
            obs_dict = next_obs_dict
            step_count += 1
            
            # FIXED: Bypasses library version wrapper bugs entirely using step calculations
            # 720 steps * 5 seconds (delta_time) = 3600 total simulation seconds
            done = all(terminated_dict.values()) or all(truncated_dict.values()) or (step_count * 5 >= 3600)
            
            if step_count % 100 == 0:
                print(f"Test Step {step_count} | Accumulated Step Reward: {sum(reward_dict.values()):.2f}")
            
        print(f"\n✔ NumPy Test Evaluation finished cleanly!")
        print(f"➡ Final Deterministic Test Score: {total_test_reward:.2f}")

    except Exception as e:
        print(f"\n[TEST MATRIX CRASHED] Stopped mid-execution window due to: {e}")
    finally:
        if 'env' in locals():
            env.close()      

if __name__ == "__main__":
    main()
