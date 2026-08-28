import os
import sys
import argparse
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

# Clean local module imports from the layer library layout
from DeepQNetwork import DQN
from multiagent_zhang import mask_observations, zhang_reward

# KEEP THIS: Crucial for Linux virtual environment stability
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.insert(0, tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


def main():
    parser = argparse.ArgumentParser(description="Test a NumPy multi-agent DQN.")
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
    mode_suffix = "_zhang" if args.zhang else ""
    weights_path = os.path.join(current_dir, f"noTorch_marl_{args.scenario}{mode_suffix}_scratch_weights.npz")

    # Protective file path assertions
    if not os.path.exists(weights_path):
        sys.exit(f"❌ Error: Missing trained multi-agent matrix arrays at {weights_path}.\nRun the main NumPy MARL training pipeline first!")

    print("Initializing Multi-Agent Evaluation Environment...")
    env = sumo_rl.parallel_env(
        net_file=net_path, 
        route_file=route_path, 
        use_gui=True, 
        num_seconds=3600, 
        max_depart_delay=1,
        delta_time=5,
        reward_fn=zhang_reward if args.zhang else "diff-waiting-time"
    )
    
    # Trigger a baseline environment initialization step to pull topology dimensions
    env.reset()

    print("Loading raw NumPy parameter block weights...")
    loaded_data = np.load(weights_path)
    
    # DYNAMIC MAPPING: Reconstruct independent networks for each individual intersection
    policy_networks = {}
    for tl_id in env.possible_agents:
        # Peel off the first index scalar to pass an integer rather than a tuple object
        shape_val = env.observation_space(tl_id).shape
        state_dim = shape_val[0] if isinstance(shape_val, tuple) else shape_val
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
        if args.zhang:
            obs_dict = mask_observations(obs_dict, env)
        done = False        
        total_test_reward = 0.0
        step_count = 0
        
        while not done:
            action_dict = {}
            for tl_id, obs in obs_dict.items():
                # Consult the SPECIFIC network dictionary entry mapped to this active traffic light key
                active_net = policy_networks[tl_id]
                q_values = active_net.forward(obs.reshape(1, -1))
                action_dict[tl_id] = np.argmax(q_values)
            
            # Step execution returns 5 dictionary objects
            next_obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = env.step(action_dict)
            if args.zhang:
                next_obs_dict = mask_observations(next_obs_dict, env)
            
            total_test_reward += sum(reward_dict.values())
            obs_dict = next_obs_dict
            step_count += 1
            
            # Synchronized termination statement tracking snake_case variables perfectly
            done = all(terminated_dict.values()) or all(truncated_dict.values()) or (step_count * 5 >= 3600)
            
            if step_count % 100 == 0:
                print(f"Test Step {step_count} | Accumulated Step Reward: {sum(reward_dict.values()):.2f}")
            
        print(f"\n✔ NumPy Test Evaluation finished cleanly!")
        print(f"➡ Final Deterministic Test Score: {total_test_reward:.2f}")

    except Exception as e:
        print(f"\n[TEST MATRIX CRASHED] Stopped mid-execution window due to: {e}")
    finally:
        if 'env' in locals() and hasattr(env, 'close'):
            env.close()      

if __name__ == "__main__":
    main()
