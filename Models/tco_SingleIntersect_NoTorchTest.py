import os
import sys
import numpy as np
from sumo_rl import SumoEnvironment

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
    # 1. STEP UP PATHING FIX: Step out of Models/ to find the project root folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, ".."))
    
    net_path = os.path.join(base_dir, "dataset", "nets", "single-intersection", "single-intersection.net.xml")
    route_path = os.path.join(base_dir, "dataset", "nets", "single-intersection", "single-intersection.rou.xml")
    weights_path = os.path.join(current_dir, "noTorch_scratch_weights.npz") # Weights are inside Models/

    # Protective file path assertions
    if not os.path.exists(weights_path):
        sys.exit(f"❌ Error: Missing trained matrix arrays at {weights_path}.\nRun the main NumPy training pipeline first!")

    print("Initializing Multi-Agent Evaluation Environment...")
    # 2. HEADLESS PROTECTION: use_gui=False stabilizes the TraCI process inside WSL
    env = SumoEnvironment(
        net_file=net_path, 
        route_file=route_path, 
        use_gui=True, 
        num_seconds=3600, 
        max_depart_delay=1,
        delta_time=5, 
        single_agent=False
    )
    
    # 3. INTERFACE CONVERSION FIX: Functional mappings capture multi-agent shapes safely
    sample_tl = env.ts_ids[0]
    state_dim = env.observation_spaces(sample_tl).shape[0]
    action_dim = env.action_spaces(sample_tl).n

    # 4. INSTANTIATE AND DE-SERIALIZE MATRICES
    print("Loading raw NumPy parameter block weights...")
    policy_net = DQN(state_dim, action_dim)
    
    loaded_data = np.load(weights_path)
    policy_net.layer1.W = loaded_data['W1']
    policy_net.layer1.b = loaded_data['b1']
    policy_net.layer2.W = loaded_data['W2']
    policy_net.layer2.b = loaded_data['b2']
    print("💾 Successfully restored all layer layers! Commencing test verification curve...")

    try:
        obs_dict = env.reset()
        done = False        
        total_test_reward = 0.0
        step_count = 0
        
        while not done:
            action_dict = {}
            for tl_id, obs in obs_dict.items():
                # 5. PURE EXPLOITATION: EPSILON = 0. Always calculate the maximum mathematical Q-value
                q_values = policy_net.forward(obs.reshape(1, -1))
                action_dict[tl_id] = np.argmax(q_values)
            
            next_obs_dict, reward_dict, terminated_dict, info_dict = env.step(action_dict)
            total_test_reward += sum(reward_dict.values())
            obs_dict = next_obs_dict
            
            # Step safety ceiling mapping
            done = all(terminated_dict.values()) or (env.sim_step >= 3600)
            step_count += 1
            
            if step_count % 100 == 0:
                print(f"Test Step {step_count} | Accumulated Step Reward: {sum(reward_dict.values()):.2f}")
            
        print(f"\n✔ NumPy Test Evaluation finished cleanly!")
        print(f"➡ Final Deterministic Test Score: {total_test_reward:.2f}")

    except Exception as e:
        print(f"\n[TEST MATRIX CRASHED] Stopped mid-execution window due to: {e}")
    finally:
        if 'env' in locals() and hasattr(env, 'sumo') and env.sumo is not None:
            env.close()      

if __name__ == "__main__":
    main()
