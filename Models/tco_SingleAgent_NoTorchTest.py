import os
import sys
import argparse
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
    parser = argparse.ArgumentParser(description="Test a single-agent NumPy DQN.")
    parser.add_argument("--zhang", action="store_true", help="Use partial observations and the Zhang reward.")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, ".."))
    
    net_path = os.path.join(base_dir, "dataset", "nets", "single-intersection", "single-intersection.net.xml")
    route_path = os.path.join(base_dir, "dataset", "nets", "single-intersection", "single-intersection.rou.xml")

    weights_path = "noTorch_zhang_scratch_weights.npz" if args.zhang else "noTorch_scratch_weights.npz"

    if not os.path.exists(weights_path):
        sys.exit(f"Missing weights file at {weights_path}! Please run the NoTorch training script first.")

    print("Initializing Visual SUMO GUI Environment (NumPy)...")
    env = SumoEnvironment(
        net_file=net_path, route_file=route_path, use_gui=True, 
        num_seconds=3600, max_depart_delay=1, delta_time=5, single_agent=True,
        reward_fn=zhang_reward if args.zhang else "diff-waiting-time"
    )
    
    # Direct single-agent space configurations
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Reconstruct the custom network configuration
    agent = Agent(state_dim, action_dim)
    
    # Load raw matrix weights directly into custom neural layers object properties
    data = np.load(weights_path)
    agent.policy_net.layer1.W = data['W1']
    agent.policy_net.layer1.b = data['b1']
    agent.policy_net.layer2.W = data['W2']
    agent.policy_net.layer2.b = data['b2']
    
    # Force epsilon to 0.0 to guarantee pure evaluation exploitation 
    agent.epsilon = 0.0
    print("💾 Successfully loaded trained custom NumPy matrix weights!")

    try:
        print("\n▶ Press the green 'Play' button in the SUMO GUI window to start traffic flow...")
        
        # Correctly unpack Gymnasium single-agent tuple
        obs, info = env.reset()
        if args.zhang:
            obs = mask_observation(obs, action_dim)
        done = False        
        total_eval_reward = 0.0
        
        while not done:
            # Avoid multi-agent dictionary unpacking; feed array straight to custom matrix math
            action = agent.select_action(obs)
            
            # Process direct single-agent trajectory signals
            next_obs, reward, terminated, truncated, info = env.step(action)
            if args.zhang:
                next_obs = mask_observation(next_obs, action_dim)
            total_eval_reward += reward
            obs = next_obs
            
            # Aligned step termination checker
            done = terminated or truncated or (env.sim_step >= 3600)
            
        print(f"\n✔ Custom NumPy evaluation finished cleanly! Total Evaluation Reward: {total_eval_reward:.2f}")

    except Exception as e:
        print(f"\n[EVALUATION INTERRUPTED] Loop stopped due to: {e}")
    finally:
        if 'env' in locals() and hasattr(env, 'sumo') and env.sumo is not None:
            env.close()      

if __name__ == "__main__":
    main()
