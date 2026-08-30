import argparse
import os
import sys

import numpy as np
from sumo_rl import SumoEnvironment

from DeepQNAgent import Agent
from multiagent_zhang import mask_observation, zhang_reward


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


def evaluate(env, agent, action_dim, use_zhang, seed, label):
    # Give both evaluations the same Zhang detection/masking random sequence.
    np.random.seed(seed)
    observation, _ = env.reset()
    if use_zhang:
        observation = mask_observation(observation, action_dim)

    print(f"\n=== {label} ===")
    print("Press the green 'Play' button in SUMO if the simulation is paused.")

    total_reward = 0.0
    done = False
    while not done:
        action = agent.select_action(observation)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        if use_zhang:
            next_observation = mask_observation(next_observation, action_dim)

        total_reward += reward
        observation = next_observation
        done = terminated or truncated or env.sim_step >= 3600

    return float(total_reward)


def print_summary(before_reward, after_reward):
    delta = after_reward - before_reward
    result = "IMPROVED" if delta > 0.005 else "WORSE" if delta < -0.005 else "SAME"
    print("\n=== TEST SUMMARY ===")
    print(f"Before (Random) Reward: {before_reward:.2f}")
    print(f"After (Trained) Reward: {after_reward:.2f}")
    print(f"Delta (After - Before): {delta:+.2f}")
    print(f"Result: {result}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare a random and trained single-agent NumPy DQN."
    )
    parser.add_argument(
        "--zhang", action="store_true", help="Use partial observations and the Zhang reward."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="SUMO, NumPy, and model seed (default: 42)."
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, ".."))
    net_path = os.path.join(
        base_dir, "dataset", "nets", "single-intersection", "single-intersection.net.xml"
    )
    route_path = os.path.join(
        base_dir, "dataset", "nets", "single-intersection", "single-intersection.rou.xml"
    )
    weights_path = os.path.join(
        current_dir,
        "noTorch_zhang_scratch_weights.npz" if args.zhang else "noTorch_scratch_weights.npz",
    )

    print("Initializing single-agent SUMO evaluation environment (NumPy)...")
    env = SumoEnvironment(
        net_file=net_path,
        route_file=route_path,
        use_gui=True,
        num_seconds=3600,
        max_depart_delay=1,
        delta_time=5,
        single_agent=True,
        reward_fn=zhang_reward if args.zhang else "diff-waiting-time",
        sumo_seed=args.seed,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = Agent(state_dim, action_dim)
    agent.epsilon = 0.0

    try:
        before_reward = evaluate(
            env, agent, action_dim, args.zhang, args.seed, "BEFORE: RANDOM NETWORK"
        )
        print(f"Random network reward: {before_reward:.2f}")

        if not os.path.exists(weights_path):
            sys.exit(
                f"Missing trained weights at {weights_path}. Run the matching training script first."
            )

        with np.load(weights_path) as data:
            agent.policy_net.layer1.W = data["W1"]
            agent.policy_net.layer1.b = data["b1"]
            agent.policy_net.layer2.W = data["W2"]
            agent.policy_net.layer2.b = data["b2"]
        print(f"Loaded trained weights from {weights_path}")

        after_reward = evaluate(
            env, agent, action_dim, args.zhang, args.seed, "AFTER: TRAINED NETWORK"
        )
        print(f"Trained network reward: {after_reward:.2f}")
        print_summary(before_reward, after_reward)
    finally:
        if hasattr(env, "sumo") and env.sumo is not None:
            env.close()


if __name__ == "__main__":
    main()
