import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

try:
    import pettingzoo.utils.agent_selector as az_mod
    import pettingzoo.utils

    if hasattr(az_mod, "agent_selector"):
        pettingzoo.utils.agent_selector = az_mod.agent_selector
except ImportError:
    pass

import sumo_rl

from multiagent_zhang import mask_observations, zhang_reward


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        input_features = state_dim[0] if isinstance(state_dim, tuple) else state_dim
        self.fullyConnected = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.fullyConnected(x)


def evaluate(env, policy_networks, use_zhang, seed, label):
    # Give both evaluations the same Zhang detection/masking random sequence.
    np.random.seed(seed)
    observations, _ = env.reset()
    if use_zhang:
        observations = mask_observations(observations, env)

    print(f"\n=== {label} ===")
    print("Press the green 'Play' button in SUMO if the simulation is paused.")

    total_reward = 0.0
    step_count = 0
    done = False
    while not done:
        actions = {}
        for traffic_light_id, observation in observations.items():
            state = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                actions[traffic_light_id] = torch.argmax(
                    policy_networks[traffic_light_id](state)
                ).item()

        next_observations, rewards, terminated, truncated, _ = env.step(actions)
        if use_zhang:
            next_observations = mask_observations(next_observations, env)

        total_reward += sum(rewards.values())
        observations = next_observations
        step_count += 1
        done = (
            all(terminated.values())
            or all(truncated.values())
            or step_count * 5 >= 3600
        )

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
        description="Compare random and trained PyTorch multi-agent DQNs."
    )
    parser.add_argument("--scenario", choices=("2x2", "4x4"), required=True)
    parser.add_argument(
        "--zhang", action="store_true", help="Use partial observations and the Zhang reward."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="SUMO, NumPy, and model seed (default: 42)."
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, ".."))
    if args.scenario == "2x2":
        scenario_dir, net_file, route_file = "2x2grid", "2x2.net.xml", "2x2.rou.xml"
    else:
        scenario_dir, net_file, route_file = "4x4-Lucas", "4x4.net.xml", "4x4c1.rou.xml"

    net_path = os.path.join(base_dir, "dataset", "nets", scenario_dir, net_file)
    route_path = os.path.join(base_dir, "dataset", "nets", scenario_dir, route_file)
    mode_suffix = "_zhang" if args.zhang else ""
    weights_path = os.path.join(
        current_dir, f"marl_{args.scenario}{mode_suffix}_traffic_weights.pt"
    )

    print("Initializing multi-agent SUMO evaluation environment...")
    env = sumo_rl.parallel_env(
        net_file=net_path,
        route_file=route_path,
        use_gui=True,
        num_seconds=3600,
        max_depart_delay=1,
        delta_time=5,
        reward_fn=zhang_reward if args.zhang else "diff-waiting-time",
        sumo_seed=args.seed,
    )

    # SUMO-RL populates the traffic-light spaces after its first reset.
    env.reset()
    torch.manual_seed(args.seed)
    policy_networks = {}
    for traffic_light_id in env.possible_agents:
        state_dim = env.observation_space(traffic_light_id).shape
        action_dim = env.action_space(traffic_light_id).n
        network = DQNetwork(state_dim, action_dim)
        network.eval()
        policy_networks[traffic_light_id] = network

    try:
        before_reward = evaluate(
            env, policy_networks, args.zhang, args.seed, "BEFORE: RANDOM NETWORKS"
        )
        print(f"Random networks reward: {before_reward:.2f}")

        if not os.path.exists(weights_path):
            sys.exit(
                f"Missing trained weights at {weights_path}. Run the matching training script first."
            )

        saved_weights = torch.load(weights_path, map_location="cpu")
        missing_agents = set(policy_networks) - set(saved_weights)
        if missing_agents:
            missing = ", ".join(sorted(missing_agents))
            sys.exit(f"Trained checkpoint is missing traffic-light weights for: {missing}")

        for traffic_light_id, network in policy_networks.items():
            network.load_state_dict(saved_weights[traffic_light_id])
            network.eval()
        print(f"Loaded trained weights from {weights_path}")

        after_reward = evaluate(
            env, policy_networks, args.zhang, args.seed, "AFTER: TRAINED NETWORKS"
        )
        print(f"Trained networks reward: {after_reward:.2f}")
        print_summary(before_reward, after_reward)
    finally:
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main()
