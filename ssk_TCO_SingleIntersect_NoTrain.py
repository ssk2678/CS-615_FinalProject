import os
import sys
from sumo_rl import SumoEnvironment

# KEEP THIS: Crucial for Linux virtual environment stability
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.insert(0, tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    net_path = os.path.join(base_dir, "dataset", "nets", "single-intersection", "single-intersection.net.xml")
    
    # FIXED: Using the verified filename from your folder
    route_path = os.path.join(base_dir, "dataset", "nets", "single-intersection", "single-intersection.rou.xml")

    print("Initializing SUMO-RL Environment...")
    
    # ENABLED: use_gui is now True
    env = SumoEnvironment(
        net_file=net_path,
        route_file=route_path,
        use_gui=True,          
        num_seconds=3600,       
        delta_time=5,           
        min_green=5,
        single_agent=True       
    )

    try:
        obs, info = env.reset()
        print("Success! Environment reset cleanly. Watch the GUI window!")

        done = False
        truncated = False
        step = 0
        
        # This loop runs until the 3600 seconds of simulation time are completely up
        while not (done or truncated):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            step += 1
            if step % 20 == 0:
                print(f"Step {step} | Current Step Reward: {reward:.2f}")

    except Exception as e:
        print(f"\n[CRASH DETECTED] Script failed due to: {e}")

    finally:
        print("\nTraffic scenario finished!")
        input("Press Enter in this terminal to exit...")
        env.close()

    
    # # Run a quick 20-step visual loop test
    # for i in range(20):
    #     action = env.action_space.sample()
    #     obs, reward, done, truncated, info = env.step(action)
    #     print(f"Step {i+1} | Reward: {reward:.2f}")
    #     if done or truncated:
    #         break

    # env.close()
    # print("Simulation closed safely.")

if __name__ == "__main__":
    main()
