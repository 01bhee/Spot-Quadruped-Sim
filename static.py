import gymnasium as gym
from spot_env import SpotEnv
import time
import numpy as np

MODEL_PATH = r"D:\Users\mimim\Documents\FYP\mujoco_menagerie\boston_dynamics_spot\scene.xml"

def visualize_start():
    # render_mode="human" allows us to see the MuJoCo window
    env = SpotEnv(MODEL_PATH, render_mode="human")
    obs, _ = env.reset()
    
    print("Static View active. Spot is in his Home Pose.")
    print("Close the window or press Ctrl+C to stop.")

    try:
        while True:
            # We send a 'Zero' action (0.0 means 'stay exactly at home pose')
            # This allows you to check if the posture is stable before AI takes over
            zero_action = np.zeros(12) 
            env.step(zero_action)
            
            env.render()
            time.sleep(0.02) # Standard 50fps view
    except KeyboardInterrupt:
        print("Closing Viewer...")
    finally:
        env.close()

if __name__ == "__main__":
    visualize_start()