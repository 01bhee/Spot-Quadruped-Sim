import gymnasium as gym
from stable_baselines3 import PPO
from spot_env import SpotEnv
import time
import numpy as np

MODEL_PATH = r"D:\Users\mimim\Documents\FYP\mujoco_menagerie\boston_dynamics_spot\scene.xml"
SAVED_MODEL = r"D:\Users\mimim\Documents\FYP\models\best_modelZ.zip"
NUM_EPISODES = 10  # Evaluate over 10 rounds

def evaluate_brain():
    env = SpotEnv(MODEL_PATH, render_mode="human")
    
    try:
        model = PPO.load(SAVED_MODEL)
        print(f"--- Evaluating {SAVED_MODEL} ---")
    except:
        print(f"Error: {SAVED_MODEL} not found.")
        return

    all_rewards = []
    
    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        print(f"Starting Round {episode + 1}...")

        # We keep the 5000 step limit here so we can compare rounds fairly
        while not done and steps < 5000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            env.render()
            time.sleep(0.005) # Faster speed for evaluation
            
            episode_reward += reward
            steps += 1
            done = terminated or truncated

        all_rewards.append(episode_reward)
        print(f"Round {episode + 1} finished. Reward: {episode_reward:.2f} | Steps: {steps}")
        time.sleep(0.5)

    # --- FINAL REPORT ---
    print("\n" + "="*30)
    print(f"EVALUATION COMPLETE FOR {SAVED_MODEL}")
    print(f"Average Reward: {np.mean(all_rewards):.2f}")
    print(f"Best Round: {np.max(all_rewards):.2f}")
    print("="*30)
    
    env.close()

if __name__ == "__main__":
    evaluate_brain()