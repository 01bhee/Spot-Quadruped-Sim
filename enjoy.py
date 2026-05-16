import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from spot_env import SpotEnv
import time
import numpy as np
import os

_HERE       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(_HERE, "mujoco_menagerie", "boston_dynamics_spot", "scene.xml")
SAVED_MODEL = os.path.join(_HERE, "models", "scratch_v2.zip")
NUM_EPISODES = 10  # Evaluate over 10 rounds

def evaluate_brain():
    # 1. Paths
    brain_name = os.path.basename(SAVED_MODEL).replace(".zip", "")
    model_dir = os.path.dirname(SAVED_MODEL)
    vec_norm_path = os.path.join(model_dir, f"{brain_name}_vecnormalize.pkl")
    
    # 2. Setup Environment
    raw_env = SpotEnv(MODEL_PATH, render_mode="human")
    
    # Wrap in DummyVecEnv
    env = DummyVecEnv([lambda: raw_env])
    
    # 3. Load Normalization Stats (CRITICAL or agent goes blind)
    if os.path.exists(vec_norm_path):
        loaded_norm = VecNormalize.load(vec_norm_path, env)
        # Guard against stale stats from an old obs-space dimension (e.g. 35-dim or 51-dim).
        # A silent mismatch corrupts every normalised observation the policy receives.
        saved_dim    = loaded_norm.obs_rms.mean.shape[0]
        expected_dim = env.observation_space.shape[0]
        if saved_dim != expected_dim:
            print(f"WARNING: VecNormalize stats are {saved_dim}-dim but env is "
                  f"{expected_dim}-dim — discarding stale stats, running unnormalised.")
        else:
            env = loaded_norm
            env.training = False
            env.norm_reward = False
            print(f"Loaded observation normalization stats ({saved_dim}-dim).")
    else:
        print("WARNING: No VecNormalize stats found! Agent might perform poorly.")

    try:
        model = PPO.load(SAVED_MODEL, env=env)
        print(f"--- Evaluating {SAVED_MODEL} ---")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    all_rewards = []
    
    for episode in range(NUM_EPISODES):
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        print(f"Starting Round {episode + 1}...")

        # We keep the 5000 step limit here so we can compare rounds fairly
        while not done and steps < 5000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            raw_env.render()
            time.sleep(0.005) # Faster speed for evaluation
            
            episode_reward += reward[0]
            steps += 1

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