import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy  # <-- NEW IMPORT
from spot_env import SpotEnv
import os
import signal
import sys
import shutil

# 1. Configuration and Paths
MODEL_PATH = r"D:\Users\mimim\Documents\FYP\mujoco_menagerie\boston_dynamics_spot\scene.xml"
LOG_DIR = "./logs/"
MODEL_DIR = "./models/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================================================================
# ---> CHANGE THIS SINGLE LINE WHENEVER YOU WANT TO START A NEW BRAIN <---
BRAIN_NAME = "best_modelZ" 
# =========================================================================

BRAIN_FILE_PATH = os.path.join(MODEL_DIR, BRAIN_NAME)

def train():
    env = SpotEnv(MODEL_PATH, render_mode=None)
    eval_env = SpotEnv(MODEL_PATH, render_mode=None)

    # 2. Smart Brain Loading Logic
    print(f"Checking for existing brain: {BRAIN_NAME}.zip...")
    
    if os.path.exists(BRAIN_FILE_PATH + ".zip"):
        print(f"Found it! Loading existing weights for '{BRAIN_NAME}'...")
        model = PPO.load(BRAIN_FILE_PATH, env=env)
    else:
        print(f"Could not find '{BRAIN_NAME}.zip'. Creating a FRESH brain...")
        model = PPO("MlpPolicy", env=env, learning_rate=2.5e-4, n_steps=2048, 
                    batch_size=128, n_epochs=15, gamma=0.99, gae_lambda=0.95, 
                    clip_range=0.15, verbose=1, tensorboard_log=LOG_DIR)

    # 3. The Evaluation Callback (MOVED UP so the handler can read its scores!)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR, 
        log_path=LOG_DIR,
        eval_freq=10000,       
        n_eval_episodes=5,     
        deterministic=True,    
        render=False
    )

    # 4. INTERCEPT THE VS CODE STOP BUTTON WITH FINAL EXAM
    def emergency_save_handler(signum, frame):
        print("\n\n" + "="*50)
        print("TERMINATION SIGNAL DETECTED: Running final exam before shutting down...")
        
        # Force a strict test on the brain right now
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5, deterministic=True)
        print(f"\nFinal Test Score: {mean_reward:.2f}")
        
        # SB3 defaults to -inf if no eval has happened yet, we handle that cleanly here
        best_reward = eval_callback.best_mean_reward if eval_callback.best_mean_reward != -float('inf') else -float('inf')
        
        if best_reward != -float('inf'):
            print(f"Previous All-Time High: {best_reward:.2f}")
        else:
            print("Previous All-Time High: None (Stopping before first eval block)")

        # The Quality Check
        if mean_reward > best_reward:
            print(f"New High Score! Overwriting '{BRAIN_NAME}.zip' with these weights...")
            model.save(BRAIN_FILE_PATH)
        else:
            print(f"Score is worse. Discarding current weights and protecting the old '{BRAIN_NAME}.zip'.")
            
        print("="*50 + "\n")
        sys.exit(0) # Now we let Python safely exit

    # Wire the interceptor to both Ctrl+C (SIGINT) and the VS Code Stop button (SIGTERM)
    signal.signal(signal.SIGINT, emergency_save_handler)
    signal.signal(signal.SIGTERM, emergency_save_handler)

    # 5. Run the Training
    print(f"\nStarting Training for {BRAIN_NAME}...")
    print(">>> Press the VS Code STOP button at any time to safely save and exit. <<<\n")
    
    model.learn(
        total_timesteps=150000, 
        tb_log_name=f"PPO_{BRAIN_NAME}",
        callback=eval_callback  
    )
   # If the script naturally hits 300k steps, run the same check!
    print("\nTraining fully complete! Running one last check...")
    final_mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=5, deterministic=True)
    
    if final_mean_reward > eval_callback.best_mean_reward:
        model.save(BRAIN_FILE_PATH)
        print(f"New High Score! Saved peak final weights as '{BRAIN_NAME}.zip' in {MODEL_DIR}")
    else:
        print(f"Final weights were degraded. Throwing them in the trash!")
        
        # Grab the champion saved by the EvalCallback and rename it to your custom BRAIN_NAME
        champion_path = os.path.join(MODEL_DIR, "best_model.zip")
        if os.path.exists(champion_path):
            shutil.copy(champion_path, BRAIN_FILE_PATH + ".zip")
            print(f"Restored the all-time champion and safely saved it as '{BRAIN_NAME}.zip'!")

if __name__ == "__main__":
    train()