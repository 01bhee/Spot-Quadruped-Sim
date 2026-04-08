import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from spot_env import SpotEnv
import os
import signal
import sys
import shutil
import torch

class RewardComponentLogger(BaseCallback):
    """
    Reads per-component reward values from the step() info dict and logs
    their rollout-averaged values to TensorBoard under the 'rewards/' prefix.
    Lets you see exactly which terms are driving or limiting the agent.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._sums = {}
        self._counts = 0

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            for key, value in info.items():
                if isinstance(value, (int, float)):
                    self._sums[key] = self._sums.get(key, 0.0) + float(value)
            self._counts += 1
        return True

    def _on_rollout_end(self) -> None:
        if self._counts > 0:
            for key, total in self._sums.items():
                self.logger.record(f'rewards/{key}', total / self._counts)
        self._sums = {}
        self._counts = 0

# 1. Configuration and Paths
MODEL_PATH = r"D:\Users\mimim\Documents\FYP\mujoco_menagerie\boston_dynamics_spot\scene.xml"
LOG_DIR = "./logs/"
MODEL_DIR = "./models/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================================================================
# ---> CHANGE THIS SINGLE LINE WHENEVER YOU WANT TO START A NEW BRAIN <---
BRAIN_NAME = "model1B"
# =========================================================================
# Warm-start: when BRAIN_NAME has no saved weights yet, inherit from this
# brain instead of starting from random. The log_std is reset afterward
# so the policy explores the new reward landscape rather than rigidly
# repeating the old behaviour. Set to None to always start fresh.
WARM_START_FROM = None   # <-- weights source, or None (None = fresh random weights)
# If the warm-start brain has no vecnormalize, borrow stats from this brain instead.
# The obs space is always the same 35 dims, so stats transfer cleanly.
WARM_NORM_FROM  = None   # <-- norm stats source (can be same as WARM_START_FROM)
# Optional: override log_std on the next run to boost exploration.
# std = exp(log_std), so 0.0 → std=1.0, -0.1 → std≈0.9, -0.3 → std≈0.74
# Set to a float to reset, then set back to None to resume normally.
RESET_LOG_STD   = 0.0   # force std back down from ~1.6 to ~0.74; set None once stable
# =========================================================================
BRAIN_FILE_PATH = os.path.join(MODEL_DIR, BRAIN_NAME)

# Save the normalization statistics alongside the brain
VEC_NORM_PATH = os.path.join(MODEL_DIR, f"{BRAIN_NAME}_vecnormalize.pkl")

NUM_ENVS = 3  

def train():
    # 2. Environment Setup with Wrappers
    def make_env():
        return Monitor(SpotEnv(MODEL_PATH, render_mode=None))
        
    env = make_vec_env(
        make_env,
        n_envs=NUM_ENVS,
        vec_env_cls=SubprocVecEnv
    )
    
    # Create the evaluation environment and wrap it for SB3 compatibility
    eval_env_unwrapped = Monitor(SpotEnv(MODEL_PATH, render_mode=None))
    eval_env = DummyVecEnv([lambda: eval_env_unwrapped])
    
    # 3. Brain Loading & Normalization Logic
    print(f"Checking for existing brain: {BRAIN_NAME}.zip...")
    
    # We check for BOTH the model weights and the normalization stats
    if os.path.exists(BRAIN_FILE_PATH + ".zip") and os.path.exists(VEC_NORM_PATH):
        print(f"Found it! Loading existing weights and normalization stats for '{BRAIN_NAME}'...")
        env = VecNormalize.load(VEC_NORM_PATH, env)
        env.training = True # Ensure it continues updating stats during training
        env.norm_reward = True
        model = PPO.load(BRAIN_FILE_PATH, env=env)
        # Override ent_coef: it's baked into the saved model as 0.01 but that
        # value keeps pushing std upward during fine-tuning. 0.003 keeps just
        # enough entropy bonus to prevent premature collapse without inflating std.
        model.ent_coef = 0.003
        if RESET_LOG_STD is not None:
            model.policy.log_std.data.fill_(RESET_LOG_STD)
            print(f"  log_std reset to {RESET_LOG_STD} (std ≈ {torch.exp(torch.tensor(RESET_LOG_STD)).item():.3f})")

    elif WARM_START_FROM and os.path.exists(os.path.join(MODEL_DIR, WARM_START_FROM + ".zip")):
        # Inherit weights from a previous brain instead of random init.
        warm_model_path = os.path.join(MODEL_DIR, WARM_START_FROM)
        warm_norm_path  = os.path.join(MODEL_DIR, f"{WARM_START_FROM}_vecnormalize.pkl")
        print(f"No '{BRAIN_NAME}' found. Warm-starting weights from '{WARM_START_FROM}'...")
        # Look for norm stats: prefer the warm-start brain's own file,
        # then fall back to WARM_NORM_FROM, then create fresh.
        norm_source = WARM_START_FROM
        warm_norm_path = os.path.join(MODEL_DIR, f"{norm_source}_vecnormalize.pkl")
        if not os.path.exists(warm_norm_path) and WARM_NORM_FROM:
            norm_source = WARM_NORM_FROM
            warm_norm_path = os.path.join(MODEL_DIR, f"{norm_source}_vecnormalize.pkl")
        if os.path.exists(warm_norm_path):
            print(f"  Loading norm stats from '{norm_source}'...")
            env = VecNormalize.load(warm_norm_path, env)
        else:
            print(f"  No norm stats found — creating fresh VecNormalize.")
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
        env.training = True
        env.norm_reward = True
        model = PPO.load(warm_model_path, env=env)
        model.ent_coef = 0.003  # same override — don't let inherited ent_coef inflate std
        # Reset log_std so the policy explores the new reward landscape.
        # Loading from a converged brain inherits a collapsed std — without
        # this reset the policy is near-deterministic and won't adapt.
        # -0.1 → std ≈ 0.90, matching test_model2's final std and preserving
        # the exploration level the brain had already converged to.
        # Formula: std = exp(log_std), so for std=0.9: log_std = ln(0.9) ≈ -0.105
        model.policy.log_std.data.fill_(0.0)
        print(f"  log_std reset to 0.0 (std ≈ 1.0) to allow exploration.")
        print(f"  Will save new weights as '{BRAIN_NAME}'.")

    else:
        print(f"Could not find '{BRAIN_NAME}.zip' or stats. Creating a FRESH brain...")
        # Initialize the normalizer for a fresh brain
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
        # ent_coef=0.01 adds a small entropy bonus to keep exploration alive
        model = PPO("MlpPolicy", env=env, policy_kwargs=dict(net_arch=[dict(pi=[256, 256])]), learning_rate=2.5e-4, n_steps=4096,
                    batch_size=256, n_epochs=15, gamma=0.99, gae_lambda=0.95,
                    clip_range=0.15, ent_coef=0.003, verbose=1, tensorboard_log=LOG_DIR)

    # CRITICAL: Sync the evaluation environment's normalization stats with the training environment
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)
    eval_env.obs_rms = env.obs_rms # Share the exact same mathematical scale
    eval_env.training = False # Freeze the stats so evaluation doesn't alter the training baseline

    # 4. Evaluation Callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR, 
        log_path=LOG_DIR,
        eval_freq=25000,
        n_eval_episodes=5, 
        deterministic=True,
        render=False
    )

    # 5. Intercept the VS Code Stop Button (SIGTERM)
    def emergency_save_handler(signum, frame):
        print("\n\n" + "="*50)
        print("TERMINATION SIGNAL DETECTED: Saving current progress before shutting down...")
        
        # IMMEDIATELY save the normalization stats so we don't lose the scale!
        env.save(VEC_NORM_PATH)
        print(f"Saved observation statistics to {VEC_NORM_PATH}")

        # Save the current model explicitly (no champion logic)
        model.save(BRAIN_FILE_PATH)
        print(f"Saved current weights as '{BRAIN_NAME}.zip' in {MODEL_DIR}")
        print("="*50 + "\n")
        
        # Close the parallel environments gracefully 
        print("Safely spinning down MuJoCo environments...")
        env.close()
        eval_env.close()
        
        sys.exit(0)

    signal.signal(signal.SIGINT, emergency_save_handler)
    signal.signal(signal.SIGTERM, emergency_save_handler)

    # 6. Run Training
    print(f"\nStarting Training for {BRAIN_NAME}...")
    print(">>> Press the VS Code STOP button at any time to safely save and exit. <<<\n")

    model.learn(
        total_timesteps=500_000,
        tb_log_name=f"PPO_{BRAIN_NAME}",
        callback=[eval_callback, RewardComponentLogger()]
    )
    
    # 7. Final Wrap Up (If it naturally hits the step limit)
    print("\nTraining fully complete!")
    
    env.save(VEC_NORM_PATH) # Save the final stats
    model.save(BRAIN_FILE_PATH)
    print(f"Saved final weights as '{BRAIN_NAME}.zip' in {MODEL_DIR}")

if __name__ == "__main__":
    train()
