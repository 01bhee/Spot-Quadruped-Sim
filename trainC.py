import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from spot_sar_env import SpotSAREnv

BASE_BRAIN      = "scratch_v2"
BASE_MODEL_PATH = f"models/{BASE_BRAIN}.zip"
BASE_NORM_PATH  = f"models/{BASE_BRAIN}_vecnormalize.pkl"

def make_env(scene_name):
    def _init():
        return SpotSAREnv(scene=scene_name)
    return _init

def _load_norm(pkl_path, env):
    loaded = VecNormalize.load(pkl_path, env)
    saved_dim    = loaded.obs_rms.mean.shape[0]
    expected_dim = env.observation_space.shape[0]
    if saved_dim != expected_dim:
        raise ValueError(
            f"VecNormalize dim mismatch: saved={saved_dim}, env={expected_dim}. "
            f"Make sure {pkl_path} was produced by a 52-dim training run."
        )
    return loaded

def main():
    os.makedirs("models/curriculum", exist_ok=True)
    os.makedirs("logs/curriculum", exist_ok=True)
    os.makedirs("models/curriculum_checkpoints", exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path="./models/curriculum_checkpoints/",
        name_prefix="spot_cl",
    )

    # Stage 1: rough terrain
    print("\n--- Stage 1: Rough Terrain ---")
    env_stage1 = DummyVecEnv([
        make_env("rough"), make_env("rough"), make_env("rough")
    ])
    env_stage1 = _load_norm(BASE_NORM_PATH, env_stage1)
    env_stage1.training    = True
    env_stage1.norm_reward = True

    model = PPO.load(BASE_MODEL_PATH, env=env_stage1)
    model.ent_coef = 0.003
    with torch.no_grad():
        model.policy.log_std.fill_(-0.5)
    print("  log_std reset to -0.5")

    model.learn(total_timesteps=1_000_000, reset_num_timesteps=False,
                tb_log_name="PPO_curriculum_stage1",
                callback=checkpoint_callback)

    model.save("models/curriculum/spot_stage1_rough_v2")
    env_stage1.save("models/curriculum/stage1_v2_vecnormalize.pkl")
    env_stage1.close()
    print("Stage 1 complete. Saved to: models/curriculum/spot_stage1_rough_v2.zip")
    input(">>> Press Enter to start Stage 2, or Ctrl+C to stop here. <<<\n")

    # Stage 2: friction adaptation — flood + snow
    print("\n--- Stage 2: Flood + Snow ---")
    env_stage2 = DummyVecEnv([
        make_env("flood"), make_env("flood"), make_env("snow")
    ])
    env_stage2 = _load_norm("models/curriculum/stage1_v2_vecnormalize.pkl", env_stage2)
    env_stage2.training    = True
    env_stage2.norm_reward = True

    model.set_env(env_stage2)
    model.ent_coef = 0.001
    with torch.no_grad():
        model.policy.log_std.fill_(-0.5)
    print("  log_std reset to -0.5")

    model.learn(total_timesteps=500_000, reset_num_timesteps=False,
                tb_log_name="PPO_curriculum_stage2",
                callback=checkpoint_callback)

    model.save("models/curriculum/spot_stage2_friction_v2")
    env_stage2.save("models/curriculum/stage2_v2_vecnormalize.pkl")
    env_stage2.close()
    print("Stage 2 complete. Saved to: models/curriculum/spot_stage2_friction_v2.zip")
    input(">>> Press Enter to start Stage 3, or Ctrl+C to stop here. <<<\n")

    # Stage 3: all terrains including stairs
    print("\n--- Stage 3: All Terrains ---")
    env_stage3 = DummyVecEnv([
        make_env("rough"), make_env("flood"),
        make_env("snow"),  make_env("stairs"),
    ])
    env_stage3 = _load_norm("models/curriculum/stage2_v2_vecnormalize.pkl", env_stage3)
    env_stage3.training    = True
    env_stage3.norm_reward = True

    model.set_env(env_stage3)
    model.ent_coef = 0.001
    with torch.no_grad():
        model.policy.log_std.fill_(-0.5)
    print("  log_std reset to -0.5")

    model.learn(total_timesteps=500_000, reset_num_timesteps=False,
                tb_log_name="PPO_curriculum_stage3",
                callback=checkpoint_callback)

    model.save("models/curriculum/spot_final_policy_v2")
    env_stage3.save("models/curriculum/final_v2_vecnormalize.pkl")
    env_stage3.close()
    print("Stage 3 complete. Final curriculum model saved!")

if __name__ == "__main__":
    main()
