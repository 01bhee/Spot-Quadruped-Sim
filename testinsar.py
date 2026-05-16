import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from spot_sar_env import SpotSAREnv

# Set STAGE (1/2/3) and SCENE before running
STAGE = 1
SCENE = "rough"

_STAGE_FILES = {
    1: ("spot_stage1_rough_v2",    "stage1_v2_vecnormalize.pkl"),
    2: ("spot_stage2_friction_v2", "stage2_v2_vecnormalize.pkl"),
    3: ("spot_final_policy_v2",    "final_v2_vecnormalize.pkl"),
}
MODEL_DIR       = "./models/curriculum"
_brain, _norm   = _STAGE_FILES[STAGE]
BRAIN_FILE_PATH = os.path.join(MODEL_DIR, _brain)
VEC_NORM_PATH   = os.path.join(MODEL_DIR, _norm)

NUM_EPISODES             = 5
DETERMINISTIC            = True
STALL_THRESHOLD_STEPS    = 200
STALL_THRESHOLD_DISTANCE = 0.1


def test_in_sar():
    base_env  = SpotSAREnv(scene=SCENE, render_mode="human")
    monitored = Monitor(base_env)
    vec_env   = DummyVecEnv([lambda: monitored])

    print(f"Loading normalisation stats: {VEC_NORM_PATH}")
    vec_env = VecNormalize.load(VEC_NORM_PATH, vec_env)

    saved_dim    = vec_env.obs_rms.mean.shape[0]
    expected_dim = base_env.observation_space.shape[0]
    if saved_dim != expected_dim:
        raise ValueError(
            f"VecNormalize dim mismatch: stats are {saved_dim}-dim "
            f"but env is {expected_dim}-dim."
        )

    vec_env.training    = False
    vec_env.norm_reward = False

    print(f"Loading policy: {BRAIN_FILE_PATH}.zip")
    model = PPO.load(BRAIN_FILE_PATH, env=vec_env)

    print("=" * 60)
    print(f"Stage {STAGE} policy — scene: '{SCENE}'")
    print("Press Ctrl+C to stop early.")
    print("=" * 60)

    for ep in range(NUM_EPISODES):
        obs       = vec_env.reset()
        done      = False
        steps     = 0
        total_rew = 0.0
        start_pos = base_env.data.qpos[:3].copy()
        min_height = float("inf")
        max_x      = start_pos[0]

        last_progress_pos    = start_pos[:2].copy()
        steps_since_progress = 0
        stalled = False

        print(f"\n─── Episode {ep + 1}/{NUM_EPISODES} ───")
        print(f"  Start: x={start_pos[0]:.2f}  y={start_pos[1]:.2f}  z={start_pos[2]:.2f}")

        last_render_time = time.time()

        while not done:
            action, _ = model.predict(obs, deterministic=DETERMINISTIC)
            obs, reward, dones, infos = vec_env.step(action)

            steps     += 1
            total_rew += float(reward[0])
            cur = base_env.data.qpos[:3]
            min_height = min(min_height, float(cur[2]))
            max_x      = max(max_x, float(cur[0]))

            moved = float(np.linalg.norm(cur[:2] - last_progress_pos))
            if moved > STALL_THRESHOLD_DISTANCE:
                last_progress_pos    = cur[:2].copy()
                steps_since_progress = 0
            else:
                steps_since_progress += 1

            if steps_since_progress >= STALL_THRESHOLD_STEPS:
                stalled = True
                break

            base_env.render()
            elapsed    = time.time() - last_render_time
            sleep_time = 0.02 - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_render_time = time.time()

            done = bool(dones[0])

        end         = base_env.data.qpos[:3].copy()
        x_travelled = float(end[0] - start_pos[0])

        print(f"  End:   x={end[0]:.2f}  y={end[1]:.2f}  z={end[2]:.2f}")
        print(f"  Steps: {steps}  |  X travelled: {x_travelled:.2f}m  |  Max X: {max_x:.2f}m")
        print(f"  Min body height: {min_height:.3f}m  |  Total reward: {total_rew:.1f}")

        if min_height < 0.30:
            print("  >>> Spot FELL")
        elif stalled:
            print(f"  >>> Spot STALLED at x={end[0]:.2f} — no progress for {STALL_THRESHOLD_STEPS} steps")
        elif steps >= base_env.max_steps - 1:
            print("  >>> Full episode survived!")

    print("\n" + "=" * 60)
    print("Test complete.")
    print("=" * 60)
    base_env.close()


if __name__ == "__main__":
    try:
        test_in_sar()
    except KeyboardInterrupt:
        print("\nStopped by user.")
