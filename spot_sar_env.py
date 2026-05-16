import numpy as np
import os
import mujoco
from spot_env import SpotEnv, ENABLE_GAIT_SHAPING

SCENE_PATHS = {
    "rough":  "sar_rough.xml",
    "flood":  "sar_flood.xml",
    "snow":   "sar_snow.xml",
    "stairs": "sar_stairs.xml",
}

class SpotSAREnv(SpotEnv):
    START_X = -6.0
    START_Y =  0.0
    START_Z =  0.45

    def __init__(self, scene="rough", model_dir=None, render_mode=None):
        if scene not in SCENE_PATHS:
            raise ValueError(f"Unknown scene '{scene}'. Available: {list(SCENE_PATHS.keys())}")

        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "mujoco_menagerie", "boston_dynamics_spot")

        scene_path = os.path.join(model_dir, SCENE_PATHS[scene])
        super().__init__(scene_path, render_mode=render_mode)
        self.scene_name = scene

        self._rubble_ids      = []
        self._rubble_base_pos = []
        self._path_rubble_ids = []
        if self.scene_name == "rough":
            self._cache_rubble_geoms()

    def _cache_rubble_geoms(self):
        scattered = [
            "r1_a","r1_b","r1_c","r1_d","r1_e","r1_f","r1_g","r1_h","r1_i",
            "r2_a","r2_b","r2_c","r2_d","r2_e","r2_f","r2_g","r2_h",
            "r2_i","r2_j","r2_k","r2_pile",
            "r3_a","r3_b","r3_c","r3_d","r3_e","r3_f","r3_g","r3_h","r3_i","r3_j",
        ]
        for name in scattered:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self._rubble_ids.append(gid)
                self._rubble_base_pos.append(self.model.geom_pos[gid].copy())

        for name in [f"pr{i}" for i in range(1, 26)]:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self._path_rubble_ids.append(gid)

    def _randomize_rubble(self):
        # size is HALF-EXTENT: uniform(0.015, 0.025) = 3-5 cm total height
        for gid, base_pos in zip(self._rubble_ids, self._rubble_base_pos):
            h = float(np.random.uniform(0.015, 0.025))
            self.model.geom_size[gid, 2] = h
            self.model.geom_pos[gid, 2]  = h
            self.model.geom_pos[gid, 0]  = base_pos[0] + float(np.random.uniform(-0.20, 0.20))
            self.model.geom_pos[gid, 1]  = base_pos[1] + float(np.random.uniform(-0.20, 0.20))

        for gid in self._path_rubble_ids:
            h = float(np.random.uniform(0.015, 0.025))
            self.model.geom_size[gid, 2] = h
            self.model.geom_pos[gid, 2]  = h

    def _calc_reward(self, action):
        base_reward = super()._calc_reward(action)

        if not ENABLE_GAIT_SHAPING:
            return base_reward

        z     = float(self.data.qpos[2])
        x_vel = float(self.data.qvel[0])

        # one-sided height: only penalise being too low, not too high on rubble
        old_height = 2.5 * float(np.exp(-np.square(z - 0.44) / 0.05))
        low_err    = float(max(0.44 - z, 0.0))
        new_height = 2.5 * float(np.exp(-np.square(low_err) / 0.05))

        stall_penalty = 4.0 * float(np.clip(1.0 - x_vel / 0.20, 0.0, 1.0))
        forward_drive = 1.5 * float(np.clip(x_vel / 0.30, 0.0, 1.0))

        roll_rate    = float(self.data.qvel[3])
        pitch_rate   = float(self.data.qvel[4])
        tilt_speed   = float(np.sqrt(roll_rate**2 + pitch_rate**2))
        tilt_penalty = 2.0 * float(np.clip(tilt_speed / 2.0, 0.0, 1.0))

        adjusted = base_reward - old_height + new_height - stall_penalty + forward_drive - tilt_penalty

        self._last_components['stall_penalty'] = stall_penalty
        self._last_components['forward_drive'] = forward_drive
        self._last_components['tilt_penalty']  = tilt_penalty

        return adjusted

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        if self._rubble_ids or self._path_rubble_ids:
            self._randomize_rubble()

        self.data.qpos[0] = self.START_X
        self.data.qpos[1] = self.START_Y
        self.data.qpos[2] = self.START_Z
        self.data.qpos[3] = 1.0  # qw — faces +x
        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0
        self.data.qpos[6] = 0.0

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}
