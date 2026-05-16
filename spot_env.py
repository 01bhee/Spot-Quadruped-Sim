import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np

# =============================================================
# TWO-PHASE TRAINING TOGGLE
# =============================================================
# PHASE 1 (False): retrain the base trot with the new 52-dim
#                  observation. Path-fidelity rewards are OFF
#                  so the random initial policy can discover
#                  walking without being trapped in a "stand
#                  still" local optimum.
#
# PHASE 2 (True):  warm-start from Phase 1's saved model, flip
#                  this to True, and the path-fidelity terms
#                  switch on. The existing trot is now refined
#                  to stay near y=0.
#
# Flip this ONE flag between the two training runs. Nothing
# else in the file needs to change.
# =============================================================
ENABLE_PATH_FIDELITY = False   # Phase A: off. Switch on in Phase C.
ENABLE_GAIT_SHAPING  = True   # Phase A: off. Switch on in Phase B.


class SpotEnv(gym.Env):
    """
    Spot quadruped locomotion env with a two-phase training plan.

    PHASE 1: ENABLE_PATH_FIDELITY=False, ENABLE_GAIT_SHAPING=False
        - Velocity-first reward only. Robot discovers forward motion.
        - 52-dim observation (y-position added but reward ignores it).
        - Train ~100k steps from scratch. Save as model1B_v2_phaseA.

    PHASE 2: ENABLE_PATH_FIDELITY=False, ENABLE_GAIT_SHAPING=True
        - Full gait shaping terms active. Trot emerges.
        - Warm-start from phaseA. Train ~150k steps.
        - Save as model1B_v2_phaseB.

    PHASE 3: ENABLE_PATH_FIDELITY=True, ENABLE_GAIT_SHAPING=True
        - Path fidelity terms active. Trot learns to stay centered.
        - Warm-start from phaseB. Train ~100k steps.
        - Save as model1B_v2 (curriculum warm-start).

    FIXES IN THIS VERSION
    ---------------------
    1. Y-boundary termination: flat ground XML has no walls, so the
       robot could drift to y=-23m accumulating -1000 reward/step
       for the full 10k episode. Now the episode terminates if the
       robot goes beyond ±3m in Y.
    2. Path penalty weight reduced 2.0 -> 0.5 for flat-ground Phase C.
       On flat ground there are no physical barriers so large Y drift
       was producing -9M episode rewards and destabilising the value
       function. 0.5 is enough to incentivise centering without
       catastrophic penalties during the correction phase.
       NOTE: restore to 2.0 in spot_sar_env.py for the curriculum —
       the SAR scenes have terrain boundaries that naturally limit drift.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, model_path, render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._viewer = None

        # --- GAIT TRACKING ---
        self._feet_air_time = np.zeros(4)
        self._pair_contact_time = np.zeros(2)
        self._prev_contacts = np.zeros(4, dtype=bool)
        self._feet_ground_time = np.zeros(4)
        self.dt = self.model.opt.timestep * 10
        self.max_steps = 10000
        self.current_step = 0

        self.desired_velocity = np.array([0.5, 0, 0])
        self.home_pose = np.array([0, 1.04, -1.8, 0, 1.04, -1.8, 0, 1.04, -1.8, 0, 1.04, -1.8])
        self.last_action = np.zeros(12)
        self._last_components = {}

        self.action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        # 52-dim: y-position is always in the observation regardless of phase.
        # The reward may or may not use it depending on the toggles above.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(52,), dtype=np.float32)

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()

    def _get_obs(self):
        """
        Observation (52-dim):
            qpos[2:]   -> 17 values : z-height, quat(4), 12 joint angles
            y_position -> 1 value   : lateral offset from centerline (y=0)
            qvel       -> 18 values : full velocity state
            last_action-> 12 values : previous action for smoothness signal
            contacts   -> 4 values  : foot contact booleans

        x-position is still stripped — the policy shouldn't condition
        on how far from the start it is. y-position is kept so the
        policy can see and correct lateral drift.
        """
        qpos_rest  = self.data.qpos.flat[2:]
        y_position = np.array([self.data.qpos[1]])
        qvel       = self.data.qvel.flat
        contacts   = self._get_contact_states().astype(np.float32)
        return np.concatenate([
            qpos_rest, y_position, qvel, self.last_action, contacts
        ]).astype(np.float32)

    def _get_contact_states(self):
        contacts = np.zeros(4, dtype=bool)
        foot_geoms = ["FL", "FR", "HL", "HR"]
        for i, name in enumerate(foot_geoms):
            try:
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                for c in range(self.data.ncon):
                    contact = self.data.contact[c]
                    if contact.geom1 == geom_id or contact.geom2 == geom_id:
                        contacts[i] = True
            except:
                pass
        return contacts

    def _calc_reward(self, action):
        x_vel = self.data.qvel[0]

        # 1. VELOCITY TRACKING
        target_v = 0.4
        reward_vel = 10.0 * np.clip(x_vel / target_v, 0.0, 1.3)

        # 2. SURVIVAL & POSTURE
        reward_orient = 2.5 * np.exp(-np.sum(np.square(self.data.qpos[4:6])) / 0.25)
        reward_height = 2.5 * np.exp(-np.square(self.data.qpos[2] - 0.44) / 0.05)

        # 3. ALIVE BONUS
        reward_alive = 0.5 + 2.0 * np.clip(x_vel / 0.4, 0.0, 1.0)

        # 4. CONTACT STATES
        current_contacts = self._get_contact_states()

        # 5. AIRTIME (Gait)
        vel_bonus = np.clip(x_vel / 0.25, 0.0, 1.0)
        first_contact = (self._feet_air_time > 0.4) * current_contacts
        capped_airtime = np.minimum(self._feet_air_time[first_contact] - 0.1, 0.4)
        reward_airtime = 5.0 * np.sum(np.maximum(capped_airtime, 0.0)) * vel_bonus

        sustained_contacts = current_contacts & self._prev_contacts
        self._prev_contacts = current_contacts.copy()
        self._feet_air_time += self.dt
        self._feet_air_time[sustained_contacts] = 0

        # 6. DIAGONAL TROT PHASE REWARD (soft continuous)
        pair_A_sync = float(current_contacts[0]) * float(current_contacts[3])
        pair_B_sync = float(current_contacts[1]) * float(current_contacts[2])
        reward_trot_phase = 2.5 * max(
            pair_A_sync * (1.0 - pair_B_sync),
            pair_B_sync * (1.0 - pair_A_sync),
        )

        # 7. COST_FLIGHT
        feet_on_ground = int(np.sum(current_contacts))
        feet_deficit = max(0, 2 - feet_on_ground)
        cost_flight = 3.0 * (feet_deficit ** 2)

        # 8. COST_LEG_NEGLECT
        cost_leg_neglect = 4.0 * float(np.sum(np.maximum(self._feet_air_time - 0.5, 0.0)))

        # 9. COST_STATIC_HOLD
        pair_A_down = bool(current_contacts[0] and current_contacts[3])
        pair_B_down = bool(current_contacts[1] and current_contacts[2])
        self._pair_contact_time[0] = (self._pair_contact_time[0] + self.dt) if pair_A_down else 0.0
        self._pair_contact_time[1] = (self._pair_contact_time[1] + self.dt) if pair_B_down else 0.0
        cost_static_hold = 0.2 * float(np.sum(np.maximum(self._pair_contact_time - 1.0, 0.0)))

        # 10. HOT GROUND / FOOT OVERSTANCE
        self._feet_ground_time[current_contacts] += self.dt
        self._feet_ground_time[~current_contacts] = 0.0
        cost_foot_overstance = 1.5 * float(np.sum(np.maximum(self._feet_ground_time - 0.7, 0.0)))

        # 11. FOOT CLEARANCE
        reward_foot_clearance = 0.0
        for i, name in enumerate(["FL", "FR", "HL", "HR"]):
            if not current_contacts[i]:
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if geom_id >= 0:
                    foot_height = self.data.geom_xpos[geom_id][2]
                    reward_foot_clearance += np.exp(-np.square(foot_height - 0.1) / (0.05**2))
        reward_foot_clearance *= 0.5

        # 12. SLIP PENALTY
        foot_slip_cost = 0.0
        for i, name in enumerate(["FL", "FR", "HL", "HR"]):
            if current_contacts[i]:
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if geom_id != -1:
                    res = np.zeros(6, dtype=np.float64)
                    mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_GEOM, geom_id, res, 0)
                    foot_slip_cost += np.sum(np.square(res[:2]))
        cost_slip = foot_slip_cost * 0.5

        # 13. SMOOTHNESS & ENERGY
        cost_smooth = 0.15 * np.sum(np.square(action - self.last_action))
        cost_torque = 0.001 * np.sum(np.square(self.data.ctrl))

        # 14. GAIT SYMMETRY
        contacts_float = current_contacts.astype(float)
        trot_error = np.abs(contacts_float[0] - contacts_float[3]) + \
                     np.abs(contacts_float[1] - contacts_float[2])
        sync_penalty = trot_error * 1.5

        pair_A_air = self._feet_air_time[0] + self._feet_air_time[3]
        pair_B_air = self._feet_air_time[1] + self._feet_air_time[2]
        reward_symmetry = 2.5 * float(np.exp(-np.square(pair_A_air - pair_B_air) / 0.15))

        # 15. DIRECTION, VERTICAL, POSE
        # Always use weight 5.0 — lateral velocity is never desirable, even
        # in early phases. The old phase-dependent 2.0 was too weak and
        # allowed the policy to exploit diagonal walking.
        direction_penalty = (np.square(self.data.qvel[1]) + np.square(self.data.qvel[5])) * 5.0
        cost_vertical      = 0.3 * np.square(self.data.qvel[2])
        cost_pose_deviation = 0.3 * np.sum(np.square(self.data.qpos[7:19] - self.home_pose))

        # 15b. YAW ANGLE PENALTY (new)
        # Penalise the robot for *being* turned off-heading, not just for
        # turning. Without this the robot can settle at a fixed diagonal
        # heading (qvel[5] ≈ 0 so no yaw-rate penalty) and walk sideways
        # indefinitely. We extract yaw from the quaternion:
        #   qpos[3:7] = [qw, qx, qy, qz]
        #   yaw = atan2(2(qw*qz + qx*qy), 1 - 2(qy² + qz²))
        qw, qx, qy_q, qz = self.data.qpos[3], self.data.qpos[4], self.data.qpos[5], self.data.qpos[6]
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy_q),
                         1.0 - 2.0 * (qy_q**2 + qz**2))
        cost_yaw_heading = 4.0 * np.square(yaw)

        # 16. PATH FIDELITY
        # Always apply a mild lateral-position penalty (weight 0.3) so that
        # drift is discouraged from the start, not just in Phase C.
        # Phase C ups the weight to 0.5 for stronger centering.
        # In spot_sar_env.py the weight stays at 2.0 because SAR terrain
        # has physical boundaries that naturally cap how far y can drift.
        y_pos = self.data.qpos[1]
        if ENABLE_PATH_FIDELITY:
            cost_path_deviation = 0.5 * np.square(y_pos)
        else:
            cost_path_deviation = 0.3 * np.square(y_pos)   # mild always-on drift penalty

        # --- TOTAL ---
        if not ENABLE_GAIT_SHAPING:
            # Phase A: velocity-first only. Robot's sole job is to move forward.
            # No gait terms — they create conflicting gradients on a random policy
            # and cause the "stand still" local optimum.
            # direction_penalty, cost_yaw_heading, cost_path_deviation included
            # even here so the policy never learns a slanted-walk shortcut.
            total_reward = (
                reward_vel + reward_orient + reward_height + reward_alive
            ) - (
                cost_smooth + cost_torque + cost_pose_deviation +
                direction_penalty + cost_yaw_heading + cost_path_deviation
            )
        else:
            # Phases B and C: full reward. All gait quality terms active.
            total_reward = (
                reward_vel + reward_orient + reward_height + reward_airtime +
                reward_alive + reward_trot_phase + reward_symmetry + reward_foot_clearance
            ) - (
                cost_smooth + cost_torque + cost_slip + sync_penalty +
                direction_penalty + cost_yaw_heading + cost_flight + cost_leg_neglect +
                cost_vertical + cost_static_hold + cost_pose_deviation +
                cost_foot_overstance + cost_path_deviation
            )

        # TensorBoard logging
        self._last_components = {
            'reward_vel':            float(reward_vel),
            'reward_airtime':        float(reward_airtime),
            'reward_trot_phase':     float(reward_trot_phase),
            'reward_foot_clearance': float(reward_foot_clearance),
            'reward_symmetry':       float(reward_symmetry),
            'cost_flight':           float(cost_flight),
            'cost_leg_neglect':      float(cost_leg_neglect),
            'cost_foot_overstance':  float(cost_foot_overstance),
            'sync_penalty':          float(sync_penalty),
            'direction_penalty':     float(direction_penalty),
            'cost_yaw_heading':      float(cost_yaw_heading),
            'cost_pose_deviation':   float(cost_pose_deviation),
            'cost_path_deviation':   float(cost_path_deviation),
            'y_position':            float(y_pos),
            'yaw_angle_deg':         float(np.degrees(yaw)),
            # Phase A=1, B=2, C=3 — was always 1.0 for A and B, making them indistinguishable in TensorBoard.
            'phase':                 (3.0 if ENABLE_PATH_FIDELITY else 2.0) if ENABLE_GAIT_SHAPING else 1.0,
        }
        return total_reward

    def step(self, action):
        self.current_step += 1
        self.data.ctrl[:] = self.home_pose + (action * 0.2)
        mujoco.mj_step(self.model, self.data, nstep=10)

        obs    = self._get_obs()
        reward = self._calc_reward(action)
        self.last_action = action.copy()

        qx, qy = self.data.qpos[4], self.data.qpos[5]
        up_z = 1.0 - 2.0 * (qx**2 + qy**2)

        # FIXED: Y-boundary termination.
        # The flat ground XML has no walls. Without this, the robot can
        # drift sideways indefinitely — we observed y=-23m across a full
        # 10k-step episode, producing ep_rew_mean of -9.3M. Terminating
        # at ±3m gives the episode a consequence for lateral drift and
        # keeps per-episode penalties manageable so the value function
        # can actually learn. 3m is generous — in the SAR scenes the
        # usable corridor is roughly ±2m before terrain becomes degenerate.
        y_out_of_bounds = bool(abs(self.data.qpos[1]) > 3.0)

        terminated = bool(
            self.data.qpos[2] < 0.30   # fallen (height too low)
            or up_z < 0.5              # tipped over (>60° tilt)
            or y_out_of_bounds         # drifted too far sideways — NEW
        )
        truncated = bool(self.current_step >= self.max_steps)

        return obs, reward, terminated, truncated, dict(self._last_components)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.current_step = 0
        self._feet_air_time.fill(0)
        self._pair_contact_time.fill(0)
        self._prev_contacts.fill(False)
        self._feet_ground_time.fill(0)

        self.data.qpos[2] = 0.443
        self.data.qpos[3] = 1.0
        self.data.qpos[7:19] = self.home_pose

        return self._get_obs(), {}

    def close(self):
        if self._viewer is not None:
            self._viewer.close()