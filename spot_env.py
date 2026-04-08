import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer # FIXED: Added this import for rendering
import numpy as np

class SpotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, model_path, render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._viewer = None # FIXED: Initialize viewer as None

        # --- GAIT TRACKING VARIABLES ---
        self._feet_air_time = np.zeros(4)
        # Tracks how long each diagonal pair has been CONTINUOUSLY on the ground.
        # pair 0 = FL+HR,  pair 1 = FR+HL
        self._pair_contact_time = np.zeros(2)
        # Requires 2 consecutive contact steps to count as a real landing,
        # preventing passive ground-skims from resetting the air_time counter.
        self._prev_contacts = np.zeros(4, dtype=bool)
        # HOT GROUND: tracks how long each individual foot has been continuously grounded.
        # Penalises any foot on the ground too long, forcing it to lift periodically.
        self._feet_ground_time = np.zeros(4)
        self.dt = self.model.opt.timestep * 10 
        self.max_steps = 10000
        self.current_step = 0
        
        self.desired_velocity = np.array([0.5, 0, 0]) 
        self.home_pose = np.array([0, 1.04, -1.8, 0, 1.04, -1.8, 0, 1.04, -1.8, 0, 1.04, -1.8])
        self.last_action = np.zeros(12)
        self._last_components = {}  # populated each step for TensorBoard logging

        self.action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(51,), dtype=np.float32)

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                # launch_passive opens the MuJoCo window
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync() # Syncs the physics data to the visual window

    def _get_obs(self):
        """
        Qpos is 19 values(ask korede) but removing 
        the first two(x and y from world frame) it becomes 17

        qvel -  
        """
        qpos = self.data.qpos.flat[2:] 
        qvel = self.data.qvel.flat  
        contacts = self._get_contact_states().astype(np.float32)  # 4 values
        return np.concatenate([qpos, qvel, self.last_action, contacts]).astype(np.float32)

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
        reward_alive = 0.5 + 2.5 * np.clip(x_vel / 0.4, 0.0, 1.0)

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

        # 6. DIAGONAL TROT PHASE REWARD
        # CHANGED: replaced hard binary reward with a soft continuous version.
        # Old version gave 0 reward during the double-support transition phase
        # (the brief moment where both pairs partially touch down simultaneously),
        # creating a reward cliff right at footstrike. The robot learned to hesitate
        # or rush through this transition, causing the lurch/freeze pattern.
        # New version gives partial credit proportional to how synchronised each pair is,
        # so the transition is always rewarded rather than punished.
        pair_A_sync = float(current_contacts[0]) * float(current_contacts[3])  # FL+HR: 1.0 only if both down
        pair_B_sync = float(current_contacts[1]) * float(current_contacts[2])  # FR+HL: 1.0 only if both down
        reward_trot_phase = 2.5 * max(
            pair_A_sync * (1.0 - pair_B_sync),  # A fully down, B lifting off
            pair_B_sync * (1.0 - pair_A_sync),  # B fully down, A lifting off
        )

        # 7. COST_FLIGHT (quadratic)... why?
        feet_on_ground = int(np.sum(current_contacts))
        feet_deficit = max(0, 2 - feet_on_ground)
        cost_flight = 3.0 * (feet_deficit ** 2)

        # 8. COST_LEG_NEGLECT
        cost_leg_neglect = 4.0 * float(np.sum(np.maximum(self._feet_air_time - 0.5, 0.0)))

        # 9. COST_STATIC_HOLD
        # CHANGED: threshold 0.7s → 1.0s.
        # At 0.7s the penalty was firing during normal stance phases (~0.4-0.6s),
        # doubly punishing the robot alongside cost_foot_overstance. The robot
        # responded by rushing its stance and stumbling. 1.0s gives real margin.
        pair_A_down = bool(current_contacts[0] and current_contacts[3])
        pair_B_down = bool(current_contacts[1] and current_contacts[2])
        self._pair_contact_time[0] = (self._pair_contact_time[0] + self.dt) if pair_A_down else 0.0
        self._pair_contact_time[1] = (self._pair_contact_time[1] + self.dt) if pair_B_down else 0.0
        cost_static_hold = 0.2 * float(np.sum(np.maximum(self._pair_contact_time - 1.0, 0.0)))

        # 10. HOT GROUND / FOOT OVERSTANCE
        # CHANGED: threshold 0.5s → 0.7s, same reasoning as static_hold.
        # A normal trot stance phase can easily reach 0.5s — this was firing
        # constantly during healthy gait, not just during freezing. 0.7s only
        # fires for genuinely prolonged overstance.
        self._feet_ground_time[current_contacts] += self.dt
        self._feet_ground_time[~current_contacts] = 0.0
        cost_foot_overstance = 1.5 * float(np.sum(np.maximum(self._feet_ground_time - 0.7, 0.0)))

        # FOOT CLEARANCE: gentle polish reward for lifting feet cleanly.
        # Peaks at +0.5/foot when foot is exactly 0.1m above ground during swing.
        # Uses geom_xpos (GEOM lookup) — "FL" etc. are geom names, not body names.
        # sigma=0.05m: reward drops to 0.18 if foot is only 5cm high or 15cm high.
        reward_foot_clearance = 0.0
        for i, name in enumerate(["FL", "FR", "HL", "HR"]):
            if not current_contacts[i]:  # swing phase only
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if geom_id >= 0:
                    foot_height = self.data.geom_xpos[geom_id][2]
                    reward_foot_clearance += np.exp(-np.square(foot_height - 0.1) / (0.05**2))
        reward_foot_clearance *= 0.5

        # 11. SLIP PENALTY
        foot_slip_cost = 0.0
        foot_names = ["FL", "FR", "HL", "HR"]
        for i, name in enumerate(foot_names):
            if current_contacts[i]:
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if geom_id != -1:
                    res = np.zeros(6, dtype=np.float64)
                    mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_GEOM, geom_id, res, 0)
                    foot_slip_cost += np.sum(np.square(res[:2]))
        cost_slip = foot_slip_cost * 0.5

        # 12. SMOOTHNESS & ENERGY
        cost_smooth = 0.15 * np.sum(np.square(action - self.last_action))
        cost_torque = 0.001 * np.sum(np.square(self.data.ctrl))

        # 13. GAIT SYMMETRY
        # CHANGED: sigma 0.05 → 0.15. The old value was so tight that any real-world
        # timing jitter between pairs wiped the reward out entirely. The robot's
        # response was to minimise both pairs' airtime amplitude (keeping them close
        # to zero and therefore symmetric) rather than trotting boldly and accepting
        # minor asymmetry. 0.15 rewards genuine trotting even when one pair is
        # slightly ahead of the other in its swing phase.
        contacts_float = current_contacts.astype(float)
        trot_error = np.abs(contacts_float[0] - contacts_float[3]) + \
                    np.abs(contacts_float[1] - contacts_float[2])
        sync_penalty = trot_error * 1.5

        pair_A_air = self._feet_air_time[0] + self._feet_air_time[3]
        pair_B_air = self._feet_air_time[1] + self._feet_air_time[2]
        reward_symmetry = 2.5 * float(np.exp(-np.square(pair_A_air - pair_B_air) / 0.15))

        # 14. DIRECTION, VERTICAL, POSE
        direction_penalty = (np.square(self.data.qvel[1]) + np.square(self.data.qvel[5])) * 2.0
        cost_vertical = 0.3 * np.square(self.data.qvel[2])
        cost_pose_deviation = 0.3 * np.sum(np.square(self.data.qpos[7:19] - self.home_pose))

        # --- TOTAL ---
        total_reward = (
            reward_vel + reward_orient + reward_height + reward_airtime +
            reward_alive + reward_trot_phase + reward_symmetry + reward_foot_clearance
        ) - (
            cost_smooth + cost_torque + cost_slip + sync_penalty +
            direction_penalty + cost_flight + cost_leg_neglect +
            cost_vertical + cost_static_hold + cost_pose_deviation +
            cost_foot_overstance
        )

        # Cache key components so step() can pass them to TensorBoard via info dict.
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
            'cost_pose_deviation':   float(cost_pose_deviation),
        }
        return total_reward

    def step(self, action):
        self.current_step += 1
        # Reduced 0.45 → 0.38: hind legs have more mechanical leverage — smaller
        # scale directly limits the extreme positions they can reach during a kick.
        self.data.ctrl[:] = self.home_pose + (action * 0.2)
        
        # frame_skip=10: performs 10 physics steps for every 1 AI action
        mujoco.mj_step(self.model, self.data, nstep=10) 
        
        obs = self._get_obs()
        reward = self._calc_reward(action)
        self.last_action = action.copy()

        # Up-vector termination: catches falls in ANY direction (roll, pitch, diagonal).
        # The old abs(qpos[4])>0.6 check only caught sideways roll.
        qx, qy = self.data.qpos[4], self.data.qpos[5]
        up_z = 1.0 - 2.0 * (qx**2 + qy**2)  # 1.0=upright, <0.5 means >60° tilt
        terminated = bool(self.data.qpos[2] < 0.30 or up_z < 0.5)
        truncated = bool(self.current_step >= self.max_steps)

        return obs, reward, terminated, truncated, dict(self._last_components)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # Domain randomization disabled — flat ground, fixed mass, fixed speed.
        # Re-enable once the agent can walk stably.

        self.current_step = 0
        self._feet_air_time.fill(0)
        self._pair_contact_time.fill(0)
        self._prev_contacts.fill(False)
        self._feet_ground_time.fill(0)
        
        # Standing pose initialization
        self.data.qpos[2] = 0.443
        self.data.qpos[3] = 1.0  
        self.data.qpos[7:19] = self.home_pose
        
        return self._get_obs(), {}
    
    def close(self):
        """Properly close the viewer window"""
        if self._viewer is not None:
            self._viewer.close()
