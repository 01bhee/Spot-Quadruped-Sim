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
        # self._last_contacts = np.zeros(4, dtype=bool)
        self.dt = self.model.opt.timestep * 10 
        self.max_steps = 10000
        self.current_step = 0
        
        self.desired_velocity = np.array([0.5, 0, 0]) 
        self.home_pose = np.array([0, 1.04, -1.8, 0, 1.04, -1.8, 0, 1.04, -1.8, 0, 1.04, -1.8])
        self.last_action = np.zeros(12)

        self.action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(35,), dtype=np.float32)

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
        return np.concatenate([qpos, qvel]).astype(np.float32)

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
        # 1. Strict Target Tracking
        target_v = 0.4 
        x_vel = self.data.qvel[0]
        vel_ratio = np.clip(x_vel / target_v, 0.0, 1.0)
        reward_vel = vel_ratio * 3.0 

        # 2. The Energy Drain
        step_penalty = 2.0

        # 3. HIGH STEP Airtime (Increased to 0.1 for a visible, proud trot!)
        current_contacts = self._get_contact_states()
        first_contact = (self._feet_air_time > 0.1) * current_contacts
        reward_airtime = np.sum(self._feet_air_time[first_contact])
        
        self._feet_air_time += self.dt
        self._feet_air_time[current_contacts] = 0 

        # 4. The Gait Enforcer (Trot Synchronization)
        contacts_float = current_contacts.astype(float)
        trot_error = np.abs(contacts_float[0] - contacts_float[3]) + \
                     np.abs(contacts_float[1] - contacts_float[2])
        sync_penalty = trot_error * 2.0  

        # 5. ALL LEGS ACTIVE & Direction Penalties 
        cost_torque = np.sum(np.square(self.data.ctrl)) * 0.001 
        cost_smooth = np.sum(np.square(action - self.last_action)) * 0.05
        
        # MASSIVE increase to limp penalty. Forces all joints to share the work equally.
        limp_penalty = np.var(np.abs(self.data.qvel[6:18])) * 0.5 
        
        y_vel = self.data.qvel[1]
        yaw_rate = self.data.qvel[5]
        direction_penalty = (np.square(y_vel) + np.square(yaw_rate)) * 2.0

        # 6. STRICT BALANCE (Posture & Height)
        reward_orient = np.exp(-np.sum(np.square(self.data.qpos[4:6])) / 0.1)
        reward_height = np.exp(-np.square(self.data.qpos[2] - 0.46) / 0.01)

        # --- FINAL CALCULATION ---
        # Notice reward_orient is now multiplied by 1.5 for strict balance
        total_reward = reward_vel + \
                       (1.0 * reward_height) + \
                       (1.5 * reward_orient) + \
                       (2.0 * reward_airtime) - \
                       step_penalty - cost_smooth - cost_torque - limp_penalty - sync_penalty - direction_penalty
        
        return total_reward

    def step(self, action):
        self.current_step += 1
        self.data.ctrl[:] = self.home_pose + (action * 0.3)
        
        # frame_skip=10: performs 10 physics steps for every 1 AI action
        mujoco.mj_step(self.model, self.data, nstep=10) 
        
        obs = self._get_obs()
        reward = self._calc_reward(action)
        self.last_action = action.copy()

        terminated = bool(self.data.qpos[2] < 0.25 or abs(self.data.qpos[4]) > 0.4)
        truncated = bool(self.current_step >= self.max_steps)

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # --- PHASE 2: DOMAIN RANDOMIZATION ---
        
        # 1. Randomize Ground Friction
        # friction[0] is the sliding friction.
        new_friction = np.random.uniform(0.2, 1.5)
        # Assuming your floor geom is the first one (index 0)
        self.model.geom_friction[0, 0] = new_friction
        
        # 2. Randomize Torso Mass
        # Trunk is usually body ID 1. We add a random 'payload'
        # mass = nominal_mass + random_offset
        trunk_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        random_payload = np.random.uniform(-1.0, 3.0) # -1kg to +3kg
        self.model.body_mass[trunk_id] = 12.0 + random_payload # Assuming 12kg is base
        
        # 3. Randomize Target Velocity (Conditioning)
        # This is vital for Phase 2: Spot needs to learn different speeds
        self.desired_velocity[0] = np.random.uniform(0.3, 0.7)

        # -------------------------------------

        self.current_step = 0
        self._feet_air_time.fill(0)
        
        # Standing pose initialization
        self.data.qpos[2] = 0.443
        self.data.qpos[3] = 1.0  
        self.data.qpos[7:19] = self.home_pose
        
        return self._get_obs(), {}
    
    def close(self):
        """Properly close the viewer window"""
        if self._viewer is not None:
            self._viewer.close()
