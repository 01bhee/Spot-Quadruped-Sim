# 🐾 Spot Quadruped Terrain Adaptation — Curriculum RL

A reinforcement learning system that trains a Boston Dynamics Spot quadruped to walk across progressively challenging terrains. The robot learns using PPO (Proximal Policy Optimization) with proprioceptive-only observations — no cameras, just joint states and body pose. 🤖

---

## ⚙️ Install

```bash
pip install stable-baselines3[extra] mujoco torch
```

The Spot robot model is from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie). Clone it and place the `boston_dynamics_spot` folder under `mujoco_menagerie/`. 📦

---

## 🧠 How It Works

Training is split into two phases:

**Phase 1 — Flat ground (`train.py`)** 🏔️  
Spot learns a stable trot gait from scratch on flat ground. Three sub-phases are controlled by flags in `spot_env.py`:
- 🟡 Phase A: robot discovers forward motion
- 🟠 Phase B: gait shaping rewards kick in — trot emerges
- 🟢 Phase C: path fidelity reward added — robot learns to walk straight

**Phase 2 — Curriculum (`trainC.py`)** 📈  
The flat-ground policy is fine-tuned across four terrain types in three stages:

| Stage | Terrain | Steps | Goal |
|-------|---------|-------|------|
| 1 | 🪨 Rough / rubble | 1 000 000 | Step over debris |
| 2 | 🌊 Flood + ❄️ Snow | 500 000 | Adapt to low friction |
| 3 | 🌍 All four terrains | 500 000 | Full generalisation |

The script pauses between stages so you can evaluate before continuing. ⏸️

---

## 🗂️ Files

| File | Description |
|------|-------------|
| `spot_env.py` | Base Gymnasium environment — defines the 52-dim observation space, reward function, and episode termination |
| `spot_sar_env.py` | Extends `SpotEnv` for terrain training — overrides the reward to handle uneven ground and adds domain randomisation for rubble |
| `train.py` | PPO training on flat ground |
| `trainC.py` | Curriculum training across the three terrain stages |
| `enjoy.py` | Loads and visualises a flat-ground policy |
| `testinsar.py` | Loads and visualises a curriculum policy, prints distance travelled and fall/stall results per episode |

---

## 👁️ Observation Space (52-dim)

| Component | Dims |
|-----------|------|
| 🧍 Body pose — height, orientation, joint angles | 17 |
| ↔️ Y position (lateral drift) | 1 |
| 💨 Linear + angular + joint velocities | 18 |
| 🦾 Last action (joint torques) | 12 |
| 🦶 Foot contacts | 4 |

---

## 🚀 Usage

### Train flat-ground base policy
```bash
python train.py
```
Output: `models/<BRAIN_NAME>.zip` 💾

### Run curriculum training
Update `BASE_BRAIN` in `trainC.py` to point to your flat-ground model, then:
```bash
python trainC.py
```
Output: `models/curriculum/` 💾

### Visualise flat-ground policy 🎥
```bash
python enjoy.py
```

### Visualise curriculum policy 🎬
Set `STAGE` and `SCENE` at the top of `testinsar.py`, then:
```bash
python testinsar.py
```

| STAGE | Valid SCENE values |
|-------|--------------------|
| 1 | `"rough"` |
| 2 | `"flood"`, `"snow"` |
| 3 | `"rough"`, `"flood"`, `"snow"`, `"stairs"` |

### 📊 Monitor training
```bash
tensorboard --logdir logs/
```

---

## 🌍 Terrain Scenes

| Scene | Friction | Description |
|-------|----------|-------------|
| 🪨 `rough` | 1.0 | Rubble debris 3–5 cm tall, randomised layout each episode |
| 🌊 `flood` | 0.4 | Wet flat ground |
| ❄️ `snow` | 0.2 | Icy ground |
| 🪜 `stairs` | 1.0 | Five steps with 5 cm rise |
