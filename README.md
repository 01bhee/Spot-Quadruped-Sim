<div align="center">
  <h1>🐕 Spot Quadruped Simulation</h1>
  <p><b>Locomotion via Proximal Policy Optimization (PPO) & MuJoCo</b></p>

  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/Framework-Stable--Baselines3-green?style=for-the-badge" alt="Framework">
  <img src="https://img.shields.io/badge/Physics-MuJoCo-red?style=for-the-badge" alt="Physics Engine">
</div>

---

## 🚀 Executive Summary
This repository contains a high-fidelity simulation of the **Boston Dynamics Spot** robot. Developed as a Final Year Project (FYP), it explores the intersection of **Deep Reinforcement Learning** and **Legged Locomotion**. The goal is to train a robust control policy capable of stable walking and recovery in dynamic environments.

### 🛠️ Core Technology Stack
<table align="center">
  <tr>
    <td align="center"><b>Physics Engine</b></td>
    <td align="center"><b>RL Algorithm</b></td>
    <td align="center"><b>Environment</b></td>
  </tr>
  <tr>
    <td align="center">MuJoCo (2.3+)</td>
    <td align="center">PPO (Proximal Policy Optimization)</td>
    <td align="center">Gymnasium / OpenAI Gym</td>
  </tr>
</table>

---

## 🧠 Training Architecture
The agent is trained using a **PPO (Proximal Policy Optimization)** actor-critic architecture. The reward function is meticulously balanced to prioritize:

* **Forward Progress:** Linear velocity tracking ($v_{target} = 0.5\text{ m/s}$).
* **Torque Efficiency:** Minimizing $L2$ norm of joint torques to prevent "jittery" movement.
* **Posture Stability:** Penalizing excessive roll and pitch to maintain a level chassis.

---

## 🔮 Future Research: Meta-Reinforcement Learning
The next milestone for this project is **Terrain Adaptation via Meta-RL**. 

> "Standard RL produces specialists; Meta-RL produces survivors."

The implementation will focus on algorithms like **PEARL (Probabilistic Embeddings for Adult RL)** to allow Spot to instantly adjust its gait when moving between surfaces like sand, gravel, and concrete without needing a full retraining session.

---
