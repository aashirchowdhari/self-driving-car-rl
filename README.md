# 🚗 Autonomous Car using DDQN

This project implements a **Deep Double Q-Network (DDQN)** agent for autonomous driving in a simulated environment.  
The goal is to train a self-driving car to navigate and avoid obstacles using **Reinforcement Learning** techniques.

---

## 📌 Features
- Implementation of **Deep Double Q-Learning (DDQN)**  
- Epsilon-greedy exploration strategy  
- Replay memory for experience replay  
- Target network for stable learning  
- Car simulation environment (custom or OpenAI Gym)  
- Training and evaluation scripts  

---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/<aashirchowdhari>/<self-driving-car-rl>.git
cd <self-driving-car-rl>
Install the dependencies:

pip install -r requirements.txt

📦 Requirements

Add this to your requirements.txt (adjust depending on your framework):

numpy
tensorflow
keras
torch
gym
pygame
matplotlib

▶️ Usage
Train the Agent
python train.py

Test the Trained Agent
python test.py

📊 Results

📈 Reward vs Episodes plots

🚘 Simulation of the trained car driving in the environment

(Optional: include gifs/screenshots of your car driving here)

📚 References

Mnih et al., Playing Atari with Deep Reinforcement Learning (2015)

Hasselt et al., Deep Reinforcement Learning with Double Q-learning (2016)

👨‍💻 Author

Developed by Aashir Chowdhari as part of an RL project.
