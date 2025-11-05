# Mussel Power

**Team Members:** Chloe Bendenoun, Christian Friis, Catarina Lorenzo, Madeleine Abu  
**Project Title:** Mussel Power

---

## Project Summary

### The Problem

Many water bodies are contaminated with pathogens and harmful algae. We are stuck between two flawed approaches:

- **Natural mussels** filter water but accumulate pathogens without killing them, making them unsuitable for safe purification. Moreover, mussels themselves are fragile:  
  - They bioaccumulate toxins and pathogens, eventually dying and releasing pollutants back into the water  
  - They depend on specific ecosystems; introducing them artificially risks upsetting local biodiversity  
  - They cannot survive in highly polluted or warm waters, limiting practical use 
- **Existing mechanical or chemical solutions** are often:
  - Energy-intensive
  - Not adaptive
  - Ineffective under variable water conditions (flow, turbidity, pathogen density)

**Challenge:** Design a system that actively kills pathogens while remaining energy-efficient and adaptive, even as the pod moves through different parts of a water body.

### Our Solution

**Mussel Power** is our answer. We started with **biomimicry**, taking direct inspiration from the mussel's brilliant natural design but fixing its fatal flaws.

Our solution is an end-to-end system with three core parts:

- **The Pod**: This is the physical heart of our system, inspired by the cilia structure of mussels. It uses an internal cilia-like motion to actively draw in water with minimal energy consumption. As the water flows through, integrated UV-C light eliminates bacteria, algae, and other contaminants in real-time. Each unit is modular and designed to work in clusters, making the system customizable and scalable for any body of water, from a small pond to an urban waterway.
- **The App**: A real-time dashboard that acts as the mission control. It allows operators to monitor the Water Safety Score, track energy usage, and manage every pod in the field. Crucially, this allows our pods to act as decentralized data-gathering points, bringing real-time water quality data to rural or remote communities that have never had access to it before.
- **The AI**: This is what makes our product unique and safe, robust, and highly efficient. Instead of a fixed "on/off" setting, our Reinforcement Learning AI gives each pod adaptive intelligence.
  - It is safe from the moment it's deployed, using a **formula-based initialization** to ensure effective purification even before the AI has fully learned.
  - It is robust, using its **dual control** over UV intensity and cilia vibration to handle even extreme changes in water turbidity or flow.
  - Most importantly, it is **energy-efficient**, continuously learning and calculating the absolute minimum energy required to keep the water safe. This isn't just automation; it's a self-optimizing system that makes purification radically more efficient.

---

## How AI Is Used

### 1. Real-Time Decision-Making
The RL agent observes:

- **Flow rate** (water movement through the pod)  
- **Turbidity** (clarity of water)  
- **Pathogen proxy** (estimated contamination)  
- **Current UV intensity**  

It then chooses:

- **Vibration frequency:** Controls water residence time in the UV chamber.  
- **UV intensity:** Determines pathogen kill rate versus energy cost.

---

### 2. Learning Through Rewards
The agent optimizes a **reward function**:

`Reward = Pathogen Kill Effect - Energy Cost`

- Positive reward: more pathogens killed with minimal energy.  
- Negative reward: high energy use with low cleaning effect.

Over time, the AI learns **optimal control strategies** without manual tuning.

---

### 3. Continuous Adaptation
SmartPod is adaptive, not static:

- Handles sudden changes in turbidity or flow.  
- Balances energy use and purification performance.  
- Can scale to multiple pods, creating intelligent colonies of water purification units.

---

## Algorithm Used
**Deep Deterministic Policy Gradient (DDPG)**

- A model-free, off-policy RL algorithm suitable for **continuous control problems**.  
- Employs **actor-critic architecture**:  
  - **Actor:** proposes continuous actions (vibration, UV).  
  - **Critic:** evaluates actions based on expected reward.  
- The simulation allows testing and training without requiring real hardware.

---

## Key Features

- **Adaptive Control:** The pod dynamically adjusts its vibration and UV intensity.  
- **Energy Efficiency:** Optimizes pathogen kill rate while minimizing energy consumption.  
- **Safe Initialization:** Formula-based starting values ensure first-time operation is effective.  
- **Visualization-Ready:** Tracks rewards, actions, and state variables for performance analysis.  

---

## Simulation Details

- **Environment Steps:** 500 per episode  
- **Episodes:** 20 (default, can be increased for longer training)  
- **Action Space:** Continuous vibration frequency [0.5–5 Hz], UV intensity [0–100%]  
- **State Space:** Normalized flow rate, turbidity, pathogen proxy, current UV  
- **Replay Interval:** 20 steps (agent updates periodically for efficiency)  

---

## Link to App

https://aqua-pure-ai-copy-935903c1.base44.app/RLAgent
