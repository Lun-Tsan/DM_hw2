# Ship Battle A2C Reinforcement Learning Project

This project implements a ship battle simulation game where an AI agent, trained using the Advantage Actor-Critic (A2C) reinforcement learning algorithm, competes against an opponent using an original greedy-focused strategy. The environment is built using OpenAI Gym, and the AI agent is trained using Stable Baselines3.

## Prerequisites

- Python 3.10 (recommand) 
- Virtual environment (`venv`)
- Required Python packages listed in `requirements.txt`

## Installation

To get started with the project, follow these steps:

1. **Create a Virtual Environment(repace python3.10 to your python version if needed)**

   ```bash
   python3.10 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**

   Upgrade `pip` and install required packages using the following commands:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Quick Start


```bash
python ship_battle_a2c.py --train --evaluate --model_path a2c_ship_battle --num_episodes 1000 & tensorboard --logdir=./a2c_ship_battle_tensorboard/
```

Then, open your browser and navigate to [http://localhost:6006/](http://localhost:6006/) to see the training progress.

## Commands

### Train the A2C Agent

To train the A2C agent, use the following command:

```bash
python ship_battle_a2c.py --train
```

### Evaluate the Trained Model

To evaluate the trained A2C agent against the original strategy, use the following command:

```bash
python ship_battle_a2c.py --evaluate --model_path a2c_ship_battle --num_episodes [num_episodes]
```
Replace `[num_episodes]` with the number of evaluation episodes you want to run.

### Train and Evaluate the Agent

To train and immediately evaluate the agent, use the following command:

```bash
python ship_battle_a2c.py --train --evaluate --model_path a2c_ship_battle --num_episodes [num_episodes]
```
Replace `[num_episodes]` with the number of evaluation episodes you want to run.

## Visualizing the Training Process

To visualize the training progress using TensorBoard, run:

```bash
tensorboard --logdir=./a2c_ship_battle_tensorboard/
```

After starting TensorBoard, open [http://localhost:6006/](http://localhost:6006/) in your browser to view the training metrics and agent performance.

## Project Overview

The `ship_battle_a2c.py` script includes:

1. **Training the A2C Agent**: The agent is trained to control Player A in a ship battle game.
2. **Original Strategy**: Player B uses a greedy-focused strategy to compete against the agent.
3. **Evaluation**: The trained agent is evaluated in multiple episodes against the original strategy, and the results are logged.
4. **Visualization**: The results are visualized using Matplotlib and TensorBoard.

## License

This project is open-source and available under the MIT License.

