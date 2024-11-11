# Project Title: Reinforcement Learning Training

## Overview
This project includes scripts for training a reinforcement learning agent using a specified environment and baseline models. Use the instructions below to set up and run the training scripts on your local machine.

## Prerequisites
Before you start, make sure you have the following installed:
- Python 3.8 or higher
- [Gym](https://github.com/openai/gym) or [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)

## Training the Agent
To start training the agent, use the following command:
`python train_back_RL.py --group myoback_1 --num_envs 1 --learning_rate 0.0002 --clip_range 0.1 --seed 7`
- `--group`: Wandb training group name
- `--num_envs`: Number of envs to train in parallel
- `--learning_rate`: learning rate for PPO
- `--clip_range`: clip range for PPO
- `--seed`: env seeding

## Loading a Baseline Model
If you want to load a pre-trained baseline model for further training or evaluation, use the command:
`python loading_baseline.py`

## Baseline Model Video
To view the performance of the baseline model, navigate to:
`videos/baseline_side_video.mp4`

## Contact Information
For more information or assistance, contact:  <br />

Email: huiyi.wang@mail.mcgill.ca <br />
GitHub: cherylwang20
