import os
from collections import deque
import random
from DDPG import DDPG
import gymnasium as gym
import panda_gym
import time
import datetime
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


current_time = datetime.datetime.now()
writer = SummaryWriter("Base_solution/runs")
directory_name = f"Base_solution/DDPG_checkpoints/{current_time}"
os.makedirs(directory_name)


bath_size = 128
episode_n = 5
trajectory_len = 50
start_learning = 80

env = gym.make('PandaReachDense-v3', render_mode='human')

obs_shape = env.observation_space['observation'].shape[0] + \
    env.observation_space['achieved_goal'].shape[0] + \
    env.observation_space['desired_goal'].shape[0]

action_shape = env.action_space.shape[0]

action_min = env.action_space.low[0]
action_max = env.action_space.high[0]

agent = DDPG(obs_shape, action_shape, action_min=action_min, action_max=action_max, bath_size=bath_size, start_learning=start_learning,
             noise_decrease=1 / (episode_n * trajectory_len), polyak=0.05, buffer_size=10**6, critic_lr=2e-3, actor_lr=1e-3, noise_scaler=0.3)

log_data = {'Total_reward': []}


cur_episode = 0
step = 0

agent.load_models(
    'Base_solution/DDPG_checkpoints/best_model/episode1500')

for episode in range(episode_n):
    total_reward = 0
    observation, info = env.reset()
    state = np.concatenate([observation['observation'],
                            observation['achieved_goal'], observation['desired_goal']], axis=0)

    for _ in range(trajectory_len):
        action = agent.get_action(state).detach().numpy()
        next_observation, reward, terminated, truncated, info = env.step(
            action)

        next_state = np.concatenate([next_observation['observation'],
                                     next_observation['achieved_goal'], next_observation['desired_goal']], axis=0)
        total_reward += reward

        step += 1
        state = next_state
        time.sleep(0.1)
    print(total_reward)
writer.close()
