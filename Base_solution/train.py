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
episode_n = 250
trajectory_len = 50
start_learning = 80
checkpoint_episode_interval = 20

env = gym.make('PandaReachDense-v3')

obs_shape = env.observation_space['observation'].shape[0] + \
    env.observation_space['achieved_goal'].shape[0] + \
    env.observation_space['desired_goal'].shape[0]

action_shape = env.action_space.shape[0]

action_min = env.action_space.low[0]
action_max = env.action_space.high[0]

agent = DDPG(obs_shape, action_shape, action_min=action_min, action_max=action_max, bath_size=bath_size, start_learning=start_learning,
             noise_decrease=1 / (episode_n * trajectory_len), polyak=0.05, buffer_size=10**6, critic_lr=2e-3, actor_lr=1e-3, noise_scaler=0.3)

log_data = {'Total_reward': []}


step = 0


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

        policy_loss, Q_los = agent.fit(state, action, reward,
                                       terminated, next_state, step)

        step += 1
        state = next_state

    log_data['Total_reward'].append(total_reward)
    if episode % checkpoint_episode_interval == 0:
        writer.add_scalar('Max_total_reward', max(
            log_data['Total_reward']), step)
        writer.add_scalar('Min_total_reward', min(
            log_data['Total_reward']), step)
        writer.add_scalar('Mean_total_reward', sum(
            log_data['Total_reward']) / len(log_data['Total_reward']), step)
        log_data['Total_reward'] = []
        if policy_loss and Q_los:
            writer.add_scalar('Q_loss', Q_los, step)
            writer.add_scalar('Pi_Loss', policy_loss, step)
        writer.flush()
        agent.save_models(directory_name, episode)
    print(total_reward)
writer.close()
