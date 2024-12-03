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

from skrl.envs.wrappers.torch import wrap_env
from skrl.models.torch import DeterministicMixin, Model
from gymnasium.wrappers import RecordVideo
from gymnasium.wrappers import TimeLimit

current_time = datetime.datetime.now()

bath_size = 128
episode_n = 5
trajectory_len = 50
start_learning = 80


def trigger(t): return t % 50 == 0


env = gym.make('PandaReachDense-v3', render_mode='rgb_array')
env = RecordVideo(env, video_folder="Base_solution/videos", step_trigger=trigger, video_length=episode_n * trajectory_len,
                  disable_logger=True, name_prefix=current_time)

obs_shape = env.observation_space['observation'].shape[0] + \
    env.observation_space['achieved_goal'].shape[0] + \
    env.observation_space['desired_goal'].shape[0]

action_shape = env.action_space.shape[0]

action_min = env.action_space.low[0]
action_max = env.action_space.high[0]

agent = DDPG(obs_shape, action_shape, action_min=action_min, action_max=action_max, bath_size=bath_size, start_learning=start_learning,
             noise_decrease=1 / (episode_n * trajectory_len), polyak=0.05, buffer_size=10**6, critic_lr=2e-3, actor_lr=1e-3, noise_scaler=0.3)

cur_episode = 0
step = 0

agent.load_models(
    'Base_solution/DDPG_checkpoints/200_episodes/episode240')

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
    print(total_reward)
