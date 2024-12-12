import numpy as np
import random
from Environment import Environment
from DDPG import DDPG
import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import time

bath_size = 128
episode_n = 10
trajectory_len = 50
start_learning = 80
checkpoint_episode_interval = 20


current_time = datetime.datetime.now()
writer = SummaryWriter("project/runs")
directory_name = f"project/DDPG_checkpoints/{current_time}"
os.makedirs(directory_name)

env = Environment('project/model', max_steps=50, render=True)
observation = env.reset()
agent = DDPG(12, 3, action_min=-1, action_max=1, bath_size=bath_size, start_learning=start_learning,
             noise_decrease=1 / (episode_n * trajectory_len), polyak=0.05, buffer_size=1_000_000, critic_lr=2e-3, actor_lr=1e-3, noise_scaler=0)


agent.load_models(
    'project/DDPG_checkpoints/best_model_1.3/episode9980')

step = 0


for episode in range(episode_n):
    observation = env.reset()
    total_reward = 0
    state = observation
    for i in range(trajectory_len):
        action = agent.get_action(state).detach().numpy()
        next_state, reward, terminated = env.step(
            action, i)

        total_reward += reward
        step += 1
        state = next_state
        time.sleep(0.1)
    print(total_reward)
