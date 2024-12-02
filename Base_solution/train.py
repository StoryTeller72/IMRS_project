import os
from collections import deque
import random
from DDPG import DDPG
import gymnasium as gym
import panda_gym
import time
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter


current_time = datetime.datetime.now()
writer = SummaryWriter("Base_solution/runs")
directory_name = f"Base_solution/DDPG_checkpoints/{current_time}"
os.makedirs(directory_name)


bath_size = 4096
episode_n = 2_000
trajectory_len = 50
start_learning = 80
checkpoint_episode_interval = 2

env = gym.make('PandaReachDense-v3')
# env = gym.make('PandaReachDense-v3')
agent = DDPG(env.observation_space['observation'].shape[0],
             env.action_space._shape[0], bath_size=bath_size, start_learning=start_learning, noise_decrease=1 / (episode_n * trajectory_len), polyak=0.005, buffer_size=15000, critic_lr=5e-4, actor_lr=5e-4, noise_scaler=0.7)

log_data = {'Total_reward': []}


cur_episode = 0
step = 0


for episode in range(episode_n):
    total_reward = 0
    observation, info = env.reset()

    for _ in range(trajectory_len):
        action = agent.get_action(observation['observation']).detach().numpy()
        next_observation, reward, terminated, truncated, info = env.step(
            action)
        total_reward += reward

        policy_loss, Q_los = agent.fit(observation['observation'], action, reward,
                                       terminated, next_observation['observation'], step)

        step += 1
        observation = next_observation
    cur_episode += 1
    log_data['Total_reward'].append(total_reward)
    if cur_episode % checkpoint_episode_interval == 0:
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
        torch.save({
            'policy_model': agent.policy.state_dict(),
            'target_policy_model': agent.policy_target.state_dict(),
            'Q_fun_model': agent.Q_fun.state_dict(),
            'Q_fun_target_model': agent.Q_fun_target.state_dict()

        }, directory_name + '/episode' + str(cur_episode))
    print(total_reward)
writer.close()
