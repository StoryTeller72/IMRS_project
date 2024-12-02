from collections import deque
import random
from DDPG import DDPG
import gymnasium as gym
import panda_gym
import time
import datetime


bath_size = 4096
episode_n = 50
trajectory_len = 50
start_learning = 80
write_episode_interval = 10

env = gym.make('PandaReachDense-v3')
# env = gym.make('PandaReachDense-v3')
agent = DDPG(env.observation_space['observation'].shape[0],
             env.action_space._shape[0], bath_size=bath_size, start_learning=start_learning, noise_decrease=1 / (episode_n * trajectory_len), polyak=0.005, buffer_size=15000, critic_lr=5e-4, actor_lr=5e-4, noise_scaler=0.7)

log_data = {'Total_reward': []}


cur_episode = 0
step = 0
current_time = datetime.datetime.now()
f = open(f"Base_solution/runs/{current_time}.txt", "a")

for episode in range(episode_n):
    total_reward = 0
    observation, info = env.reset()

    for _ in range(trajectory_len):
        action = agent.get_action(observation['observation']).detach().numpy()
        next_observation, reward, terminated, truncated, info = env.step(
            action)
        total_reward += reward

        agent.fit(observation['observation'], action, reward,
                  terminated, next_observation['observation'], step)

        step += 1
        observation = next_observation
    cur_episode += 1
    log_data['Total_reward'].append(total_reward)
    if cur_episode % write_episode_interval == 0:
        f.write(f'{cur_episode} {max(log_data['Total_reward'])} {min(log_data["Total_reward"])} {
                sum(log_data['Total_reward']) / len(log_data["Total_reward"])}\n')
f.close()
