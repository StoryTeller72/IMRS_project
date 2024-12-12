import numpy as np
import random
from Environment import Environment
from DDPG import DDPG
import datetime
from torch.utils.tensorboard import SummaryWriter
import os


bath_size = 128
episode_n = 10_000
trajectory_len = 100
start_learning = 80
checkpoint_episode_interval = 20


current_time = datetime.datetime.now()
writer = SummaryWriter("project/runs")
directory_name = f"project/DDPG_checkpoints/{current_time}"
os.makedirs(directory_name)

env = Environment('project/model', max_steps=50, render=True)
observation = env.reset()
agent = DDPG(12, 3, action_min=-1, action_max=1, bath_size=bath_size, start_learning=start_learning,
             noise_decrease=1 / (episode_n * trajectory_len), polyak=0.05, buffer_size=1_000_000, critic_lr=2e-3, actor_lr=1e-3, noise_scaler=0.3)


step = 0
log_data = {'Total_reward': []}


for episode in range(episode_n):
    observation = env.reset()
    terminated = False
    # print(observation)
    total_reward = 0
    state = observation
    for i in range(trajectory_len):
        action = agent.get_action(state).detach().numpy()
        next_state, reward, terminated = env.step(
            action, i)

        total_reward += reward

        policy_loss, Q_los = agent.fit(state, action, reward,
                                       terminated, next_state, step)
        step += 1
        state = next_state
        if terminated:
            break
    if step > start_learning:
        log_data['Total_reward'].append(total_reward)

    if episode and episode % checkpoint_episode_interval == 0:
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
