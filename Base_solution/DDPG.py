import copy
import numpy as np
import torch
import torch.nn as nn
import random
from Noise import OUNoise
from collections import deque


class Actor(nn.Module):
    def __init__(self, observation_space_dim,  action_space_dim, layer_1_dim=512, layer_2_dim=256):
        super().__init__()
        self.layer_1 = nn.Linear(observation_space_dim, layer_1_dim)
        self.layer_2 = nn.Linear(layer_1_dim, layer_2_dim)
        self.layer_3 = nn.Linear(layer_2_dim, action_space_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        hidden = self.layer_1(input)
        hidden = self.relu(hidden)
        hidden = self.layer_2(hidden)
        hidden = self.relu(hidden)
        output = self.layer_3(hidden)
        return self.tanh(output)


class Critic(nn.Module):
    def __init__(self, observation_space_dim,  action_space_dim, layer_1_dim=512, layer_2_dim=256):
        super().__init__()
        self.layer_1 = nn.Linear(
            observation_space_dim + action_space_dim, layer_1_dim)
        self.layer_2 = nn.Linear(layer_1_dim, layer_2_dim)
        self.layer_3 = nn.Linear(layer_2_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, input):
        hidden = self.layer_1(input)
        hidden = self.relu(hidden)
        hidden = self.layer_2(hidden)
        hidden = self.relu(hidden)
        return self.layer_3(hidden)


class DDPG():
    def __init__(self, observation_space_dim,  action_space_dim, noise_decrease, start_learning=80, polyak=1e-2, bath_size=512, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, noise_scaler=1.0, mu=0, theta=0.15, sigma=0.3,  buffer_size=15_000):
        self.observation_dim = observation_space_dim
        self.action_dim = action_space_dim
        self.buffer_size = buffer_size
        self.experience_buffer = deque(maxlen=self.buffer_size)
        self.bath_size = bath_size
        self.start_learning = start_learning

        self.policy = Actor(self.observation_dim, self.action_dim)
        self.policy_target = copy.deepcopy(self.policy)
        self.Q_fun = Critic(self.observation_dim, self.action_dim)
        self.Q_fun_target = copy.deepcopy(self.Q_fun)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.polyak = polyak
        self.noise_scaler = noise_scaler
        self.noise_decrease = noise_decrease

        self.mu = mu
        self.theta = theta
        self.sigma = sigma

        self.gamma = gamma
        self.noise = OUNoise(self.action_dim, self.mu, self.theta, self.sigma)
        self.q_optimizer = torch.optim.Adam(
            self.Q_fun.parameters(), lr=critic_lr)
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=actor_lr)

    def get_action(self, state):
        pred_action = self.policy(torch.FloatTensor(state))
        action = pred_action + self.noise_scaler * \
            torch.FloatTensor(self.noise.sample())
        self.noise_scaler = max(0, self.noise_decrease - self.noise_decrease)
        return torch.clamp(action, -1.0, 1.0)

    def update(self, target_model, model, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(
                (1 - self.polyak) * target_param.data + self.polyak * param.data)

    def fit(self, state, action, reward, done, next_state, step):
        self.experience_buffer.append(
            (state, action, reward, done, next_state))

        if step > self.start_learning:
            bath = random.choices(self.experience_buffer, k=self.bath_size)
            # Get samples from experience buffer
            states, actions, rewards, done, next_states = map(
                torch.FloatTensor, zip(*bath))
            rewards = rewards.reshape(self.bath_size, 1)
            done = done.reshape(self.bath_size, 1)

            # Calculate Loss for critic (Q_function) and update
            pred_next_actions = self.policy_target(next_states)
            next_states_and_actions = torch.cat(
                (next_states, pred_next_actions), dim=1)
            targets = rewards + self.gamma * \
                (1 - done) * self.Q_fun_target(next_states_and_actions)
            states_and_action = torch.cat((states, actions), dim=1)

            Q_loss = torch.mean(
                (targets.detach() - self.Q_fun(states_and_action))**2)
            self.update(self.Q_fun_target, self.Q_fun,
                        self.q_optimizer, Q_loss)

            # Calculate Loss for actor(policy) and update
            pred_action = self.policy(states)
            states_and_pred_actions = torch.cat((states, pred_action), dim=1)
            policy_loss = -torch.mean(self.Q_fun(states_and_pred_actions))

            self.update(self.policy_target, self.policy,
                        self.policy_optimizer, policy_loss)
            return Q_loss, policy_loss
