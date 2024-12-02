import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F

# import the skrl components to build the RL system
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.models.torch import DeterministicMixin, Model
from gymnasium.wrappers import RecordVideo
from gymnasium.wrappers import TimeLimit
import os
import time


class Actor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.action_layer = nn.Linear(300, self.num_actions)

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(inputs["states"]))
        x = F.relu(self.linear_layer_2(x))
        return torch.tanh(self.action_layer(x)), {}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(
            self.num_observations + self.num_actions, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.linear_layer_3 = nn.Linear(300, 1)

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(
            torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)))
        x = F.relu(self.linear_layer_2(x))
        return self.linear_layer_3(x), {}


try:
    env = gym.make('Reacher-v5', render_mode="rgb_array")
    env = RecordVideo(env, video_folder="Reacher_SCRL/videos",
                      video_length=250, disable_logger=True)
except (gym.error.DeprecatedEnv, gym.error.VersionNotFound) as e:
    env_id = [spec for spec in gym.envs.registry if spec.startswith(
        'Reacher-v5')][0]
    print("Reacher-v5 not found. Trying {}".format(env_id))
    env = gym.make(env_id)
env = wrap_env(env)

device = env.device


models = {}
models["policy"] = Actor(env.observation_space, env.action_space, device)
models["target_policy"] = Actor(
    env.observation_space, env.action_space, device)
models["critic"] = Critic(env.observation_space,
                          env.action_space, device)
models["target_critic"] = Critic(
    env.observation_space, env.action_space, device)


# configure and instantiate the agent
cfg = DDPG_DEFAULT_CONFIG.copy()
agent = DDPG(models=models)

agent.load(
    'runs/torch/Reacher/24-12-03_01-32-00-792505_DDPG/checkpoints/best_agent.pt')
agent.init()


for _ in range(5):
    state, info = env.reset()
    step = 0
    episode_over = False
    while not episode_over:
        action = agent.act(state, step, 50)
        next_state, reward, terminated, truncated, info = env.step(
            action[0].detach())
        state = next_state
        episode_over = terminated or truncated
        step += 1
env.close()
