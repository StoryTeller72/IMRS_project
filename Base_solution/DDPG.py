import numpy
import torch
import torch.nn as nn
import random
from collections import deque
from copy import deepcopy
from torch import normal


class Actor(nn.Module):
    def __init__(self, observation_space,  action_space, layer_1_dim=512, layer_2_dim=256):
        super().__init__()
        self.layer_1 = nn.Linear(observation_space, layer_1_dim)
        self.layer_2 = nn.Linear(layer_1_dim, layer_2_dim)
        self.layer_3 = nn.Linear(layer_2_dim, action_space)
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
    def __init__(self, observation_space,  action_space, layer_1_dim=512, layer_2_dim=256):
        super().__init__()
        self.layer_1 = nn.Linear(observation_space + action_space, layer_1_dim)
        self.layer_2 = nn.Linear(layer_1_dim, layer_2_dim)
        self.layer_3 = nn.Linear(layer_2_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, input):
        hidden = self.layer_1(input)
        hidden = self.relu(hidden)
        hidden = self.layer_2(hidden)
        hidden = self.relu(hidden)
        return self.layer_3(hidden)
