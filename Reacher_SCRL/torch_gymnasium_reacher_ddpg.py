import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F

# import the skrl components to build the RL system
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise
from skrl.trainers.torch import SequentialTrainer
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils import set_seed


# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


# define models (deterministic models) using mixin
class Actor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, self.num_actions),
                                 nn.Tanh())

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}


# load and wrap the gymnasium environment.
# note: the environment version may change depending on the gymnasium version
try:
    env = gym.make('Reacher-v5')
except (gym.error.DeprecatedEnv, gym.error.VersionNotFound) as e:
    env_id = [spec for spec in gym.envs.registry if spec.startswith(
        'Reacher-v5')][0]
    print("Reacher-v5 not found. Trying {}".format(env_id))
    env = gym.make(env_id)
env = wrap_env(env)

device = env.device


# instantiate a memory as experience replay
memory = RandomMemory(memory_size=2048, num_envs=env.num_envs,
                      device=device)


# instantiate the agent's models (function approximators).
# DDPG requires 4 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#models
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
cfg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(
    theta=0.15, sigma=0.1, base_scale=1.0, device=device)
cfg["gradient_steps"] = 1
cfg["batch_size"] = 1024
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 5e-3
cfg["critic_learning_rate"] = 5e-3
cfg["random_timesteps"] = 80
cfg["learning_starts"] = 2000
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {
    "size": env.observation_space, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 800
cfg["experiment"]["checkpoint_interval"] = 500
cfg["experiment"]["directory"] = "runs/torch/Reacher"

agent = DDPG(models=models,
             memory=memory,
             cfg=cfg,
             observation_space=env.observation_space,
             action_space=env.action_space,
             device=device)

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 20_000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# start training
trainer.train()
