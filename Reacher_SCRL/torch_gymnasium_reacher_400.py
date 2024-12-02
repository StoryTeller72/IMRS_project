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


set_seed()


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
    env = gym.make('Reacher-v5')
except (gym.error.DeprecatedEnv, gym.error.VersionNotFound) as e:
    env_id = [spec for spec in gym.envs.registry if spec.startswith(
        'Reacher-v5')][0]
    print("Reacher-v5 not found. Trying {}".format(env_id))
    env = gym.make(env_id)
env = wrap_env(env)

device = env.device


memory = RandomMemory(memory_size=25000, num_envs=env.num_envs,
                      device=device, replacement=False)


models = {}
models["policy"] = Actor(env.observation_space, env.action_space, device)
models["target_policy"] = Actor(
    env.observation_space, env.action_space, device)
models["critic"] = Critic(env.observation_space, env.action_space, device)
models["target_critic"] = Critic(
    env.observation_space, env.action_space, device)

for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

cfg = DDPG_DEFAULT_CONFIG.copy()
cfg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(
    theta=0.15, sigma=0.1, base_scale=1.0, device=device)
cfg["batch_size"] = 100
cfg["random_timesteps"] = 100
cfg["learning_starts"] = 100
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 300
cfg["experiment"]["checkpoint_interval"] = 1500
cfg["experiment"]["directory"] = "runs/torch/Reacher"
cfg['experiment']['"experiment_name"'] = 'try_bigger_bath_size'

agent = DDPG(models=models,
             memory=memory,
             cfg=cfg,
             observation_space=env.observation_space,
             action_space=env.action_space,
             device=device)

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 50_000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# start training
trainer.train()
