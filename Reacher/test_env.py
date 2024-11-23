import gymnasium as gym
import time
import numpy as np

try:
    env = gym.make('Reacher-v5', reward_dist_weight=5,
                   reward_control_weight=0.01, render_mode='human', width=1020, height=800)
except (gym.error.DeprecatedEnv, gym.error.VersionNotFound) as e:
    env_id = [spec for spec in gym.envs.registry if spec.startswith(
        'Reacher-v5')][0]
    print("Reacher-v5 not found. Trying {}".format(env_id))
    env = gym.make(env_id)

for _ in range(5):
    state, info = env.reset()
    episode_over = False
    step = 0
    while not episode_over:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        state = next_state
        episode_over = terminated or truncated
        step += 1
        time.sleep(0.1)
env.close()
