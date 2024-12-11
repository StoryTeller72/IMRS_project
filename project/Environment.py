import numpy as np
import time
import mujoco.viewer
import mujoco
import matplotlib.pyplot as plt
import random
import mediapy as media
import matplotlib.pyplot as plt
import mujoco.viewer
import copy


class Environment():
    def __init__(self, path, max_steps, render=False):
        self.path = path
        self.max_steps = max_steps
        self.cur_step = 0
        self.model = mujoco.MjModel.from_xml_path(self.path)
        self.data = mujoco.MjData(self.model)
        self.phi_max = 55
        self.phi_min = -55
        self.theta_max = 60
        self.theta_min = -60
        self.r_max = 10
        self.r_min = 4
        self.link_offset = 10
        self.max_steps = max_steps
        self.render = render
        self.terminated = False

        if self.render == True:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance = 70.0

    def reset(self):

        mujoco.mj_resetData(self.model, self.data)

        r, phi, theta = self.generate_goal_position()
        x_pos = r * np.cos(np.deg2rad(phi))
        y_pos = r * np.sin(np.deg2rad(phi))
        z_pos = r * np.sin(np.deg2rad(theta)) + self.link_offset

        self.model.site('goal').pos[0] = x_pos
        self.model.site('goal').pos[1] = y_pos
        self.model.site('goal').pos[2] = z_pos
        self.data.site('goal').xpos[0] = x_pos
        self.data.site('goal').xpos[1] = y_pos
        self.data.site('goal').xpos[2] = z_pos
        observation = np.concatenate([self.data.site(
            'end_effector').xpos, self.data.site('goal').xpos])
        return observation

    def generate_goal_position(self):
        r = random.uniform(self.r_min, self.r_max)
        phi = random.uniform(self.phi_min, self.phi_max)
        theta = random.uniform(self.theta_min, self.theta_max)
        return (r, phi, theta)

    def calculate_reward(self, end_effector_pos, goal_pos):
        square = np.square(goal_pos - end_effector_pos)
        sum_squares = np.sum(square)
        distance = np.sqrt(sum_squares)
        return -distance

    def step(self, action, cur_step):
        self.data.actuator('R_1').ctrl = action[0]
        self.data.actuator('P_1').ctrl = action[1]
        self.data.actuator('R_2').ctrl = action[2]
        mujoco.mj_step(self.model, self.data)
        reward = self.calculate_reward(
            self.data.site('end_effector').xpos, self.data.site('goal').xpos)

        observation = np.concatenate([self.data.site(
            'end_effector').xpos, self.data.site('goal').xpos])

        self.viewer.sync()
        if cur_step >= self.max_steps - 1:
            self.terminated = True
            self.viewer.close
        time.sleep(.1)
        return observation, reward, self.terminated
