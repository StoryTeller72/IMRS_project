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
        self.phi_max = 40
        self.phi_min = -40
        self.theta_max = 40
        self.theta_min = -40
        self.r_max = 8
        self.r_min = 5
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

        observation = np.concatenate([self.data.site('end_effector').xpos, self.data.site(
            'goal').xpos, self.data.xaxis[1], self.data.xaxis[2]])

        # observation = np.concatenate([self.data.site('end_effector').xpos, self.data.site(
        #     'goal').xpos, self.data.ctrl, self.data.body('link_1').xquat,  self.data.body('link_2').xquat])
        # observation = np.concatenate([self.data.site(
        #     'end_effector').xpos, self.data.site('goal').xpos, self.data.body('link_1').xpos, self.data.body('link_1').xquat, self.data.body('link_2').xpos, self.data.body('link_2').xquat])
        # # print(self.data.body('link_1'))
        return observation

    def generate_goal_position(self):
        r = random.uniform(self.r_min, self.r_max)
        phi = random.uniform(self.phi_min, self.phi_max)
        theta = random.uniform(self.theta_min, self.theta_max)
        return (r, phi, theta)

    def calculate_distance(self, end_effector_pos, goal_pos):
        square = np.square(goal_pos - end_effector_pos)
        sum_squares = np.sum(square)
        distance = np.sqrt(sum_squares)
        return distance

    def calculate_force(self, action):
        norm = 0.1 * np.square(np.dot(action, action))
        return norm

    def step(self, action, cur_step):
        self.data.actuator('R_1').ctrl = action[0]
        self.data.actuator('P_1').ctrl = action[1]
        self.data.actuator('R_2').ctrl = action[2]
        mujoco.mj_step(self.model, self.data)
        distance = self.calculate_distance(
            self.data.site('end_effector').xpos, self.data.site('goal').xpos)
        force = self.calculate_force(action)
        reward = -distance - force
        # print(self.data.body('link_1').xquat)
        # print(self.data.xaxis)

        # observation = np.concatenate([self.data.site(
        #     'end_effector').xpos, self.data.site('goal').xpos, self.data.body('link_1').xpos, self.data.body('link_1').xquat, self.data.body('link_2').xpos, self.data.body('link_2').xquat])
        # observation = np.concatenate([self.data.site('end_effector').xpos, self.data.site(
        #     'goal').xpos, self.data.ctrl, self.data.body('link_1').xquat,  self.data.body('link_2').xquat])
        observation = np.concatenate([self.data.site('end_effector').xpos, self.data.site(
            'goal').xpos, self.data.xaxis[1], self.data.xaxis[2]])

        self.viewer.sync()
        if abs(distance) < 2:
            return observation, reward, True

        return observation, reward, False
