import gym
from gym import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
import torch

from config import *
from nn import *
from solver_mass_spring import SolverMassSpring

import multitask
import matplotlib.pyplot as plt
import taichi as ti
import numpy as np

from taichi.lang.ops import mul

class MassSpringEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(MassSpringEnv, self).__init__()
        self.action_space = spaces.Box(low=0, high=1, shape=(n_springs, ), dtype=np.float64)
        self.observation_space = spaces.Box(low=-5, high=5, shape=(n_input_states, ), dtype=np.float64)
        multitask.setup_robot()
        self.t = 0

    def step(self, action):
        for k in range(batch_size):
            for i in range(n_springs):
                multitask.solver.pass_actuation(self.t, k, i, action[i])
        multitask.solver.apply_spring_force(self.t)
        multitask.solver.advance_toi(self.t+1)
        multitask.solver.compute_center(self.t+1)
        multitask.solver.compute_height(self.t+1)
        multitask.nn_input(self.t+1, 0, 0.8, 0.2)
        # multitask.nn_input(self.t+1, 0, 0.08, 0.1)
        pos = multitask.solver.center.to_numpy()[self.t+1, 0][0]
        height = multitask.solver.height.to_numpy()[self.t+1, 0]
        observation = multitask.input_state.to_numpy()[self.t+1, 0]
        reward = self.get_reward()

        self.t += 1

        done = False
        if pos < -2 or pos > 2 or self.t == 1040:
            done = True
            print(reward)
            print(pos, height)
            print(action)

        info = {}
        return observation, reward, done, info

    def get_reward(self):
        reward = 0.
        pos = multitask.solver.center.to_numpy()[self.t, 0][0]
        height = multitask.solver.height.to_numpy()[self.t, 0]
        # reward -= (0.2-height) ** 2
        # reward -= (0.1-pos) ** 2
        reward = height
        return reward

    def reset(self):
        self.t = 0
        multitask.solver.clear_states(1040)
        multitask.nn_input(self.t, 0, 0.8, 0.2)
        multitask.solver.compute_center(self.t)
        multitask.solver.compute_height(self.t)
        pos = multitask.solver.center.to_numpy()[self.t, 0][0]
        height = multitask.solver.height.to_numpy()[self.t, 0]
        observation = multitask.input_state.to_numpy()[self.t, 0]
        return observation
    
    def render(self, mode):
        visualizer(self.t)

gui = ti.GUI(background_color=0xFFFFFF)
def visualizer(t):
    gui.clear()
    gui.line((0, multitask.ground_height), (1, multitask.ground_height),
             color=0x000022,
             radius=3)
    multitask.solver.draw_robot(gui, t, multitask.target_v)
    gui.show('video/interactive/{:04d}.png'.format(visualizer.frame))
    visualizer.frame += 1
visualizer.frame = 0

env = MassSpringEnv()
policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[dict(pi=[64, 64], vf=[64, 64])])
model = PPO('MlpPolicy', env, gamma=0.99, learning_rate=1e-2, verbose=1)
model.learn(total_timesteps=10000)

# multitask.setup_robot()
obs = env.reset()
for i in range(1040):
    action, _states = model.predict(obs, deterministic=True)
    action = action * np.random.random()
    obs, reward, done, info = env.step(action)
    env.render(mode="human")
    if done:
      obs = env.reset()

env.close()