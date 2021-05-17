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
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float64)
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
        # multitask.nn_input(self.t+1, 0, 0.08, 0.1)
        pos = multitask.solver.center.to_numpy()[self.t+1, 0][0]
        height = multitask.solver.height.to_numpy()[self.t+1, 0]
        observation = np.array([pos, height])
        reward = self.get_reward()

        self.t += 1

        done = False
        if self.t > 100:
            done = True
            print(reward)
            print(observation)

        info = {}
        return observation, reward, done, info

    def get_reward(self):
        reward = 0.
        pos = multitask.solver.center.to_numpy()[self.t, 0][0]
        height = multitask.solver.height.to_numpy()[self.t, 0]
        reward -= (0.2-height) ** 2
        reward -= (0.9-pos) ** 2
        return reward

    def reset(self):
        self.t = 0
        multitask.solver.clear_states(100)
        multitask.solver.compute_center(self.t)
        multitask.solver.compute_height(self.t)
        pos = multitask.solver.center.to_numpy()[self.t, 0][0]
        height = multitask.solver.height.to_numpy()[self.t, 0]
        observation = np.array([pos, height])
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
policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[dict(pi=[128], vf=[64, 64])])
model = PPO('MlpPolicy', env)
model.learn(total_timesteps=100)

# multitask.setup_robot()
obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    action = action * np.random.random()
    obs, reward, done, info = env.step(action)
    env.render(mode="human")
    if done:
      obs = env.reset()

env.close()