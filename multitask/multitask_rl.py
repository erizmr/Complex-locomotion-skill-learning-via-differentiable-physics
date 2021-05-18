import gym
from gym import spaces
from numpy.random import get_state
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

from taichi.lang.ops import mul, sin

class MassSpringEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(MassSpringEnv, self).__init__()
        self.act_spring = [0, 5, 6, 10, 15, 20, 21, 26, 30]
        self.action_space = spaces.Box(low=-1, high=1, shape=(9, ), dtype=np.float64)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1, ), dtype=np.float64)
        self.rollout_lenth = 1000
        multitask.setup_robot()
        self.t = 0

    def step(self, action):
        for k in range(batch_size):
            for i in range(9):
                multitask.solver.pass_actuation(self.t, k, self.act_spring[i], action[i])
        multitask.solver.apply_spring_force(self.t)
        multitask.solver.advance_toi(self.t+1)
        multitask.solver.compute_center(self.t+1)
        multitask.solver.compute_height(self.t+1)
        # multitask.nn_input(self.t+1, 0, 0.8, 0.2)
        # observation = multitask.input_state.to_numpy()[self.t+1, 0]
        reward = self.get_state()
        observation = np.array([reward])

        self.t += 1

        done = False
        if self.t == self.rollout_lenth:
            done = True
            print(reward)
            print(action)

        info = {}
        return observation, reward, done, info

    def get_state(self):
        pos = multitask.solver.center.to_numpy()
        pos1 = pos[max(self.t-100, 0), 0][0]
        pos2 = pos[self.t, 0][0]
        # height = multitask.solver.height.to_numpy()[self.t, 0]
        # reward -= (0.2-height) ** 2
        # reward -= (0.1-pos) ** 2
        # reward = height/500
        reward = (pos2-pos1)
        return reward

    def reset(self):
        self.t = 0
        multitask.solver.clear_states(self.rollout_lenth)
        # multitask.nn_input(self.t, 0, 0.8, 0.2)
        multitask.solver.compute_center(self.t)
        multitask.solver.compute_height(self.t)
        observation = np.array([self.get_state()])
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
check_env(env)
policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=[dict(vf=[64, 64])])
model = PPO('MlpPolicy', env, gamma=1, learning_rate=1e-3, verbose=1, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=2000000)

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