import gym
from gym import spaces
from numpy.random import get_state
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3 import PPO
import torch

from config import *
from nn import *
from solver_mass_spring import SolverMassSpring

import multitask
import matplotlib.pyplot as plt
import taichi as ti
import numpy as np
import os

from taichi.lang.ops import mul, sin

np.seterr(all='raise')
torch.autograd.set_detect_anomaly(True)

class MassSpringEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(MassSpringEnv, self).__init__()
        self.act_spring = [0, 5, 6, 10, 15, 20, 21, 26, 30]
        self.action_space = spaces.MultiDiscrete([5 for _ in range(9)])
        self.observation_space = spaces.Box(low=-1, high=1, shape=(n_input_states, ), dtype=np.float32)
        self.rollout_lenth = 1000
        self.rewards = 0.
        multitask.setup_robot()
        self.t = 0

    def step(self, action):
        for k in range(batch_size):
            for i in range(9):
                act = 0.5*action[i]-1
                multitask.solver.pass_actuation(self.t, k, self.act_spring[i], np.double(act))
        multitask.solver.apply_spring_force(self.t)
        multitask.solver.advance_toi(self.t+1)
        multitask.solver.compute_center(self.t+1)
        multitask.solver.compute_height(self.t+1)
        multitask.nn_input(self.t+1, 0, 0.8, 0.2)
        # observation = multitask.input_state.to_numpy()[self.t+1, 0]
        observation = self.get_state(self.t+1)
        reward = self.get_reward()

        self.t += 1

        done = False
        if self.t == self.rollout_lenth:
            done = True
            #print(self.rewards)

        info = {}
        return observation, reward, done, info

    def get_reward(self):
        pos = multitask.solver.center.to_numpy()
        pos1 = pos[max(self.t-100, 0), 0][0]
        pos2 = pos[self.t, 0][0]
        # height = multitask.solver.height.to_numpy()[self.t, 0]
        # reward -= (0.2-height) ** 2
        # reward -= (0.1-pos) ** 2
        # reward = height/500
        reward = (pos2-pos1)
        return reward

    def get_state(self, t):
        return multitask.input_state.to_numpy()[t, 0]

    def reset(self):
        self.t = 0
        multitask.solver.clear_states(self.rollout_lenth)
        # multitask.nn_input(self.t, 0, 0.8, 0.2)
        multitask.solver.compute_center(self.t)
        multitask.solver.compute_height(self.t)
        multitask.nn_input(self.t, 0, 0.1, 0.2)
        observation = self.get_state(self.t)
        return observation
    
    def render(self, mode):
        visualizer(self.t)

class SaveBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveBestTrainingRewardCallback, self).__init__(verbose=verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)
        return True

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

log_dir = "./log/"
os.makedirs(log_dir, exist_ok=True)

env = MassSpringEnv()
# check_env(env)
# env = Monitor(env, log_dir)

policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=[dict(vf=[64, 64])])
model = PPO('MlpPolicy', env, gamma=1, learning_rate=3e-3, verbose=1, tensorboard_log=log_dir)

# callback = SaveBestTrainingRewardCallback(check_freq=50000, log_dir=log_dir)

total_step = 1000000
model.learn(total_timesteps=total_step)

# multitask.setup_robot()
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render(mode="human")
    if done:
      obs = env.reset()

env.close()