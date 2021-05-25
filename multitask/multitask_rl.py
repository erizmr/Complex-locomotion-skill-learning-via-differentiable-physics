import config

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
import shutil

from taichi.lang.ops import mul, sin

np.seterr(all='raise')
torch.autograd.set_detect_anomaly(True)

class MassSpringEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, act_list, video_dir):
        super(MassSpringEnv, self).__init__()
        self.act_spring = act_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.act_spring), ), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(n_input_states, ), dtype=np.float32)
        self.rollout_length = 1000
        self.rollout_times = 0
        self.rewards = 0.
        self.last_height = 0.
        self.t = 0
        self.video_dir = video_dir

    def step(self, action):
        for k in range(batch_size):
            for i in range(len(self.act_spring)):
                multitask.solver.pass_actuation(self.t, k, self.act_spring[i], np.double(action[i]))
        # multitask.solver.pass_actuation_fast(self.t, np.array(self.act_spring, dtype=np.int32), action)
        multitask.solver.apply_spring_force(self.t)

        self.t += 1
        
        multitask.solver.advance_toi(self.t)
        multitask.solver.compute_center(self.t)
        multitask.solver.compute_height(self.t)
        multitask.nn_input(self.t, 0, max_speed, max_height)
        # observation = multitask.input_state.to_numpy()[self.t+1, 0]
        observation = self.get_state(self.t)

        if self.t % 500 == 0:
            self.last_height = 0.1

        reward = self.get_reward()
        if self.rollout_times % 100 == 1:
            save_dir = os.path.join(self.video_dir, "rl_{:04d}".format(self.rollout_times))
            os.makedirs(save_dir, exist_ok = True)
            multitask.gui.clear()
            multitask.gui.line((0, multitask.ground_height), (1, multitask.ground_height),
                     color=0x000022,
                     radius=3)
            multitask.solver.draw_robot(multitask.gui, self.t, multitask.target_v)
            multitask.gui.show(os.path.join(save_dir, '{:04d}.png'.format(self.t)))
        done = False
        if self.t == self.rollout_length:
            done = True

        info = {}
        return observation, reward, done, info

    def get_reward(self):
        reward = 0.
        target_v = multitask.target_v[self.t, 0][0]
        target_h = multitask.target_h[self.t, 0]
        if abs(target_v) > 1e-4:
            d = self.t // 500 * 500
            post = multitask.solver.center[self.t, 0][0]
            post_ = multitask.solver.center[self.t - 1, 0][0]
            for i in range(max(self.t - 100, d), self.t):
                tar = multitask.solver.center[i][0] + target_v
                pre_r = -(post_ - tar) ** 2 
                now_r = -(post - tar) ** 2
                reward += (now_r - pre_r) / (max_speed ** 2) / 400.
        elif target_h > max_height + 1e-4:
            height = multitask.solver.height[self.t]
            if height > self.last_height:
                d_reward = ((height - target_h) ** 2 - (self.last_height - target_h) ** 2) / (target_h ** 2)
                reward -= d_reward
                self.last_height = height
        return reward

    def get_state(self, t):
        return multitask.input_state.to_numpy()[t, 0]

    def reset(self):
        multitask.initialize_train(0, self.rollout_length, max_speed, max_height)
        self.t = 0
        self.rollout_times += 1
        self.last_height = 0.1
        print('Starting rollout times: ', self.rollout_times)
        multitask.solver.clear_states(self.rollout_length)
        # multitask.nn_input(self.t, 0, max_speed, 0.2)
        multitask.solver.compute_center(self.t)
        multitask.solver.compute_height(self.t)
        self.las_pos = multitask.solver.center[0, 0][0]
        multitask.nn_input(self.t, 0, max_speed, max_height)
        observation = self.get_state(self.t)
        return observation
    
    def render(self, mode):
        visualizer(self.t)

class SaveBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveBestTrainingRewardCallback, self).__init__(verbose=verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.best_mean_reward = -np.inf
        self.models_dir = os.path.join(log_dir, "rl_robot_{}".format(config.robot_id))
        self.save_path = os.path.join(self.models_dir, "best_model")
        if os.path.exists(self.models_dir):
            shutil.rmtree(self.models_dir)
        os.makedirs(self.models_dir, exist_ok = True)

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

                save_path = os.path.join(self.models_dir, "model_{}".format(self.num_timesteps // 1000))
                #print("Saving model to {}".format(save_path))
                self.model.save(save_path)

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)
        return True
'''
def visualizer(t):
    gui.clear()
    gui.line((0, multitask.ground_height), (1, multitask.ground_height),
             color=0x000022,
             radius=3)
    multitask.solver.draw_robot(gui, t, multitask.target_v)
    gui.show('video/interactive3/{:04d}.png'.format(visualizer.frame))
    visualizer.frame += 1
'''
if __name__ == '__main__':
    import sys
    robot_id = sys.argv[1]
    gui = ti.GUI(background_color=0xFFFFFF, show_gui = False)
    #visualizer.frame = 0
    log_dir = "./log"
    video_dir = "video/robot_{}".format(config.robot_id)
    if os.path.exists(video_dir):
        shutil.rmtree(video_dir)
    os.makedirs(video_dir, exist_ok = True)
    os.makedirs(log_dir, exist_ok = True)

    multitask.setup_robot()
    env = MassSpringEnv(multitask.solver.act_list, video_dir = video_dir)
    # check_env(env)
    env = Monitor(env, log_dir)

    policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=[64])
    model = PPO('MlpPolicy', env, gamma=1, learning_rate=3e-3, verbose=1, tensorboard_log=log_dir, policy_kwargs = policy_kwargs, batch_size = 64)

    callback = SaveBestTrainingRewardCallback(check_freq=50000, log_dir=log_dir)

    total_step = 200000000
    model.learn(total_timesteps=total_step, callback=callback)

    # multitask.setup_robot()
    #multitask.initialize_validate(1000, 0.08, 0.1)
    '''
    obs = env.reset()
    os.makedirs("interactive3", exist_ok=True)
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if i % 5 == 0:
            env.render(mode="human")
        if done:
            obs = env.reset()

    env.close()
    '''