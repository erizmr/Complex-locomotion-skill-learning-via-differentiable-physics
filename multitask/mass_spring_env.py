import os
import torch
import gym
import logging
import numpy as np
from multitask.solver_mass_spring import SolverMassSpring
from multitask.taichi_env import TaichiEnv
from gym import spaces

np.seterr(all='raise')
torch.autograd.set_detect_anomaly(True)


class MassSpringEnv(gym.Env):

    def __init__(self, config, rank=0, training=True):
        super(MassSpringEnv, self).__init__()
        self.rank = rank
        self.logger = config.get_logger(name=__name__+f"rank_{rank}")
        assert config.get_config()["robot"]["simulator"] == "mass_spring"
        self.taichi_env = TaichiEnv(config)
        self.taichi_env.setup_robot()
        self.act_spring = self.taichi_env.solver.act_list
        self.max_speed = config.get_config()["process"]["max_speed"]
        self.max_height = config.get_config()["process"]["max_height"]
        self.batch_size = config.get_config()["nn"]["batch_size"]
        self.n_input_states = config.get_config()["nn"]["n_input_states"]
        self.max_steps = config.get_config()["process"]["max_steps"]
        self.robot_id = config.get_config()["robot"]["robot_id"]
        self.random_seed = config.get_config()["train"]["random_seed"]
        self.ground_height = config.get_config()["simulator"]["ground_height"]
        self.task = config.get_config()["train"]["task"]

        # Flatten the features of all batches i.e. [1, batch_size * action_shape], [1, batch_size * obs_shape]
        max_act = np.ones(self.batch_size * len(self.act_spring), dtype=np.float64)
        max_obs = np.ones(self.batch_size * self.n_input_states, dtype=np.float64)

        self.training = training  # control which initialize function to use
        self.action_space = spaces.Box(-max_act, max_act)
        self.observation_space = spaces.Box(-max_obs, max_obs)

        self.rollout_length = self.max_steps
        self.rollout_times = 0
        self.rewards = 0.
        self.last_height = [0.] * self.batch_size
        # self.last_height = 0.
        self.t = 0
        self.is_output_video = config.get_config()["train"]["output_video_in_train"]
        self.video_dir = os.path.join(config.video_dir,
                                      "_{}_seed{}".format(self.robot_id, self.random_seed))

    def step(self, action):
        for k in range(self.batch_size):
            for i in range(len(self.act_spring)):
                self.taichi_env.solver.pass_actuation(self.t, k, self.act_spring[i], np.double(action[i]))
        # multitask.solver.pass_actuation_fast(self.t, np.array(self.act_spring, dtype=np.int32), action)
        self.taichi_env.solver.apply_spring_force(self.t)

        self.t += 1

        self.taichi_env.solver.advance_toi(self.t)
        self.taichi_env.solver.compute_center(self.t)
        self.taichi_env.solver.compute_height(self.t)
        self.taichi_env.nn_input(self.t, 0, self.max_speed, self.max_height)
        # observation = multitask.input_state.to_numpy()[self.t+1, 0]
        observation = self.get_state(self.t)

        # Reset he initial height?
        if self.t % 500 == 0:
            self.last_height = [0.1] * self.batch_size
            # self.last_height = 0.1

        reward = self.get_reward()

        if self.is_output_video:
            self.output_video()

        done = False
        if self.t == self.rollout_length - 1:
            done = True

        info = {}
        return observation, reward, done, info

    def output_video(self):
        if self.rollout_times % 50 == 1:
            save_dir = os.path.join(self.video_dir, "rl_{:04d}".format(self.rollout_times))
            os.makedirs(save_dir, exist_ok=True)
            self.taichi_env.gui.clear()
            self.taichi_env.gui.line((0, self.ground_height),
                                  (1, self.ground_height),
                                  color=0x000022,
                                  radius=3)
            self.taichi_env.solver.draw_robot(self.taichi_env.gui, self.t, self.taichi_env.target_v)
            self.taichi_env.gui.show(os.path.join(save_dir, '{:04d}.png'.format(self.t)))

    def get_reward(self):
        reward = np.zeros(self.batch_size)
        # Reward for moving forward
        d = self.t // 500 * 500
        for k in range(self.batch_size):
            target_v = self.taichi_env.target_v[self.t, k][0]
            target_h = self.taichi_env.target_h[self.t, k]

            post = self.taichi_env.solver.center[self.t, k][0]
            post_ = self.taichi_env.solver.center[self.t - 1, k][0]
            for i in range(max(self.t - 100, d), self.t):
                tar = self.taichi_env.solver.center[i, k][0] + target_v
                pre_r = -(post_ - tar) ** 2
                now_r = -(post - tar) ** 2
                reward[k] += (now_r - pre_r) / (self.max_speed ** 2) / 400.

            # Reward for jumping
            height = self.taichi_env.solver.height[self.t, k]
            if height > self.last_height[k]:
                d_reward = ((height - target_h) ** 2 - (self.last_height[k] - target_h) ** 2) / (target_h ** 2) * 10.0
                reward[k] -= d_reward
                self.last_height[k] = height
        return sum(reward) / self.batch_size

    # def get_state(self, t):
    #     np_state = self.trainer.input_state.to_numpy()[t, 0]
    #     if np.amax(np_state) > 1. or np.amin(np_state) < -1.:
    #         # print('action range error, try to clip')
    #         np_state = np.clip(np_state, a_min=-1., a_max=1.)
    #         # print(np_state)
    #         # assert False
    #     return np_state

    def get_state(self, t):
        np_state = self.taichi_env.input_state.to_numpy()[t]
        if np.amax(np_state) > 1. or np.amin(np_state) < -1.:
            # print('action range error, try to clip')
            np_state = np.clip(np_state, a_min=-1., a_max=1.)
            # print(np_state)
            # assert False
        np_state = np_state.flatten()
        return np_state

    def clear_losses(self):
        self.taichi_env.loss[None] = 0.
        for l in self.taichi_env.losses:
            l[None] = 0.

        # Clear all batch losses
        for k in range(self.batch_size):
            self.taichi_env.loss_batch[k] = 0.
        for l in self.taichi_env.losses_batch:
            for k in range(self.batch_size):
                l[k] = 0.

    def compute_losses(self):
        self.taichi_env.get_loss(self.max_steps+1, loss_enable=self.task)

    def get_losses(self):
        return self.taichi_env.loss[None], self.taichi_env.loss_dict

    def reset(self):
        print(f"Rank:{self.rank} reset called")
        self.logger.info(f"Rank:{self.rank} reset called")
        if self.training:
            self.taichi_env.initialize_train(0, self.rollout_length, self.max_speed, self.max_height)
        else:
            self.clear_losses()
            self.taichi_env.solver.clear_states(self.max_steps)
            validate_v = self.taichi_env.validate_targets_values['velocity'][self.rank]
            validate_h = self.taichi_env.validate_targets_values['height'][self.rank]
            self.taichi_env.logger.info(f"Rank: {self.rank}, current max speed: {validate_v}, max height {validate_h}")
            print(f"Rank: {self.rank}, current max speed: {validate_v}, max height {validate_h}")
            self.taichi_env.initialize_validate(self.rollout_length, np.array([validate_v]), np.array([validate_h]))
        self.t = 0
        self.rollout_times += 1
        self.last_height = [0.1] * self.batch_size
        # self.last_height = 0.1
        self.logger.info(f'Starting rollout times: {self.rollout_times}')
        self.taichi_env.solver.clear_states(self.rollout_length)
        # multitask.nn_input(self.t, 0, max_speed, 0.2)
        self.taichi_env.solver.compute_center(self.t)
        self.taichi_env.solver.compute_height(self.t)
        self.las_pos = self.taichi_env.solver.center[0, 0][0]
        self.taichi_env.nn_input(self.t, 0, self.max_speed, self.max_height)
        observation = self.get_state(self.t)
        return observation

    def render(self, mode):
        # TODO: seems like lacking the second argument
        self.taichi_env.visualizer(self.t)