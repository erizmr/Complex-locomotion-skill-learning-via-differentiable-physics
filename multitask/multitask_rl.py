import os
import torch
import gym
import numpy as np

from gym import spaces

np.seterr(all='raise')
torch.autograd.set_detect_anomaly(True)


class MassSpringEnv(gym.Env):

    def __init__(self, trainer):
        super(MassSpringEnv, self).__init__()
        self.trainer = trainer
        self.act_spring = trainer.solver.act_list
        self.max_speed = trainer.config.get_config()["process"]["max_speed"]
        self.max_height = trainer.config.get_config()["process"]["max_height"]

        # Flatten the features of all batches i.e. [1, batch_size * action_shape], [1, batch_size * obs_shape]
        max_act = np.ones(self.trainer.batch_size * len(self.act_spring), dtype=np.float64)
        max_obs = np.ones(self.trainer.batch_size * self.trainer.n_input_states, dtype=np.float64)

        self.training = True  # control which initialize function to use
        self.action_space = spaces.Box(-max_act, max_act)
        self.observation_space = spaces.Box(-max_obs, max_obs)

        self.rollout_length = trainer.max_steps
        self.rollout_times = 0
        self.rewards = 0.
        self.last_height = [0.] * self.trainer.batch_size
        # self.last_height = 0.
        self.t = 0
        self.is_output_video = self.trainer.config.get_config()["train"]["output_video_in_train"]
        self.video_dir = os.path.join(trainer.config.video_dir,
                                      "_{}_seed{}".format(trainer.robot_id, trainer.random_seed))

    def step(self, action):
        for k in range(self.trainer.batch_size):
            for i in range(len(self.act_spring)):
                self.trainer.solver.pass_actuation(self.t, k, self.act_spring[i], np.double(action[k, i]))
        # multitask.solver.pass_actuation_fast(self.t, np.array(self.act_spring, dtype=np.int32), action)
        self.trainer.solver.apply_spring_force(self.t)

        self.t += 1

        self.trainer.solver.advance_toi(self.t)
        self.trainer.solver.compute_center(self.t)
        self.trainer.solver.compute_height(self.t)
        self.trainer.nn_input(self.t, 0, self.max_speed, self.max_height)
        # observation = multitask.input_state.to_numpy()[self.t+1, 0]
        observation = self.get_state(self.t)

        # Reset he initial height?
        if self.t % 500 == 0:
            self.last_height = [0.1] * self.trainer.batch_size
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
            self.trainer.gui.clear()
            self.trainer.gui.line((0, self.trainer.config.get_config()["simulator"]["ground_height"]),
                                  (1, self.trainer.config.get_config()["simulator"]["ground_height"]),
                                  color=0x000022,
                                  radius=3)
            self.trainer.solver.draw_robot(self.trainer.gui, self.t, self.trainer.target_v)
            self.trainer.gui.show(os.path.join(save_dir, '{:04d}.png'.format(self.t)))

    # def get_reward(self):
    #     reward = 0.
    #     target_v = self.trainer.target_v[self.t, 0][0]
    #     target_h = self.trainer.target_h[self.t, 0]
    #     # if abs(target_v) > 1e-4:
    #     #     d = self.t // 500 * 500
    #     #     post = multitask.solver.center[self.t, 0][0]
    #     #     post_ = multitask.solver.center[self.t - 1, 0][0]
    #     #     for i in range(max(self.t - 100, d), self.t):
    #     #         tar = multitask.solver.center[i][0] + target_v
    #     #         pre_r = -(post_ - tar) ** 2
    #     #         now_r = -(post - tar) ** 2
    #     #         reward += (now_r - pre_r) / (max_speed ** 2) / 400.
    #     # elif target_h > max_height + 1e-4:
    #     #     height = multitask.solver.height[self.t]
    #     #     if height > self.last_height:
    #     #         d_reward = ((height - target_h) ** 2 - (self.last_height - target_h) ** 2) / (target_h ** 2)
    #     #         reward -= d_reward
    #     #         self.last_height = height
    #
    #     # Reward for moving forward
    #     d = self.t // 500 * 500
    #     post = self.trainer.solver.center[self.t, 0][0]
    #     post_ = self.trainer.solver.center[self.t - 1, 0][0]
    #     for i in range(max(self.t - 100, d), self.t):
    #         tar = self.trainer.solver.center[i][0] + target_v
    #         pre_r = -(post_ - tar) ** 2
    #         now_r = -(post - tar) ** 2
    #         reward += (now_r - pre_r) / (self.max_speed ** 2) / 400.
    #
    #     # Reward for jumping
    #     height = self.trainer.solver.height[self.t]
    #     if height > self.last_height:
    #         d_reward = ((height - target_h) ** 2 - (self.last_height - target_h) ** 2) / (target_h ** 2) * 5.0
    #         reward -= d_reward
    #         self.last_height = height
    #     return reward

    def get_reward(self):
        # reward = 0.
        reward = np.zeros(self.trainer.batch_size)
        # Reward for moving forward
        d = self.t // 500 * 500
        for k in range(self.trainer.batch_size):
            target_v = self.trainer.target_v[self.t, k][0]
            target_h = self.trainer.target_h[self.t, k]

            post = self.trainer.solver.center[self.t, k][0]
            post_ = self.trainer.solver.center[self.t - 1, k][0]
            for i in range(max(self.t - 100, d), self.t):
                tar = self.trainer.solver.center[i, k][0] + target_v
                pre_r = -(post_ - tar) ** 2
                now_r = -(post - tar) ** 2
                reward[k] += (now_r - pre_r) / (self.max_speed ** 2) / 400.

            # Reward for jumping
            height = self.trainer.solver.height[self.t, k]
            if height > self.last_height[k]:
                d_reward = ((height - target_h) ** 2 - (self.last_height[k] - target_h) ** 2) / (target_h ** 2) * 10.0
                reward[k] -= d_reward
                self.last_height[k] = height
        return sum(reward) / self.trainer.batch_size

    # def get_state(self, t):
    #     np_state = self.trainer.input_state.to_numpy()[t, 0]
    #     if np.amax(np_state) > 1. or np.amin(np_state) < -1.:
    #         # print('action range error, try to clip')
    #         np_state = np.clip(np_state, a_min=-1., a_max=1.)
    #         # print(np_state)
    #         # assert False
    #     return np_state

    def get_state(self, t):
        np_state = self.trainer.input_state.to_numpy()[t]
        if np.amax(np_state) > 1. or np.amin(np_state) < -1.:
            # print('action range error, try to clip')
            np_state = np.clip(np_state, a_min=-1., a_max=1.)
            # print(np_state)
            # assert False
        np_state = np_state.flatten()
        return np_state

    def reset(self):
        self.trainer.logger.info("reset called")
        if self.trainer.training:
            self.trainer.initialize_train(0, self.rollout_length, self.max_speed, self.max_height)
        else:
            validate_v = self.trainer.validate_targets_values['velocity']
            validate_h = self.trainer.validate_targets_values['height']
            self.trainer.logger.info(f"current max speed: {validate_v}, max height {validate_h}")
            self.trainer.initialize_validate(self.rollout_length, np.array(validate_v), np.array(validate_h))
        self.t = 0
        self.rollout_times += 1
        self.last_height = [0.1] * self.trainer.batch_size
        # self.last_height = 0.1
        self.trainer.logger.info(f'Starting rollout times: {self.rollout_times}')
        self.trainer.solver.clear_states(self.rollout_length)
        # multitask.nn_input(self.t, 0, max_speed, 0.2)
        self.trainer.solver.compute_center(self.t)
        self.trainer.solver.compute_height(self.t)
        self.las_pos = self.trainer.solver.center[0, 0][0]
        self.trainer.nn_input(self.t, 0, self.max_speed, self.max_height)
        observation = self.get_state(self.t)
        return observation

    def render(self, mode):
        # TODO: seems like lacking the second argument
        self.trainer.visualizer(self.t)