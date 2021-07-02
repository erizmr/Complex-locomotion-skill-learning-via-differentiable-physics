

import gym
from gym import spaces
import torch

# from multitask.nn import *
# from multitask.solver_mass_spring import SolverMassSpring

import numpy as np
import os
# import shutil
# from taichi.lang.ops import mul, sin

np.seterr(all='raise')
torch.autograd.set_detect_anomaly(True)


class MassSpringEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, trainer):
        super(MassSpringEnv, self).__init__()
        self.trainer = trainer
        self.act_spring = trainer.solver.act_list
        self.max_speed = trainer.config.get_config()["process"]["max_speed"]
        self.max_height = trainer.config.get_config()["process"]["max_height"]

        max_act = np.ones(len(self.act_spring), dtype=np.float64)
        max_obs = np.ones(self.trainer.n_input_states, dtype=np.float64)
        self.action_space = spaces.Box(-max_act, max_act)
        self.observation_space = spaces.Box(-max_obs, max_obs)
        self.rollout_length = trainer.max_steps
        self.rollout_times = 0
        self.rewards = 0.
        self.last_height = 0.
        self.t = 0
        self.is_output_video = self.trainer.config.get_config()["train"]["output_video"]
        self.video_dir = os.path.join(trainer.config.video_dir, "_{}_seed{}".format(trainer.robot_id, trainer.random_seed))

    def step(self, action):
        for k in range(self.trainer.batch_size):
            for i in range(len(self.act_spring)):
                self.trainer.solver.pass_actuation(self.t, k, self.act_spring[i], np.double(action[i]))
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
            self.last_height = 0.1

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
            os.makedirs(save_dir, exist_ok = True)
            self.trainer.gui.clear()
            self.trainer.gui.line((0, self.trainer.config.get_config()["simulator"]["ground_height"]),
                                  (1, self.trainer.config.get_config()["simulator"]["ground_height"]),
                     color=0x000022,
                     radius=3)
            self.trainer.solver.draw_robot(self.trainer.gui, self.t, self.trainer.target_v)
            self.trainer.gui.show(os.path.join(save_dir, '{:04d}.png'.format(self.t)))

    def get_reward(self):
        reward = 0.
        target_v = self.trainer.target_v[self.t, 0][0]
        target_h = self.trainer.target_h[self.t, 0]
        # if abs(target_v) > 1e-4:
        #     d = self.t // 500 * 500
        #     post = multitask.solver.center[self.t, 0][0]
        #     post_ = multitask.solver.center[self.t - 1, 0][0]
        #     for i in range(max(self.t - 100, d), self.t):
        #         tar = multitask.solver.center[i][0] + target_v
        #         pre_r = -(post_ - tar) ** 2
        #         now_r = -(post - tar) ** 2
        #         reward += (now_r - pre_r) / (max_speed ** 2) / 400.
        # elif target_h > max_height + 1e-4:
        #     height = multitask.solver.height[self.t]
        #     if height > self.last_height:
        #         d_reward = ((height - target_h) ** 2 - (self.last_height - target_h) ** 2) / (target_h ** 2)
        #         reward -= d_reward
        #         self.last_height = height

        # Reward for moving forward
        d = self.t // 500 * 500
        post = self.trainer.solver.center[self.t, 0][0]
        post_ = self.trainer.solver.center[self.t - 1, 0][0]
        for i in range(max(self.t - 100, d), self.t):
            tar = self.trainer.solver.center[i][0] + target_v
            pre_r = -(post_ - tar) ** 2
            now_r = -(post - tar) ** 2
            reward += (now_r - pre_r) / (self.max_speed ** 2) / 400.

        # Reward for jumping
        height = self.trainer.solver.height[self.t]
        if height > self.last_height:
            d_reward = ((height - target_h) ** 2 - (self.last_height - target_h) ** 2) / (target_h ** 2)
            reward -= d_reward
            self.last_height = height
        return reward

    def get_state(self, t):
        np_state = self.trainer.input_state.to_numpy()[t, 0]
        if np.amax(np_state) > 1. or np.amin(np_state) < -1.:
            print('action range error, try to clip')
            np_state = np.clip(np_state, a_min=-1., a_max=1.)
            print(np_state)
            # assert False
        return np_state

    def reset(self):
        self.trainer.logger.info("reset called")
        self.trainer.initialize_train(0, self.rollout_length, self.max_speed, self.max_height)
        self.t = 0
        self.rollout_times += 1
        self.last_height = 0.1
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

# class SaveBestTrainingRewardCallback(BaseCallback):
#     def __init__(self, check_freq: int, log_dir: str, verbose=1):
#         super(SaveBestTrainingRewardCallback, self).__init__(verbose=verbose)
#         self.check_freq = check_freq
#         self.log_dir = log_dir
#         self.best_mean_reward = -np.inf
#         self.models_dir = os.path.join(log_dir, "rl_robot_{}".format(config.robot_id))
#         self.save_path = os.path.join(self.models_dir, "best_model")
#         if os.path.exists(self.models_dir):
#             shutil.rmtree(self.models_dir)
#         os.makedirs(self.models_dir, exist_ok = True)
#
#     def _init_callback(self) -> None:
#         if self.save_path is not None:
#             os.makedirs(self.save_path, exist_ok=True)
#
#     def _on_step(self) -> bool:
#         if self.n_calls % self.check_freq == 0:
#             x, y = ts2xy(load_results(self.log_dir), 'timesteps')
#             if len(x) > 0:
#                 mean_reward = np.mean(y[-100:])
#                 if self.verbose > 0:
#                     print("Num timesteps: {}".format(self.num_timesteps))
#                     print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
#
#                 save_path = os.path.join(self.models_dir, "model_{}".format(self.num_timesteps // 1000))
#                 #print("Saving model to {}".format(save_path))
#                 self.model.save(save_path)
#
#                 if mean_reward > self.best_mean_reward:
#                     self.best_mean_reward = mean_reward
#                     if self.verbose > 0:
#                         print("Saving new best model to {}".format(self.save_path))
#                     self.model.save(self.save_path)
#         return True
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
# if __name__ == '__main__':
#     import sys
#     robot_id = sys.argv[1]
#     gui = ti.GUI(background_color=0xFFFFFF, show_gui = False)
#     #visualizer.frame = 0
#     log_dir = "./log"
#     video_dir = "video/robot_{}".format(config.robot_id)
#     if os.path.exists(video_dir):
#         shutil.rmtree(video_dir)
#     os.makedirs(video_dir, exist_ok = True)
#     os.makedirs(log_dir, exist_ok = True)
#
#     multitask.setup_robot()
#     env = MassSpringEnv(multitask.solver.act_list, video_dir = video_dir)
#     # check_env(env)
#     env = Monitor(env, log_dir)
#
#     policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=[64])
#     model = None
#     load_path = "log/rl_robot_{}".format(robot_id)
#     if os.path.exists(load_path):
#         name_list = os.listdir(load_path)
#         t = 0
#         for name in name_list:
#             if name[:6] == 'model_' and name[-4:] == ".zip":
#                 t = max(t, int(name[6:-4]))
#         model = PPO.load(os.path.join(load_path, "model_{}.zip".format(t)), env)
#         env.env.rollout_times = t
#     else:
#         model = PPO('MlpPolicy', env, gamma=1, learning_rate=3e-3, verbose=1, tensorboard_log=log_dir, policy_kwargs = policy_kwargs, batch_size = 64, device = "cuda")
#
#     callback = SaveBestTrainingRewardCallback(check_freq=50000, log_dir=log_dir)
#
#     total_step = 200000000
#     model.learn(total_timesteps=total_step, callback=callback)
#
#     # multitask.setup_robot()
#     #multitask.initialize_validate(1000, 0.08, 0.1)
#     '''
#     obs = env.reset()
#     os.makedirs("interactive3", exist_ok=True)
#     for i in range(1000):
#         action, _states = model.predict(obs, deterministic=True)
#         obs, reward, done, info = env.step(action)
#         if i % 5 == 0:
#             env.render(mode="human")
#         if done:
#             obs = env.reset()
#
#     env.close()
#     '''