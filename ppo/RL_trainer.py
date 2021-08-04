import glob
import os
import torch
import itertools
from collections import deque
import numpy as np
import taichi as ti

from collections import defaultdict
from multitask.base_trainer import BaseTrainer
from ppo.a2c_ppo_acktr import algo, utils
from ppo.a2c_ppo_acktr.algo import gail
from ppo.a2c_ppo_acktr.envs import make_env, make_vec_envs
from ppo.a2c_ppo_acktr.model import Policy
from ppo.a2c_ppo_acktr.storage import RolloutStorage
# from evaluation import evaluate
from multitask.hooks import Checkpointer, InfoPrinter, Timer, MetricWriter
from logger import TensorboardWriter
from util import MetricTracker


class CheckpointerRL(Checkpointer):
    def __init__(self, save_path):
        super(CheckpointerRL, self).__init__(save_path)

    def after_step(self):
        if (self.trainer.iter % self.trainer.save_interval == 0 or
            self.trainer.iter == self.trainer.num_updates - 1) and \
                self.save_path != "":

            torch.save([
                self.trainer.actor_critic,
                getattr(utils.get_vec_normalize(self.trainer.envs), 'obs_rms',
                        None)
            ],
                       os.path.join(
                           self.save_path, self.trainer.args.env_name +
                           str(self.trainer.iter) + ".pt"))
            self.trainer.logger.info("save {}......".format(self.trainer.iter))


class InfoPrinterRL(InfoPrinter):
    def __init__(self):
        super(InfoPrinterRL, self).__init__()

    def after_step(self):
        if self.trainer.iter % self.trainer.args.log_interval == 0 and len(
                self.trainer.episode_rewards) > 1:
            total_num_steps = (
                self.trainer.iter + 1
            ) * self.trainer.args.num_processes * self.trainer.args.num_steps
            self.trainer.logger.info(
                "Iterations: {}, num timesteps {}, FPS {} \n "
                "Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                "mean reward v {}, mean reward h {}"
                .format(
                    self.trainer.iter, total_num_steps,
                    int(total_num_steps / (self.trainer.timer.step_end -
                                           self.trainer.timer.step_start)),
                    len(self.trainer.episode_rewards),
                    np.mean(self.trainer.episode_rewards),
                    np.median(self.trainer.episode_rewards),
                    np.min(self.trainer.episode_rewards),
                    np.max(self.trainer.episode_rewards),
                    np.mean(np.array(self.trainer.reward_v)),
                    np.mean(np.array(self.trainer.reward_h))
                ))  #, dist_entropy, value_loss, action_loss))


class RLTrainer(BaseTrainer):
    def __init__(self, args, config):
        super(RLTrainer, self).__init__(args, config)
        self.args = args
        self.config = config
        self.device = torch.device("cuda:0" if args.cuda else "cpu")
        self.num_processes = args.num_processes
        self.save_interval = args.save_interval // self.num_processes
        self.max_steps = config.get_config()["process"]["max_steps"]
        self.ground_height = config.get_config()["simulator"]["ground_height"]
        self.task = config.get_config()["train"]["task"]
        self.envs = make_vec_envs(config, args.env_name, args.seed,
                                  args.num_processes, args.gamma, self.device,
                                  False, training=args.train, interactive=False)

        self.actor_critic = Policy(
            self.envs.observation_space.shape,
            self.envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy})
        self.actor_critic.to(self.device)

        # Define the RL algorithm
        self.agent = None
        self._select_algorithm()

        # Whether GAIL
        self.discr = None
        self.gail_train_loader = None
        self._gail()

        self.rollouts = RolloutStorage(
            args.num_steps, args.num_processes,
            self.envs.observation_space.shape, self.envs.action_space,
            self.actor_critic.recurrent_hidden_state_size)

        self.obs = self.envs.reset()
        self.rollouts.to(self.device)

        self.episode_rewards_len = 10
        self.episode_rewards = deque(maxlen=self.episode_rewards_len)
        # RL Training epochs
        self.num_updates = int(
            args.num_env_steps) // args.num_steps // args.num_processes

        model_save_path = self.config.model_dir
        self.checkpointer = CheckpointerRL(model_save_path)
        self.info_printer = InfoPrinterRL()
        self.timer = Timer()
        self.metric_writer = MetricWriter()

        # Metrics to tracking during training
        self.metrics_train = ["mean_reward", "max_reward", "min_reward", "reward_v", "reward_h"]
        self.metrics_validation = ["task_loss", "velocity_loss", "height_loss"]

        self.register_hooks([
            self.timer, self.checkpointer, self.info_printer,
            self.metric_writer
        ])

        self.validate_targets = self.config.get_config()["validation"]
        _to_remove = []
        for k in self.validate_targets.keys():
            if k not in self.config.get_config()["train"]["task"]:
                _to_remove.append(k)
        for k in _to_remove:
            self.validate_targets.pop(k, None)

        self.reward_v = []
        self.reward_h = []

    def _select_algorithm(self):
        args = self.args
        if args.algo == 'a2c':
            self.agent = algo.A2C_ACKTR(self.actor_critic,
                                        args.value_loss_coef,
                                        args.entropy_coef,
                                        lr=args.lr,
                                        eps=args.eps,
                                        alpha=args.alpha,
                                        max_grad_norm=args.max_grad_norm)
        elif args.algo == 'ppo':
            self.agent = algo.PPO(self.actor_critic,
                                  args.clip_param,
                                  args.ppo_epoch,
                                  args.num_mini_batch,
                                  args.value_loss_coef,
                                  args.entropy_coef,
                                  lr=args.lr,
                                  eps=args.eps,
                                  max_grad_norm=args.max_grad_norm)
        elif args.algo == 'acktr':
            self.agent = algo.A2C_ACKTR(self.actor_critic,
                                        args.value_loss_coef,
                                        args.entropy_coef,
                                        acktr=True)
        else:
            raise NotImplementedError

    def _gail(self):
        if self.args.gail:
            assert len(self.envs.observation_space.shape) == 1
            self.discr = gail.Discriminator(
                self.envs.observation_space.shape[0] +
                self.envs.action_space.shape[0], 100, self.device)
            file_name = os.path.join(
                self.args.gail_experts_dir,
                "trajs_{}.pt".format(self.args.env_name.split('-')[0].lower()))

            expert_dataset = gail.ExpertDataset(file_name,
                                                num_trajectories=4,
                                                subsample_frequency=20)
            drop_last = len(expert_dataset) > self.args.gail_batch_size
            self.gail_train_loader = torch.utils.data.DataLoader(
                dataset=expert_dataset,
                batch_size=self.args.gail_batch_size,
                shuffle=True,
                drop_last=drop_last)

    def run_step(self):
        if self.args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                self.agent.optimizer, self.iter, self.num_updates,
                self.agent.optimizer.lr
                if self.args.algo == "acktr" else self.args.lr)

        for step in range(self.args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                    self.rollouts.obs[step],
                    self.rollouts.recurrent_hidden_states[step],
                    self.rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = self.envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    self.episode_rewards.append(info['episode']['r'])
                    self.reward_v = np.sum(np.array(self.envs.get_attr("reward_v")), axis=1)
                    self.reward_h = np.sum(np.array(self.envs.get_attr("reward_h")), axis=1)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            self.rollouts.insert(obs, recurrent_hidden_states, action,
                                 action_log_prob, value, reward, masks,
                                 bad_masks)

        with torch.no_grad():
            next_value = self.actor_critic.get_value(
                self.rollouts.obs[-1],
                self.rollouts.recurrent_hidden_states[-1],
                self.rollouts.masks[-1]).detach()

        if self.args.gail:
            if self.iter >= 10:
                self.envs.venv.eval()

            gail_epoch = self.args.gail_epoch
            if self.iter < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                self.discr.update(self.gail_train_loader, self.rollouts,
                                  utils.get_vec_normalize(self.envs)._obfilt)

            for step in range(self.args.num_steps):
                self.rollouts.rewards[step] = self.discr.predict_reward(
                    self.rollouts.obs[step], self.rollouts.actions[step],
                    self.args.gamma, self.rollouts.masks[step])

        self.rollouts.compute_returns(next_value, self.args.use_gae,
                                      self.args.gamma, self.args.gae_lambda,
                                      self.args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = self.agent.update(
            self.rollouts)
        self.rollouts.after_update()

        # Write to tensorboard
        self.metric_writer.writer.set_step(step=self.iter - 1)
        self.metric_writer.train_metrics.update("mean_reward",
                                                np.mean(self.episode_rewards))
        self.metric_writer.train_metrics.update("max_reward",
                                                np.amax(self.episode_rewards))
        self.metric_writer.train_metrics.update("min_reward",
                                                np.amin(self.episode_rewards))
        self.metric_writer.train_metrics.update("reward_v",
                                                np.mean(self.reward_v))
        self.metric_writer.train_metrics.update("reward_h",
                                                np.mean(self.reward_h))


    def evaluate(self, exp_folder):
        # import glob
        # exp_folders = glob.glob(os.path.join(exp_folder, "*"))
        # exp_folders = sorted(exp_folders, key=os.path.getmtime)
        # paths_to_evaluate = []
        # # Check whether this experiment has been evaluated before
        # for ef in exp_folders:
        #     if len(os.listdir(os.path.join(ef, "validation"))) == 0:
        #         paths_to_evaluate.append(ef)
        # print(f"All experiments to evaluate {paths_to_evaluate}")
        # for ef in paths_to_evaluate:
        output_folder = os.path.join(exp_folder, 'validation')
        os.makedirs(output_folder, exist_ok=True)
        evaluator_writer = MetricTracker(*[],
                                         writer=TensorboardWriter(
                                             output_folder,
                                             self.logger,
                                             enabled=True))
        self._evaluate(exp_folder=exp_folder,
                       evaluator_writer=evaluator_writer)
        self.envs.close()

        # self._evaluate(exp_folder=exp_folder, evaluator_writer=evaluator_writer)

    def _evaluate(self,
                  exp_folder,
                  evaluator_writer,
                  output_video=False,
                  show_gui=False):
        training_num_processes = self.config.get_config()["train"]["num_processes"]
        print(f'Training processes: {training_num_processes}')
        # training_save_interval = self.config.get_config()["train"]["save_interval"]
        metric_writer = evaluator_writer
        model_folder = os.path.join(exp_folder, 'models')
        video_path = os.path.join(exp_folder, 'video')

        def visualizer(t, batch_rank, folder, output_video):
            gui.clear()
            gui.line((0, self.ground_height), (1, self.ground_height),
                     color=0x000022,
                     radius=3)
            self.solver.draw_robot(gui, batch_rank, t, self.target_v)
            if output_video:
                gui.show('{}/{:04d}.png'.format(folder, t))
            else:
                gui.show()

        gui = ti.GUI(background_color=0xFFFFFF, show_gui=show_gui)

        # Construct the validation matrix i.e., combinations of all validation targets
        targets_keys = []
        targets_values = []
        for k, v in self.validate_targets.items():
            targets_keys.append(k)
            targets_values.append(v)
        validation_matrix = list(itertools.product(*targets_values))
        self.logger.info(f"Validation targets: {targets_keys} "
                         f"Validation Matrix: {validation_matrix}")
        suffix = []
        for element in validation_matrix:
            s_base = ""
            for i, name in enumerate(targets_keys):
                s_base += f"_{name}_{element[i]}"
            suffix.append(s_base)

        all_model_names = glob.glob(os.path.join(model_folder, "*.pt"))
        all_model_names = sorted(all_model_names, key=os.path.getmtime)
        print("All models: ", all_model_names)
        # for iter_num in range(0, self.max_iter, self.save_interval):
        #     model_name = self.args.env_name + str(iter_num) + ".pt"
        for iter_num, load_path in enumerate(all_model_names):
            iter_num *= self.args.save_interval # // training_num_processes
            print(f"Iter num: {iter_num}")
            # load_path = os.path.join(model_folder, model_name)
            [actor_critic, obs_rms] = torch.load(load_path)
            self.logger.info(f"load {load_path}")

            vec_norm = utils.get_vec_normalize(self.envs)
            if vec_norm is not None:
                vec_norm.eval()
                vec_norm.obs_rms = obs_rms

            sub_video_paths = []
            for k in range(self.num_processes):
                # Make sub folder for each validation case
                sub_video_path = os.path.join(video_path, suffix[k][1:], str(iter_num))
                os.makedirs(sub_video_path, exist_ok=True)
                sub_video_paths.append(sub_video_path)

            # self.loss[None] = 0.
            # for l in self.losses:
            #     l[None] = 0.
            #
            # # Clear all batch losses
            # for k in range(self.batch_size):
            #     self.loss_batch[k] = 0.
            # for l in self.losses_batch:
            #     for k in range(self.batch_size):
            #         l[k] = 0.

            # self.taichi_env.solver.clear_states(self.max_steps)

            eval_recurrent_hidden_states = torch.zeros(
                self.num_processes, actor_critic.recurrent_hidden_state_size, device=self.device)
            eval_masks = torch.zeros(self.num_processes, 1, device=self.device)
            obs = self.obs
            for step in range(self.max_steps):
                # Sample actions
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs,
                        eval_recurrent_hidden_states,
                        eval_masks,
                        deterministic=True)
                # Obser reward and next obs
                obs, _, done, infos = self.envs.step(action)

                for info in infos:
                    if 'episode' in info.keys():
                        self.episode_rewards.append(info['episode']['r'])
                # If done then clean the history of observations.
                eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                                for done_ in done])
            if show_gui:
                for i in range(self.max_steps):
                    if i % 20 == 0:
                        for k in range(self.num_processes):
                            visualizer(i, k, sub_video_paths[k], output_video=output_video)

            metric_writer.writer.set_step(step=iter_num)
            for rank in range(self.num_processes):
                self.envs.env_method("compute_losses", indices=rank)
                loss, loss_dict = self.envs.env_method("get_losses", indices=rank)[0]
                # print(f"Rank: {rank}", loss, loss_dict)
                metric_writer.update(f"task_loss{suffix[rank]}", loss)
                for name in self.validate_targets.keys():
                    metric_writer.update(
                        f"{name}_loss{suffix[rank]}", loss_dict[f"loss_{name}"])

            # self.get_loss(self.max_steps + 1, loss_enable=self.task)
            # Write to tensorboard
            # metric_writer.writer.set_step(step=iter_num)
            # for k in range(self.num_processes):
            #     metric_writer.update(f"task_loss{suffix[k]}",
            #                          self.loss_batch[k])
            #     for name in self.validate_targets_values.keys():
            #         metric_writer.update(
            #             f"{name}_loss{suffix[k]}",
            #             self.loss_dict_batch[f"loss_{name}"][k])


