import os
import torch
import itertools
from collections import deque
import numpy as np
import taichi as ti

from gym import spaces
from multitask.multitask_obj import BaseTrainer
from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.envs import make_env, make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from multitask.hooks import Checkpointer, InfoPrinter, Timer, MetricWriter
from logger import TensorboardWriter
from util import MetricTracker


class CheckpointerRL(Checkpointer):
    def __init__(self, save_path):
        super(CheckpointerRL, self).__init__(save_path)

    def after_step(self):
        if (self.trainer.iter % self.trainer.args.save_interval == 0 or
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
                "Iterations: {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(
                    self.trainer.iter, total_num_steps,
                    int(total_num_steps / (self.trainer.timer.step_end -
                                           self.trainer.timer.step_start)),
                    len(self.trainer.episode_rewards),
                    np.mean(self.trainer.episode_rewards),
                    np.median(self.trainer.episode_rewards),
                    np.min(self.trainer.episode_rewards),
                    np.max(self.trainer.episode_rewards)
                ))  #, dist_entropy, value_loss, action_loss))


class RLTrainer(BaseTrainer):
    def __init__(self, args, config):
        super(RLTrainer, self).__init__(config)
        self.args = args
        self.config = config
        self.device = torch.device("cuda:0" if args.cuda else "cpu")
        self.envs = make_vec_envs(self, args.env_name, args.seed,
                                  args.num_processes, args.gamma, self.device,
                                  False)
        # self.envs = make_env(self, args.env_name, args.seed, rank=0, allow_early_resets=False)()

        obs_shape = list(self.envs.observation_space.shape)
        obs_shape[0] = obs_shape[0] // self.batch_size
        self.obs_shape = tuple(obs_shape)
        print('env obs ', obs_shape, ' env act', self.envs.action_space)

        # We only need one shared parameters NN for all batches
        act_assign = np.ones(len(self.solver.act_list), dtype=np.float64)
        action_space_assigner = spaces.Box(-act_assign, act_assign)
        self.actor_critic = Policy(
            obs_shape,
            action_space_assigner,
            base_kwargs={'recurrent': args.recurrent_policy})

        self.actor_critic.to(self.device)

        # Define the RL algorithm
        self.agent = None
        self._select_algorithm()

        # Whether GAIL
        self.discr = None
        self.gail_train_loader = None
        self._gail()

        # self.rollouts = RolloutStorage(
        #     args.num_steps, args.num_processes,
        #     self.envs.observation_space.shape, self.envs.action_space,
        #     self.actor_critic.recurrent_hidden_state_size)

        self.rollouts = RolloutStorage(
            args.num_steps, self.batch_size,
            obs_shape, action_space_assigner,
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
        self.metrics_train = ["mean_reward", "max_reward", "min_reward"]
        self.metrics_validation = ["task_loss", "velocity_loss", "height_loss"]

        self.register_hooks([
            self.timer, self.checkpointer, self.info_printer,
            self.metric_writer
        ])

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
            obs, reward, done, infos = self.envs.step(torch.unsqueeze(action, dim=0))

            for info in infos:
                if 'episode' in info.keys():
                    self.episode_rewards.append(info['episode']['r'])
                    # print('obs ', obs.shape, 'reward ', reward, 'done ', done, 'infos', infos)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])

            # Recover the obs shape back to [batch_size, obs_shape]
            self.rollouts.insert(obs.view(self.batch_size, self.obs_shape[0]), recurrent_hidden_states, action,
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

    def validate(self):
        self._evaluate()

    def evaluate(self, exp_folder):
        output_folder = os.path.join(exp_folder, 'validation')
        os.makedirs(output_folder, exist_ok=True)
        evaluator_writer = MetricTracker(*[],
                                         writer=TensorboardWriter(
                                             output_folder,
                                             self.logger,
                                             enabled=True))
        self._evaluate(exp_folder=exp_folder, evaluator_writer=evaluator_writer)

    def _evaluate(self,
                  exp_folder=None,
                  output_video=True,
                  show_gui=False,
                  evaluator_writer=None):
        metric_writer = None
        if evaluator_writer is None:
            metric_writer = self.metric_writer.valid_metrics
        else:
            metric_writer = evaluator_writer

        video_path = None
        if exp_folder is None:
            model_folder = str(self.config.model_dir)
            video_path = self.config.video_path
        else:
            model_folder = os.path.join(exp_folder, 'models')
            video_path = os.path.join(exp_folder, 'video')

        def visualizer(t, folder, output_video):
            gui.clear()
            gui.line((0, self.ground_height), (1, self.ground_height),
                     color=0x000022,
                     radius=3)
            self.solver.draw_robot(gui, t, self.target_v)
            if output_video:
                gui.show('{}/{:04d}.png'.format(folder, t))
            else:
                gui.show()

        # Set the trainer training mode to False to invoke the `initialize_validate`
        self.training = False
        gui = ti.GUI(background_color=0xFFFFFF, show_gui=show_gui)

        # Construct the validation matrix i.e., combinations of all validation targets
        targets_keys = []
        targets_values = []
        for k, v in self.validate_targets.items():
            targets_keys.append(k)
            targets_values.append(v)
        validation_matrix = list(itertools.product(*targets_values))
        self.validate_targets_values = dict.fromkeys(self.task)
        self.logger.info(f"Validation targets: {targets_keys} "
                         f"Validation Matrix: {validation_matrix}")

        for iter_num in range(0, self.max_iter, self.args.save_interval):
            model_name = self.args.env_name + str(iter_num) + ".pt"
            load_path = os.path.join(model_folder, model_name)
            [actor_critic, self.envs.venv.obs_rms] = torch.load(load_path)
            self.logger.info(f"load {load_path}")

            for element in validation_matrix:
                for i, name in enumerate(targets_keys):
                    self.validate_targets_values[name] = element[i]

                # Make sub folder for each validation case
                suffix = ""
                for k in targets_keys:
                    suffix += f"_{k}_{self.validate_targets_values[k]}"

                sub_video_path = os.path.join(video_path, suffix[1:], str(iter_num))
                os.makedirs(sub_video_path, exist_ok=True)

                self.loss[None] = 0.
                for l in self.losses:
                    l[None] = 0.
                self.solver.clear_states(self.max_steps)

                eval_recurrent_hidden_states = self.rollouts.recurrent_hidden_states
                eval_masks = self.rollouts.masks
                obs = self.obs
                for step in range(self.args.num_steps):
                    # Sample actions
                    with torch.no_grad():
                        _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                            obs,
                            eval_recurrent_hidden_states,
                            eval_masks)
                    # Obser reward and next obs
                    obs, _, done, infos = self.envs.step(action)
                    for info in infos:
                        if 'episode' in info.keys():
                            self.episode_rewards.append(info['episode']['r'])
                    # If done then clean the history of observations.
                    eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                               for done_ in done])

                for i in range(self.max_steps):
                    if i % 20 == 0:
                        visualizer(i, sub_video_path, output_video=output_video)

                self.get_loss(self.max_steps + 1, loss_enable=self.task)

                # Write to tensorboard
                metric_writer.writer.set_step(step=iter_num)
                metric_writer.update(f"task_loss{suffix}",
                                     self.loss[None])
                for k in self.validate_targets_values.keys():
                    metric_writer.update(
                        f"{k}_loss{suffix}",
                        self.loss_dict[f"loss_{k}"].to_numpy())

    # def evaluate(self):
    #     if (self.args.eval_interval is not None and len(self.episode_rewards) > 1
    #             and self.iter % self.args.eval_interval == 0):
    #         obs_rms = utils.get_vec_normalize(self.envs).obs_rms
    #         evaluate(self.actor_critic, obs_rms, self.args.env_name, self.args.seed,
    #                  self.args.num_processes, eval_log_dir, self.device)
