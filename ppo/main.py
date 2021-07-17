import copy
import glob
import os
import time
from collections import deque
import sys

sys.path.append('../../env_difftaichi/lib/python3.8/site-packages')
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import *
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

# import logging
# from logger import TensorboardWriter
# from parse_config import ConfigParser
# from util import inf_loop, MetricTracker

# def get_logger(name, verbosity=2):
#     log_levels = {
#         0: logging.WARNING,
#         1: logging.INFO,
#         2: logging.DEBUG
#     }
#     msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, log_levels.keys())
#     assert verbosity in log_levels, msg_verbosity
#     logger = logging.getLogger(name)
#     logger.setLevel(log_levels[verbosity])
#     return logger


def main():
    args = get_args()

    # logger = get_logger('trainer', verbosity=2)
    # log_writer = TensorboardWriter(args.log_dir, logger, enabled=True)
    #
    # train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns],
    #                                    writer=log_writer)
    # metrics = [getattr(module_metric, met) for met in config['metrics']]

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    actor_critic = Policy(envs.observation_space.shape,
                          envs.action_space,
                          base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic,
                               args.value_loss_coef,
                               args.entropy_coef,
                               lr=args.lr,
                               eps=args.eps,
                               alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic,
                         args.clip_param,
                         args.ppo_epoch,
                         args.num_mini_batch,
                         args.value_loss_coef,
                         args.entropy_coef,
                         lr=args.lr,
                         eps=args.eps,
                         max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic,
                               args.value_loss_coef,
                               args.entropy_coef,
                               acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir,
            "trajs_{}.pt".format(args.env_name.split('-')[0].lower()))

        expert_dataset = gail.ExpertDataset(file_name,
                                            num_trajectories=4,
                                            subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    if args.validate >= 0:
        task_iter = []
        task_loss = []
        gui = ti.GUI(background_color=0xFFFFFF)
        for iter_num in range(9950, 10000, args.save_interval):
            load_path = os.path.join(args.save_dir, args.algo)
            [actor_critic, envs.venv.obs_rms] = torch.load(
                os.path.join(load_path, args.env_name + str(iter_num) + ".pt"))
            print(
                "load ",
                os.path.join(load_path, args.env_name + str(iter_num) + ".pt"))

            multitask.loss[None] = 0.
            for l in multitask.losses:
                l[None] = 0.
            multitask.solver.clear_states(config.max_steps)

            for step in range(args.num_steps):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])
                # Obser reward and next obs
                obs, reward, done, infos = envs.step(action)
                for info in infos:
                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode']['r'])
                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                           for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos])
                rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks,
                                bad_masks)

            def visualizer(t, folder):
                gui.clear()
                gui.line((0, multitask.ground_height),
                         (1, multitask.ground_height),
                         color=0x000022,
                         radius=3)
                multitask.solver.draw_robot(gui, t, multitask.target_v)
                gui.show('{}/{:04d}.png'.format(folder, t))

            folder = os.path.join(
                args.log_dir, "video_{}/{}".format(args.env_name, iter_num))
            os.makedirs(folder, exist_ok=True)
            for i in range(config.max_steps):
                if i % 10 == 0:
                    visualizer(i, folder)

            multitask.get_loss(config.max_steps + 1, loss_enable={"velocity"})

            task_iter.append(iter_num)
            task_loss.append(multitask.loss[None])

            print("Task iteration: ", task_iter)
            print("Task loss: ", task_loss)
            # if iter_num == 9950:
            #     data_dict = {'iters':task_iter,'validation loss':task_loss}
            #     with open(args.log_dir+f"video_{args.env_name}/loss.json", 'w') as f:
            #         json.dump(data_dict, f)

    # Training
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], os.path.join(save_path, args.env_name + str(j) + ".pt"))
            print("save {}......".format(j))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        # log_writer.set_step((j - 1) * num_updates)
        # train_metrics.update('mean_reward', np.mean(episode_rewards))
        # train_metrics.update('max_reward', np.max(episode_rewards))
        # train_metrics.update('min_reward', np.min(episode_rewards))
        # train_metrics.update('value_loss', value_loss)
        # train_metrics.update('action_loss', action_loss)
        # train_metrics.update('dist_entropy', dist_entropy)

        # for met in self.metric_ftns:
        #     self.train_metrics.update(met.__name__, met(out_v, gt_v))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            evaluate(actor_critic, obs_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()
