import sys
import taichi as ti
import math
import numpy as np
import os
import time
import weakref
import threading
import itertools
import glob


from collections import defaultdict
from logger import TensorboardWriter
from multitask.hooks import Checkpointer, InfoPrinter, Timer, MetricWriter
from util import MetricTracker


from multitask.nn import Model
from multitask.hooks import HookBase
from multitask.utils import Debug, real, plot_curve, load_string, scalar, vec, mat
from multitask.solver_mass_spring import SolverMassSpring
from multitask.solver_mpm import SolverMPM
from multitask.taichi_env import TaichiEnv
from multitask.base_trainer import BaseTrainer
debug = Debug(False)


# @ti.data_oriented
# class BaseTrainer:
#     def __init__(self, args, config):
#         self.logger = config.get_logger(name=config.get_config()["train"]["name"])
#         self.random_seed = config.get_config()["train"]["random_seed"]
#         self.num_processes = args.num_processes
#         self.training = True
#         self.iter = 0
#         self.max_iter = 10000  # Default training iterations, can be overwrote by args
#         self.writer = TensorboardWriter(config.log_dir,
#                                         self.logger,
#                                         enabled=(not args.no_tensorboard))
#         self._hooks = []
#
#     def before_train(self):
#         for h in self._hooks:
#             h.before_train()
#
#     def before_step(self):
#         for h in self._hooks:
#             h.before_step()
#
#     def after_step(self):
#         for h in self._hooks:
#             h.after_step()
#
#     def after_train(self):
#         for h in self._hooks:
#             h.after_train()
#
#     def run_step(self):
#         raise NotImplementedError
#
#     def train(self, start_iter, max_iter):
#
#         self.iter = self.start_iter = start_iter
#         self.max_iter = max_iter // self.num_processes
#         self.logger.info(f"Starting training from iteration {start_iter}, Number of Processes: {self.num_processes}, "
#                          f"Max iterations: {self.max_iter}")
#
#         try:
#             self.before_train()
#             for self.iter in range(start_iter, self.max_iter):
#                 self.before_step()
#                 self.run_step()
#                 self.after_step()
#             # self.iter == max_iter can be used by `after_train` to
#             # tell whether the training successfully finished or failed
#             # due to exceptions.
#             self.iter += 1
#         except Exception:
#             self.logger.exception("Exception during training:")
#             raise
#         finally:
#             self.after_train()
#
#     def register_hooks(self, hooks):
#         hooks = [h for h in hooks if h is not None]
#         for h in hooks:
#             assert isinstance(h, HookBase)
#             # To avoid circular reference, hooks and trainer cannot own each other.
#             # This normally does not matter, but will cause memory leak if the
#             # involved objects contain __del__:
#             # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
#             h.trainer = weakref.proxy(self)
#         self._hooks.extend(hooks)


class LegacyIO(HookBase):
    def __init__(self):
        super(LegacyIO, self).__init__()
        self.plot_path = ""
        self.plot200_path = ""

    def before_train(self):
        prefix = self.trainer.prefix
        plot_dir = self.trainer.plot_dir
        log_name = "training.log"
        if prefix is not None:
            log_name = "{}_training.log".format(prefix)
        log_path = os.path.join(self.trainer.log_dir, log_name)

        log_file = open(log_path, 'w')
        log_file.close()

        plot_name = "training_curve.png"
        plot200_name = "training_curve_last_200.png"
        if prefix is not None:
            plot_name = "{}_training_curve.png".format(self.trainer.prefix)
            plot200_name = "{}_training_curve_last_200.png".format(prefix)
        self.plot_path = os.path.join(plot_dir, plot_name)
        self.plot200_path = os.path.join(plot_dir, plot200_name)

        self.trainer.taichi_env.setup_robot()

        if self.trainer.load_path is not None and os.path.exists(self.trainer.load_path):
            self.trainer.logger.info("load from {}".format(self.trainer.load_path))
            self.trainer.nn.load_weights(self.trainer.load_path)
        else:
            self.trainer.nn.weights_init()
        if self.trainer.optimize_method == "adam":
            self.trainer.nn.clear_adam()

    def after_step(self):

        weight_out = lambda x: os.path.join(self.trainer.weights_dir, x)
        if self.trainer.iter <= self.trainer.change_iter and self.trainer.taichi_env.loss[None] < self.trainer.best:
            self.trainer.best = self.trainer.taichi_env.loss[None]
            self.trainer.nn.dump_weights(weight_out("best.pkl"))

        if self.trainer.iter > self.trainer.change_iter + self.trainer.max_reset_step and self.trainer.taichi_env.loss[None] < self.trainer.best_finetune:
            self.trainer.best_finetune = self.trainer.taichi_env.loss[None]
            self.trainer.nn.dump_weights(weight_out("best_finetune.pkl"))

        self.trainer.nn.dump_weights(weight_out("last.pkl"))
        self.trainer.nn.dump_weights(os.path.join(self.trainer.weights_dir, "weight.pkl"))

        if self.trainer.iter % 50 == 0:
            self.trainer.nn.dump_weights(weight_out("iter{}.pkl".format(self.trainer.iter)))

        if self.trainer.iter % 100 == 0 or self.trainer.iter % 10 == 0 and self.trainer.iter < 500:
            plot_curve(self.trainer.losses_list, self.plot_path)
            plot_curve(self.trainer.losses_list[-200:], self.plot200_path)

        def print_logs(file=None):
            if self.trainer.iter > self.trainer.change_iter:
                print('Iter=', self.trainer.iter, 'Loss=', self.trainer.taichi_env.loss[None], 'Best_FT=', self.trainer.best_finetune, file=file)
            else:
                print('Iter=', self.trainer.iter, 'Loss=', self.trainer.taichi_env.loss[None], 'Best=', self.trainer.best, file=file)
            print("TNS= ", self.trainer.total_norm_sqr, file=file)
            for name, l in self.trainer.taichi_env.loss_dict.items():
                print("{}={}".format(name, l[None]), file=file)
            print("{}={}".format("max height", self.trainer.taichi_env.current_max_height[None]), file=file)

        print_logs()
        log_file = open(os.path.join(self.trainer.log_dir, "training.log"), "a")
        print_logs(log_file)
        log_file.close()
        self.trainer.taichi_env.current_max_height[None] = 0.


class DiffPhyTrainer(BaseTrainer):
    def __init__(self, args, config):
        super(DiffPhyTrainer, self).__init__(args, config)
        if args.evaluate:
            config.get_config()["process"]["max_steps"] = config.get_config()["process"]["evaluate_max_steps"]
        self.taichi_env = TaichiEnv(config)
        self.optimize_method = self.taichi_env.config["nn"]["optimizer"]
        # Initialize neural network model
        self.nn = Model(config, self.taichi_env.max_steps,
                        self.taichi_env.batch_size,
                        self.taichi_env.n_input_states,
                        self.taichi_env.n_springs,
                        self.taichi_env.input_state,
                        self.taichi_env.actuation,
                        self.taichi_env.n_hidden,
                        method=self.optimize_method)
        self.max_reset_step = self.taichi_env.config["nn"]["max_reset_step"]
        self.max_height = self.taichi_env.config["process"]["max_height"]
        self.max_speed = self.taichi_env.config["process"]["max_speed"]
        self.control_length = self.taichi_env.config["robot"]["control_length"]
        # self.max_steps = self.taichi_env.config["process"]["max_steps"]
        self.loss_enable = set(self.taichi_env.task)
        self.change_iter = 5000
        self.reset_step = 2
        self.total_norm_sqr = 0.
        self.losses_list = []

        self.best = 1e+15
        self.best_finetune = 1e+15
        self.train_steps = self.taichi_env.max_steps

        self.prefix = None
        self.log_dir = self.taichi_env.config_.log_dir
        self.plot_dir = self.taichi_env.config_.monitor_dir
        self.weights_dir = self.taichi_env.config_.model_dir
        self.load_path = None

        # Metrics to tracking during training
        self.metrics_train = []
        self.metrics_validation = []

        self.legacy_io = LegacyIO()
        self.metric_writer = MetricWriter()
        self.register_hooks([self.legacy_io, self.metric_writer])

    @ti.kernel
    def diff_copy(self, t: ti.i32):
        for k, i in ti.ndrange(self.taichi_env.batch_size, self.nn.n_output):
            self.taichi_env.solver.actuation[t, k, i] = self.taichi_env.solver.actuation[t-1, k, i]
    @debug
    def simulate(self, steps,
                 output_v=None,
                 output_h=None,
                 output_c=None,
                 train=True,
                 iter=0,
                 max_speed=0.08,
                 max_height=0.1,
                 *args,
                 **kwargs):
        # clean up cache and set up control sequence
        self.taichi_env.solver.clear_states(steps)
        self.nn.clear()
        if train:
            self.taichi_env.initialize_train(iter, steps, max_speed, max_height)
        elif not train and self.taichi_env.dim == 2:
            if output_c is None:
                output_c = np.zeros(self.taichi_env.batch_size)
            self.taichi_env.initialize_validate(steps, output_v, output_h, output_c)
        elif not train and self.taichi_env.dim == 3:
            self.taichi_env.initialize_script(steps, -max_speed, 0, 0, max_speed, max_speed, 0, 0, -max_speed)
        self.taichi_env.loss[None] = 0.
        for l in self.taichi_env.losses:
            l[None] = 0.
        # start simulation
        if train:
            with ti.ad.Tape(self.taichi_env.loss):
                # for t in range(steps + 1):
                for t in range(steps-1):
                    self.taichi_env.solver.pre_advance(t)
                    self.taichi_env.nn_input(t, 0, max_speed, max_height)
                    if t % self.control_length == 0:
                        self.nn.forward(t)
                    else:
                        self.diff_copy(t)
                    self.taichi_env.solver.advance(t)
                self.taichi_env.get_loss(steps, *args, **kwargs)
        else:
            for t in range(steps + 1):
                self.taichi_env.solver.pre_advance(t)
                self.taichi_env.nn_input(t, 0, max_speed, max_height)
                # self.nn.forward(t)
                if t % self.control_length == 0:
                    self.nn.forward(t)
                else:
                    self.diff_copy(t)
                self.taichi_env.solver.advance(t)
            # self.visualizer(steps, prefix=str(output_v) + "_" + str(output_h))
            self.taichi_env.get_loss(steps, *args, **kwargs)

    def rounded_train(self, steps, iter, reset_step):
        self.taichi_env.copy_robot(steps)
        start = iter % reset_step
        step = reset_step
        times = (self.taichi_env.batch_size + step - start - 1) // step
        self.taichi_env.reset_robot(start, step, times)

    def run_step(self, *args, **kwargs):
        if self.iter > self.change_iter:
            if self.iter % 500 == 0 and self.reset_step < self.max_reset_step:
                self.reset_step += 1
            self.rounded_train(self.taichi_env.max_steps, self.iter, reset_step=self.reset_step)
        print("-------------------- {}iter #{} --------------------" \
              .format("" if self.prefix is None else "{}, ".format(self.prefix), self.iter))

        self.simulate(self.taichi_env.max_steps, iter=self.iter,
                      max_height=self.max_height,
                      max_speed=self.max_speed,
                      loss_enable=self.loss_enable)

        # Total weights gradients norm square
        self.total_norm_sqr = self.nn.get_TNS()
        self.nn.gradient_update(self.iter)
        self.losses_list.append(self.taichi_env.loss[None])

        # Write to tensorboard
        self.metric_writer.writer.set_step(step=self.iter - 1)
        for name, l in self.taichi_env.loss_dict.items():
            self.metric_writer.train_metrics.update(name, l.to_numpy())
        self.metric_writer.train_metrics.update('training_loss', self.taichi_env.loss[None])
        self.metric_writer.train_metrics.update('best', self.best)
        self.metric_writer.train_metrics.update('TNS', self.total_norm_sqr)
        self.metric_writer.train_metrics.update('current_max_height', self.taichi_env.current_max_height[None])

    def evaluate(self, load_path, custom_loss_enable=None, output_video=False):

        load_paths = glob.glob(os.path.join(load_path, "*"))
        load_paths = sorted(load_paths, key=os.path.getmtime)

        paths_to_evaluate = []
        # Check whether this experiment has been evaluated before
        for lp in load_paths:
            if len(os.listdir(os.path.join(lp, "validation"))) == 0:
                paths_to_evaluate.append(lp)
        print(f"All experiments to evaluate {paths_to_evaluate}")
        for lp in paths_to_evaluate:
            self._evaluate(load_path=lp,
                           custom_loss_enable=custom_loss_enable,
                           output_video=output_video)

    def _evaluate(self, load_path, custom_loss_enable=None, output_video=False):
        gui = ti.GUI(background_color=0xFFFFFF, show_gui=False)

        def visualizer(t, batch_rank, folder, output_video):
            if self.taichi_env.dim == 2:
                gui.clear()
                gui.line((0, self.taichi_env.ground_height), (1, self.taichi_env.ground_height),
                         color=0x000022,
                         radius=3)
                self.taichi_env.solver.draw_robot(gui, batch_rank, t, self.taichi_env.target_v)
                if output_video:
                    gui.show('{}/{:04d}.png'.format(folder, t))
                else:
                    gui.show()
            else:
                x_ = self.taichi_env.x.to_numpy()[::10, 0:1, :, :]

                def another_output_mesh(x_, fn):
                    for t in range(0, x_.shape[0]):
                        f = open(fn + f'/{t:06d}.obj', 'w')
                        for i in range(x_.shape[2]):
                            f.write('v %.6f %.6f %.6f\n' % (x_[t, 0, i, 0], x_[t, 0, i, 1], x_[t, 0, i, 2]))
                        for [p0, p1, p2] in self.taichi_env.config["robot"]["faces"]:
                            f.write('f %d %d %d\n' % (p0 + 1, p1 + 1, p2 + 1))
                        f.close()

                t = threading.Thread(target=another_output_mesh, args=(x_, folder))
                t.start()
        evaluator_writer = MetricTracker(*[],
                                         writer=TensorboardWriter(
                                             os.path.join(load_path, "validation"),
                                             self.logger,
                                             enabled=True))

        if custom_loss_enable is None:
            loss_enable = set(self.task)
        else:
            loss_enable = custom_loss_enable
        model_paths = glob.glob(os.path.join(load_path, "models/iter*.pkl"))
        model_paths = sorted(model_paths,  key=os.path.getmtime)
        self.logger.info("{} models {}".format(len(model_paths), [m.split('/')[-1] for m in model_paths]))
        video_path = os.path.join(load_path, "video")

        targets_values = []
        targets_keys = []
        for k, v in self.taichi_env.validate_targets.items():
            targets_keys.append(k)
            targets_values.append(v)
        validation_matrix = list(itertools.product(*targets_values))
        self.taichi_env.validate_targets_values = dict.fromkeys(loss_enable)
        self.taichi_env.validate_targets_values = defaultdict(list)
        self.logger.info(f"Validation targets: {targets_keys} "
                         f"Validation Matrix: {validation_matrix}")
        print(len(validation_matrix), self.taichi_env.batch_size)
        assert len(validation_matrix) == self.taichi_env.batch_size
        suffix = []
        for element in validation_matrix:
            s_base = ""
            for i, name in enumerate(targets_keys):
                self.taichi_env.validate_targets_values[name].append(element[i])
                s_base += f"_{name}_{element[i]}"
            suffix.append(s_base)

        best_paths = glob.glob(os.path.join(load_path, "models/best.pkl"))
        last_paths = glob.glob(os.path.join(load_path, "models/last.pkl"))
        for model_path in best_paths + last_paths + model_paths[::10]:
            if model_path[-8:] == "best.pkl":
                current_iter = -1
            elif model_path[-8:] == "last.pkl":
                current_iter = -2
            else:
                current_iter = int(model_path.split('.pkl')[0].split('iter')[1])
            sub_video_paths = []
            for k in range(self.taichi_env.batch_size):
                # Make sub folder for each validation case
                sub_video_path = os.path.join(video_path, suffix[k][1:], str(current_iter))
                os.makedirs(sub_video_path, exist_ok=True)
                sub_video_paths.append(sub_video_path)

            self.taichi_env.setup_robot()
            self.logger.info("Current iter {}, load from {}".format(current_iter, model_path))
            self.nn.load_weights(model_path)

            # Clear all batch losses
            for k in range(self.taichi_env.batch_size):
                self.taichi_env.loss_batch[k] = 0.
            for l in self.taichi_env.losses_batch:
                for k in range(self.taichi_env.batch_size):
                    l[k] = 0.

            validate_v = self.taichi_env.validate_targets_values['velocity']
            validate_h = self.taichi_env.validate_targets_values['height']
            validate_c = self.taichi_env.validate_targets_values['crawl']
            if len(validate_c) == 0:
                validate_c = np.zeros(self.taichi_env.batch_size)
            self.logger.info(f"current max speed: {validate_v}, max height {validate_h}, max crawl {validate_c}")
            self.simulate(self.taichi_env.max_steps,
                          output_v=np.array(validate_v),
                          output_h=np.array(validate_h),
                          output_c=np.array(validate_c),
                          iter=0,
                          train=False,
                          loss_enable=loss_enable)

            for i in range(self.taichi_env.max_steps):
                if i % 20 == 0:
                    for k in range(self.taichi_env.batch_size):
                        visualizer(i, k, sub_video_paths[k], output_video=output_video)

            # Write to tensorboard
            evaluator_writer.writer.set_step(step=current_iter)
            for k in range(self.taichi_env.batch_size):
                evaluator_writer.update(f"task_loss{suffix[k]}", self.taichi_env.loss_batch[k])
                for name in self.taichi_env.validate_targets_values.keys():
                    # print(f"{name}_loss{suffix[k]}", self.taichi_env.loss_dict_batch[f"loss_{name}"][k])
                    evaluator_writer.update(f"{name}_loss{suffix[k]}", self.taichi_env.loss_dict_batch[f"loss_{name}"][k])

    # Legacy code from ljcc
    def optimize(self, iters=100000, change_iter=5000, prefix=None, root_dir="./", \
                 load_path=None, *args, **kwargs):
        log_dir = os.path.join(root_dir, "logs")
        plot_dir = os.path.join(root_dir, "plots")
        weights_dir = os.path.join(root_dir, "weights")

        # log_dir = self._config.log_dir
        # plot_dir = self._config.monitor_dir
        # weights_dir = self._config.model_dir

        if prefix is not None:
            weights_dir = os.path.join(weights_dir, prefix)

        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(weights_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        log_name = "training.log"
        if prefix is not None:
            log_name = "{}_training.log".format(prefix)
        log_path = os.path.join(log_dir, log_name)

        log_file = open(log_path, 'w')
        log_file.close()

        plot_name = "training_curve.png"
        plot200_name = "training_curve_last_200.png"
        if prefix is not None:
            plot_name = "{}_training_curve.png".format(prefix)
            plot200_name = "{}_training_curve_last_200.png".format(prefix)
        plot_path = os.path.join(plot_dir, plot_name)
        plot200_path = os.path.join(plot_dir, plot200_name)

        weight_out = lambda x: os.path.join(weights_dir, x)

        self.setup_robot()

        if load_path is not None and os.path.exists(load_path):
            print("load from {}".format(load_path))
            self.nn.load_weights(load_path)
        else:
            self.nn.weights_init()

        self.nn.clear_adam()

        losses = []
        best = 1e+15
        best_finetune = 1e+15
        train_steps = 1000
        if self.dim == 3 and sys.argv[0] == "validate.py":
            train_steps = 4000

        reset_step = 2

        for iter in range(iters):
            if iter > change_iter:
                if iter % 500 == 0 and reset_step < self.max_reset_step:
                    reset_step += 1
                self.rounded_train(train_steps, iter, reset_step=reset_step)

            print("-------------------- {}iter #{} --------------------" \
                  .format("" if prefix is None else "{}, ".format(prefix), iter))

            self.simulate(train_steps, iter=iter, *args, **kwargs)

            if iter <= change_iter and self.loss[None] < best:
                best = self.loss[None]
                self.nn.dump_weights(weight_out("best.pkl"))

            if iter > change_iter + self.max_reset_step and self.loss[None] < best_finetune:
                best_finetune = self.loss[None]
                self.nn.dump_weights(weight_out("best_finetune.pkl"))

            self.nn.dump_weights(weight_out("last.pkl"))
            self.nn.dump_weights(os.path.join(root_dir, "weight.pkl"))

            if iter % 50 == 0:
                self.nn.dump_weights(weight_out("iter{}.pkl".format(iter)))

            total_norm_sqr = self.nn.get_TNS()

            def print_logs(file=None):
                if iter > change_iter:
                    print('Iter=', iter, 'Loss=', self.loss[None], 'Best_FT=', best_finetune, file=file)
                else:
                    print('Iter=', iter, 'Loss=', self.loss[None], 'Best=', best, file=file)
                print("TNS= ", total_norm_sqr, file=file)
                for name, l in self.loss_dict.items():
                    print("{}={}".format(name, l[None]), file=file)

            print_logs()
            log_file = open(log_path, "a")
            print_logs(log_file)
            log_file.close()

            self.nn.gradient_update(iter)
            losses.append(self.loss[None])

            # Write to tensorboard
            # self.metric_writer.writer.set_step(step=self.iter - 1)
            # for name, l in self.loss_dict.items():
            #     self.metric_writer.train_metrics.update(name, l)

            if iter % 100 == 0 or iter % 10 == 0 and iter < 500:
                plot_curve(losses, plot_path)
                plot_curve(losses[-200:], plot200_path)

        return losses


# if __name__ == '__main__':
#     diff_phy_trainer = DiffPhyTrainer()
#     random_seed = diff_phy_trainer.random_seed
#     ti.init(arch=ti.gpu, default_fp=real, random_seed=random_seed)
#
#     root_dir = "robot_{}".format(robot_id)
#     load_path = os.path.join(root_dir, "weight")
#     if dim == 3:
#         loss_enable = ["rotation", "velocity"]
#         diff_phy_trainer.train(root_dir=root_dir, loss_enable=loss_enable)
#     else:
#         if os.path.exists(root_dir):
#             print()
#             s = load_string("{} exists, continue?(Y/N)".format(root_dir), ["Y", "N"])
#             if s == "N":
#                 exit(0)
#             os.system('rm "{}" -r'.format(root_dir))
#         # optimize(500, 250, "stage1", root_dir, loss_enable = {"height", "pose"}, max_height = 0.01)
#         # #optimize(2000, 1000, "stage2", root_dir, load_path = load_path, loss_enable = {"height", "pose"}, max_height = 0.05)
#         # optimize(2000, 1000, "stage2", root_dir, load_path = load_path, loss_enable = {"height", "pose"})
#         # #optimize(2000, 1000, "stage4", root_dir, load_path = load_path, loss_enable = {"velocity", "actuation"}, max_speed = 0.08)
#         # optimize(100000, 5000, "final", root_dir, load_path = load_path, loss_enable = {"velocity", "height", "actuation"})
#
#         # Pre train stage-1
#         diff_phy_trainer.train(500, 250, "stage1", root_dir, loss_enable={"height", "pose"}, max_height=0.01)
#         # Pre train stage-2
#         diff_phy_trainer.train(2000, 1000, "stage2", root_dir, load_path = load_path, loss_enable = {"height", "pose"})
#         # Train
#         diff_phy_trainer.train(100000, 5000, "final", root_dir, load_path = load_path, loss_enable = {"velocity", "height", "actuation"})
