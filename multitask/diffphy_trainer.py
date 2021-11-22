import sys
import pickle as pkl
import pandas as pd
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


class LegacyIO(HookBase):
    def __init__(self):
        super(LegacyIO, self).__init__()
        self.training_loss_path = ""
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
        self.training_loss_path = os.path.join(plot_dir, "training_loss.txt")
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

    # def after_train(self):
    #     with open(self.training_loss_path, "w") as f:
    #         for item in self.trainer.losses_list:
    #             f.write(item + "\n")


class DiffPhyTrainer(BaseTrainer):
    def __init__(self, args, config):
        super(DiffPhyTrainer, self).__init__(args, config)
        self.taichi_env = TaichiEnv(config, train=args.train)
        self.optimize_method = self.taichi_env.config["nn"]["optimizer"]
        # Initialize neural network model
        self.nn = Model(config, self.taichi_env.max_steps,
                        self.taichi_env.batch_size,
                        self.taichi_env.n_input_states,
                        self.taichi_env.n_springs,
                        self.taichi_env.input_state,
                        self.taichi_env.solver_actuation,
                        n_models=self.taichi_env.n_models,
                        n_hidden=self.taichi_env.n_hidden,
                        method=self.optimize_method)
        self.max_reset_step = self.taichi_env.config["nn"]["max_reset_step"]
        self.max_height = self.taichi_env.config["process"]["max_height"]
        self.max_speed = self.taichi_env.config["process"]["max_speed"]
        self.control_length = self.taichi_env.config["robot"]["control_length"]
        # self.max_steps = self.taichi_env.config["process"]["max_steps"]
        self.loss_enable = set(self.taichi_env.task)
        self.change_iter = self.taichi_env.config["process"]["state_feedback_iter"] if "state_feedback_iter" in self.taichi_env.config["process"] else 5000
        self.visual_train = args.visual_train
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

    def visual_probe(self, t, batch_rank=0):
        self.taichi_env.gui.clear()
        self.taichi_env.gui.line((0, self.taichi_env.ground_height), (1, self.taichi_env.ground_height),
                      color=0x000022,
                      radius=3)
        self.taichi_env.solver.draw_robot(self.taichi_env.gui, t, batch_rank, self.taichi_env.target_v, self.taichi_env.target_object_position)
        self.taichi_env.gui.show(os.path.join(self.taichi_env.config_.monitor_dir, f"{self.iter:04}_{t:04}.png"))

    @ti.kernel
    def diff_copy(self, t: ti.i32):
        for model_id, k, i in ti.ndrange(self.taichi_env.n_models, self.taichi_env.batch_size, self.nn.n_output):
            self.taichi_env.solver.actuation[model_id, t, k, i] = self.taichi_env.solver.actuation[model_id, t-1, k, i]

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
            self.taichi_env.n_models = 1
            self.taichi_env.initialize_train(iter, steps, max_speed, max_height)
        elif not train and self.taichi_env.dim == 2:
            if output_c is None:
                output_c = np.zeros(self.taichi_env.batch_size)
            self.taichi_env.initialize_validate(steps, output_v, output_h, output_c)
        elif not train and self.taichi_env.dim == 3:
            self.taichi_env.initialize_script(steps, 0.05, 0, 0.05, 0, 0.05, 0, 0.05, 0)
        self.taichi_env.loss[None] = 0.
        for l in self.taichi_env.losses:
            l[None] = 0.
        # start simulation
        if train:
            with ti.Tape(self.taichi_env.loss):
                for t in range(steps-1):
                    self.taichi_env.solver.pre_advance(t)
                    self.taichi_env.nn_input(t, 0, max_speed, max_height)
                    if t % self.control_length == 0:
                        self.nn.forward(t)
                    else:
                        self.diff_copy(t)
                    self.taichi_env.solver.advance(t)
                self.taichi_env.get_loss(steps, *args, **kwargs)

            if self.visual_train:
                for i in range(1, steps):
                    if iter > 900:
                        if i % 5:
                            self.visual_probe(i)
                    else:
                        if i % 200 == 0:
                            self.visual_probe(i)
        else:
            for t in range(steps-1):
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
            print("State feedback starts to be enabled.")
            if self.iter % 500 == 0 and self.reset_step < self.max_reset_step:
                self.reset_step += 1
            self.rounded_train(self.taichi_env.max_steps-1, self.iter, reset_step=self.reset_step)
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

    def evaluate(self, load_path, custom_loss_enable=None, output_video=False, write_to_tensorboard=False, evaluate_from_value=False):

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
                           output_video=output_video,
                           write_to_tensorboard=write_to_tensorboard,
                           evaluate_from_value=evaluate_from_value)

    def _evaluate(self, load_path, custom_loss_enable=None, output_video=False, write_to_tensorboard=False, evaluate_from_value=False):
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
                x_ = self.taichi_env.x.to_numpy()

                def another_output_mesh(x_, fn):
                    for t in range(1, x_.shape[0]):
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

        video_path = os.path.join(load_path, "video")
        if not evaluate_from_value:
            # Get all model paths
            model_paths = glob.glob(os.path.join(load_path, "models/iter*.pkl"))
            a = [(int(x.split('/')[-1][4:-4]), x) for x in model_paths]
            model_paths = [y for x, y in sorted(a)]
            # model_paths = sorted(model_paths,  key=os.path.getmtime)
            self.logger.info("{} models {}".format(len(model_paths), [m.split('/')[-1] for m in model_paths]))
            model_nums = len(model_paths)
        else:
            model_paths = glob.glob(os.path.join(load_path, "models/*.pkl"))
            model_data = pkl.load(open(model_paths[-1], 'rb'))
            model_nums = len(model_data)
        print(f"Model nums: {model_nums}")

        checkponiter_cnt = 0
        model_load_num = self.taichi_env.config["nn"]["n_models"] if "n_models" in self.taichi_env.config["nn"] else 1
        print(f"Model load num: {model_load_num}")
        tensorboard_buffer = {}
        for current_model_index in range(0, model_nums, model_load_num):
            current_iters = []
            if not evaluate_from_value:
                sub_model_paths = model_paths[current_model_index:current_model_index+model_load_num]
            else:
                sub_model_paths = [x for x in range(current_model_index, min(current_model_index+model_load_num, model_nums))]

            sub_video_paths = []

            for model_id, model_path in enumerate(sub_model_paths):
                current_iter = int(model_path.split('.pkl')[0].split('iter')[1]) if not evaluate_from_value else current_model_index+model_id
                current_iters.append(current_iter)
                if not evaluate_from_value:
                    self.nn.load_weights(model_path, model_id=model_id)
                else:
                    self.nn.load_weights_from_value(model_data[current_model_index+model_id], model_id=model_id)
                self.logger.info("Current iter {}, load from {}".format(current_iter, model_path))

                if not evaluate_from_value:
                    for k in range(self.taichi_env.batch_size):
                        # Make sub folder for each validation case
                        sub_video_path = os.path.join(video_path, suffix[k][1:], str(current_iter))
                        # os.makedirs(sub_video_path, exist_ok=True)
                        sub_video_paths.append(sub_video_path)

            self.taichi_env.setup_robot()

            # Clear all batch losses for all models
            for m in range(model_load_num):
                for k in range(self.taichi_env.batch_size):
                    self.taichi_env.loss_batch[m, k] = 0.

            for l in self.taichi_env.losses_batch:
                for m in range(model_load_num):
                    for k in range(self.taichi_env.batch_size):
                        l[m, k] = 0.

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
            if not evaluate_from_value:
                # Output videos
                for i in range(self.taichi_env.max_steps):
                    if i % 50000 == 0:
                        for k in range(self.taichi_env.batch_size):
                            visualizer(i, k, sub_video_paths[k], output_video=output_video)

            # Collect all results
            for m, current_iter in enumerate(current_iters):
                print(f"Collecting results, current iter {current_iter}")
                tensorboard_buffer[current_iter] = {}
                for k in range(self.taichi_env.batch_size):
                    tensorboard_buffer[current_iter][f"task_loss{suffix[k]}"] = self.taichi_env.loss_batch[m, k]
                    for name in self.taichi_env.validate_targets_values.keys():
                        # print(f"{name}_loss{suffix[k]}", self.taichi_env.loss_dict_batch[f"loss_{name}"][m, k])
                        tensorboard_buffer[current_iter][f"{name}_loss{suffix[k]}"] = self.taichi_env.loss_dict_batch[f"loss_{name}"][m, k]

            checkponiter_cnt += 1
            if (checkponiter_cnt+1) % 100 == 0:
                df_all_results = pd.DataFrame(tensorboard_buffer)
                save_file_name = os.path.join(load_path, f"validation/summary_checkpoint_{checkponiter_cnt*model_load_num:05}.csv")
                df_all_results.to_csv(save_file_name)
                print(f"Save checkpoint: {save_file_name}")

        if write_to_tensorboard:
            # Write to tensorboard
            for current_iter, sub_dict in tensorboard_buffer.items():
                print(f"Writing to tensorboard, current iter {current_iter}")
                evaluator_writer.writer.set_step(step=current_iter)
                for k, val in sub_dict.items():
                    evaluator_writer.update(k, val)
        df_all_results = pd.DataFrame(tensorboard_buffer)
        df_all_results.to_csv(os.path.join(load_path, "validation/summary.csv"))
