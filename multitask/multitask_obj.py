import sys
import taichi as ti
import math
import numpy as np
import os
import time
import weakref
import threading
import logging

from multitask.hooks import HookBase

from multitask.utils import Debug, real, plot_curve, load_string, scalar, vec, mat
from multitask.solver_mass_spring import SolverMassSpring
from multitask.solver_mpm import SolverMPM
from logger import TensorboardWriter

debug = Debug(False)


@ti.data_oriented
class BaseTrainer:
    def __init__(self, config):
        self.logger = config.get_logger(name="DiffTaichi")
        self.config = config.get_config()
        self.dim = self.config["robot"]["dim"]
        self.max_steps = self.config["process"]["max_steps"]
        self.batch_size = self.config["nn"]["batch_size"]
        self.robot_id = self.config["robot"]["robot_id"]
        self.n_input_states = self.config["nn"]["n_input_states"]
        self.n_springs = self.config["robot"]["n_springs"]
        self.objects = self.config["robot"]["objects"]
        self.simulator = self.config["robot"]["simulator"]
        self.turn_period = self.config["process"]["turn_period"]
        self.run_period = self.config["process"]["run_period"]
        self.jump_period = self.config["process"]["jump_period"]
        self.n_sin_waves = self.config["nn"]["n_sin_waves"]
        self.spring_omega = self.config["process"]["spring_omega"]
        self.dt = self.config["process"]["dt"]
        self.duplicate_v = self.config["nn"]["duplicate_v"]
        self.duplicate_h = self.config["nn"]["duplicate_h"]
        self.output_vis_interval = self.config["process"]["output_vis_interval"]
        self.ground_height = self.config["simulator"]["ground_height"]

        self.validate_v_list = self.config["validation"]["target_v"]
        self.validate_h_list = self.config["validation"]["target_h"]
        self.validate_v = 0.0
        self.validate_h = 0.0

        self.random_seed = int(time.time() * 1e6) % 10000
        self.training = True
        self.iter = 0
        self.max_iter = 10000  # Default training iterations, can be overwrote by args
        self.loss = scalar()
        self.loss_velocity = scalar()
        self.loss_height = scalar()
        self.loss_pose = scalar()
        self.loss_rotation = scalar()
        self.loss_weight = scalar()
        self.loss_act = scalar()
        self.loss_dict = {'loss_v': self.loss_velocity,
                          'loss_h': self.loss_height,
                          'loss_p': self.loss_pose,
                          'loss_r': self.loss_rotation,
                          'loss_w': self.loss_weight,
                          'loss_a': self.loss_act}
        self.losses = self.loss_dict.values()

        ti.root.place(self.loss)
        ti.root.place(*self.losses)

        self.initial_objects = vec(self.dim)
        self.initial_center = vec(self.dim)

        self.n_objects = self.config["robot"]["n_objects"]
        ti.root.dense(ti.i, self.n_objects).place(self.initial_objects)
        ti.root.place(self.initial_center)

        self.input_state = scalar()

        ti.root.dense(ti.ijk, (self.max_steps, self.batch_size, self.n_input_states)).place(self.input_state)

        self.target_v, self.target_h, self.target_c = vec(self.dim), scalar(), scalar()
        ti.root.dense(ti.ij, (self.max_steps, self.batch_size)).place(self.target_v, self.target_h, self.target_c)

        # Initialize simulation
        self.solver = SolverMPM() if self.simulator == "mpm" else SolverMassSpring(config)
        self.x = self.solver.x
        self.v = self.solver.v
        self.center = self.solver.center
        self.height = self.solver.height
        self.rotation = self.solver.rotation
        self.actuation = self.solver.actuation
        self.pool = ti.field(ti.f64, shape=(5 * self.batch_size * (1000 // self.turn_period + 1)))

        self.gui = ti.GUI(show_gui=False, background_color=0xFFFFFF)

        self.writer = TensorboardWriter(config.log_dir,
                                        self.logger,
                                        enabled=True)

        self._hooks = []
        self.task_list = ["Move B&F", "Move B&F + jump", "Move B&F + crawl", "Keep still"]

    @ti.kernel
    def nn_input(self, t: ti.i32, offset: ti.i32, max_speed: ti.f64, max_height: ti.f64):
        for k, j in ti.ndrange(self.batch_size, self.n_sin_waves):
            self.input_state[t, k, j] = ti.sin(self.spring_omega * (t + offset) * self.dt + 2 * math.pi / self.n_sin_waves * j)
        dim = int(self.dim)
        n_sin_waves = self.n_sin_waves
        duplicate_v = self.duplicate_v
        duplicate_h = self.duplicate_h
        n_objects = self.n_objects
        batch_size = self.batch_size
        for k, j in ti.ndrange(self.batch_size, self.n_objects):
            vec_x = self.x[t, k, j] - self.center[t, k]
            for d in ti.static(range(self.dim)):
                if ti.static(self.dim == 2):
                    self.input_state[t, k, j * dim * 2 + n_sin_waves + d] = vec_x[d] / 0.2
                    self.input_state[t, k, j * dim * 2 + n_sin_waves + dim + d] = 0
                else:
                    self.input_state[t, k, j * dim * 2 + n_sin_waves + d] = vec_x[d] * float(sys.argv[2])
                    self.input_state[t, k, j * dim * 2 + n_sin_waves + dim + d] = 0

        if ti.static(self.duplicate_v > 0):
            if ti.static(self.dim == 2):
                for k, j in ti.ndrange(self.batch_size, self.duplicate_v):
                    self.input_state[t, k, n_objects * dim * 2 + n_sin_waves + j * (dim - 1)] = self.target_v[t, k][0] / max_speed
            else:
                for k, j in ti.ndrange(batch_size, duplicate_v):
                    self.input_state[t, k, n_objects * dim * 2 + n_sin_waves + j * (dim - 1)] = self.target_v[t, k][0] * float(
                        sys.argv[3])
                    self.input_state[t, k, n_objects * dim * 2 + n_sin_waves + j * (dim - 1) + 1] = self.target_v[t, k][
                                                                                                   2] * float(
                        sys.argv[3])
        if ti.static(self.duplicate_h > 0):
            for k, j in ti.ndrange(self.batch_size, self.duplicate_h):
                self.input_state[t, k, n_objects * dim * 2 + n_sin_waves + duplicate_v * (dim - 1) + j] = (self.target_h[
                                                                                                          t, k] - 0.1) / max_height * 2 - 1

    @ti.kernel
    def compute_loss_velocity(self, steps: ti.template()):
        for t, k in ti.ndrange((self.run_period, steps + 1), self.batch_size):
            if t % self.turn_period > self.run_period:  # and target_h[t - run_period, k] < 0.1 + 1e-4:
                if ti.static(self.dim == 2):
                    self.loss_velocity[None] += (self.center[t, k](0) - self.center[t - self.run_period, k](0) - self.target_v[
                        t - self.run_period, k](0)) ** 2 / self.batch_size
                else:
                    self.loss_velocity[None] += (self.center[t, k](0) - self.center[t - self.run_period, k](0) - self.target_v[
                        t - self.run_period, k](0)) ** 2 / self.batch_size
                    self.loss_velocity[None] += (self.center[t, k](2) - self.center[t - self.run_period, k](2) - self.target_v[
                        t - self.run_period, k](2)) ** 2 / self.batch_size
        # if k == 0:
        #     print("Mark run: ", center[t, 0](0) - center[t - run_period, 0](0), target_v[t - run_period, 0](0))

    @ti.kernel
    def compute_loss_height(self, steps: ti.template()):
        for t, k in ti.ndrange((1, steps + 1), self.batch_size):
            if t % self.jump_period == self.jump_period - 1 and self.target_h[t, k] > 0.1:
                self.loss_height[None] += (self.height[t, k] - self.target_h[t, k]) ** 2 / self.batch_size / (steps // self.jump_period) * 100

    @ti.kernel
    def compute_loss_pose(self, steps: ti.template()):
        # TODO: This doesn't work for 3D
        for t, k, i in ti.ndrange((1, steps + 1), self.batch_size, self.n_objects):
            if t % self.jump_period == 0:
                # dist2 = sum((x[t, k, i] - center[t, k] - initial_objects[i] + initial_center[None]) ** 2)
                dist2 = sum((self.x[t, k, i] - self.initial_objects[i]) ** 2)
                self.loss_pose[None] += dist2 / self.batch_size / (steps // self.jump_period)

    @ti.kernel
    def compute_loss_rotation(self, steps: ti.template()):
        for t, k in ti.ndrange((1, steps + 1), self.batch_size):
            self.loss_rotation[None] += self.rotation[t, k] ** 2 / self.batch_size / 5

    @ti.kernel
    def compute_loss_actuation(self, steps: ti.template()):
        for t, k, i in ti.ndrange(steps, self.batch_size, self.n_springs):
            if self.target_h[t, k] < 0.1 + 1e-4:
                self.loss_act[None] += ti.max(ti.abs(self.actuation[t, k, i]) - (ti.abs(self.target_v[t, k][0]) / 0.08) ** 0.5,
                                         0.) / self.n_springs / self.batch_size / steps * 10

    @ti.kernel
    def compute_loss_final(self, l: ti.template()):
        self.loss[None] += l[None]

    def get_loss(self, steps, loss_enable, *args, **kwargs):
        if "velocity" in loss_enable:
            self.compute_loss_velocity(steps)
        if "height" in loss_enable:
            self.compute_loss_height(steps)
        if "pose" in loss_enable:
            self.compute_loss_pose(steps)
        if "actuation" in loss_enable:
            self.compute_loss_actuation(steps)
        if "rotation" in loss_enable:
            self.compute_loss_rotation(steps)

        for l in self.losses:
            self.compute_loss_final(l)

    @ti.kernel
    def initialize_interactive(self, steps: ti.template(), output_v: ti.f64, output_h: ti.f64):
        for t, k in ti.ndrange(steps, self.batch_size):
            self.target_v[t, k][0] = output_v
            self.target_h[t, k] = output_h

    @ti.kernel
    def initialize_script(self, steps: ti.template(), x0: real, y0: real, x1: real, y1: real, x2: real, y2: real, x3: real,
                          y3: real):
        for t, k in ti.ndrange(steps, self.batch_size):
            if t < 1000:
                self.target_v[t, k][0] = x0
                self.target_v[t, k][2] = y0
            elif t < 2000:
                self.target_v[t, k][0] = x1
                self.target_v[t, k][2] = y1
            elif t < 3000:
                self.target_v[t, k][0] = x2
                self.target_v[t, k][2] = y2
            elif t < 4000:
                self.target_v[t, k][0] = x3
                self.target_v[t, k][2] = y3
            self.target_h[t, k] = 0.

    @ti.kernel
    def initialize_validate(self, steps: ti.template(), output_v: ti.f64, output_h: ti.f64):
        '''
        for t, k in ti.ndrange(steps, batch_size):
            q = t // turn_period
            if q % 3 == 0:
                target_v[t, k][0] = (q // 3 % 2 * 2 - 1) * output_v
                target_h[t, k] = 0.1
            elif q % 3 == 1:
                target_v[t, k][0] = 0
                target_h[t, k] = output_h
            else:
                target_v[t, k][0] = 0
                target_h[t, k] = 0.1
            if ti.static(dim == 3):
                target_v[t, k][0] = ((t // turn_period) % 2 * 2 - 1) * output_v
                target_v[t, k][2] = 0
                target_h[t, k] = 0
        '''
        for t, k in ti.ndrange(steps, self.batch_size):  # jump
            # if steps < 500:
            self.target_v[t, k][0] = output_v
            self.target_h[t, k] = output_h
        # else:
        #    target_v[t, k][0] = output_v
        #    target_h[t, k] = 0
        '''
        if output_h < 1.0 + 1e-8:
            for t, k in ti.ndrange(steps, batch_size):
                target_v[t, k][0] = ((t // turn_period) % 2 * 2 - 1) * output_v
                target_h[t, k] = 0
        for t, k in ti.ndrange(steps, batch_size):
            q = t // turn_period
            if q % 2 == 0:
                target_v[t, k][0] = (q // 2 % 2 * 2 - 1) * output_v
                target_h[t, k] = 0.1
            else:
                target_v[t, k][0] = 0
                target_h[t, k] = output_h
        '''

    @ti.kernel
    def initialize_train(self, iter: ti.i32, steps: ti.template(), max_velocity: ti.f64, max_height: ti.f64):
        times = steps // self.turn_period + 1
        for _ in range(self.batch_size * times * 3):
            self.pool[_] = ti.random()
        # Define multi-tasks here
        for t, k in ti.ndrange(steps, self.batch_size):
            q = (t // self.turn_period * self.batch_size + k) * 3
            if ti.static(self.dim == 2):
                target_id = int(self.pool[q] * 4)
                # print('Iter:', int(self.iter), 'Step:', int(t), 'Current task:', int(target_id))
                if target_id == 1:
                    # Move backward or forward
                    self.target_v[t, k][0] = (self.pool[q + 1] * 2 - 1) * max_velocity
                    self.target_h[t, k] = 0.1
                    self.target_c[t, k] = 0
                elif target_id == 2:
                    # Move backward or forward & jump
                    self.target_v[t, k][0] = (self.pool[q + 1] * 2 - 1) * max_velocity
                    self.target_h[t, k] = self.pool[q + 2] * max_height + 0.1
                    self.target_c[t, k] = 0
                elif target_id == 3:
                    # Move backward or forward & crawl
                    self.target_v[t, k][0] = (self.pool[q + 1] * 2 - 1) * max_velocity
                    self.target_h[t, k] = 0.1
                    self.target_c[t, k] = 1.
                else:
                    # Keep still
                    self.target_v[t, k][0] = 0
                    self.target_h[t, k] = 0.1
                    self.target_c[t, k] = 0
            else:
                r = ti.sqrt(self.pool[q + 1])
                angle = self.pool[q + 2] * 2 * 3.1415926
                # r = 1.
                # angle = 0.
                self.target_v[t, k][0] = r * ti.cos(angle) * 0.05
                self.target_v[t, k][2] = r * ti.sin(angle) * 0.05
                self.target_h[t, k] = 0.

    @debug
    def visualizer(self, steps, prefix):
        interval = self.output_vis_interval
        os.makedirs('video/{}/'.format(prefix), exist_ok=True)

        for t in range(1, steps):
            if (t + 1) % interval == 0:
                self.gui.clear()
                self.gui.line((0, self.ground_height),
                              (1, self.ground_height),
                              color=0x000022,
                              radius=3)
                self.gui.line((0, self.target_h[t]), (1, self.target_h[t]), color=0x002200)
                self.solver.draw_robot(self.gui, t, self.target_v)
                self.gui.show('video/{}/{:04d}.png'.format(prefix, t))

    @ti.kernel
    def reset_robot(self, start: ti.template(), step: ti.template(), times: ti.template()):
        for k, i in ti.ndrange(times, self.n_objects):
            self.x[0, k * step + start, i] = self.initial_objects[i]

    @ti.kernel
    def get_center(self):
        for I in ti.grouped(self.initial_objects):
            self.initial_center[None] += self.initial_objects[I] / self.n_objects

    def setup_robot(self):
        self.logger.info(f'n_objects={self.n_objects}, n_springs= {self.n_springs}')
        self.initial_objects.from_numpy(np.array(self.objects))
        for i in range(self.n_objects):
            self.initial_objects[i][0] += 0.4
        self.get_center()
        self.reset_robot(0, 1, self.batch_size)
        self.solver.initialize_robot()

    @ti.kernel
    def copy_robot(self, steps: ti.i32):
        for k, i in ti.ndrange(self.batch_size, self.n_objects):
            self.x[0, k, i] = self.x[steps, k, i]
            self.v[0, k, i] = self.v[steps, k, i]

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def before_step(self):
        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()

    def after_train(self):
        for h in self._hooks:
            h.after_train()

    def run_step(self):
        raise NotImplementedError

    def train(self, start_iter, max_iter):
        # logger = logging.getLogger(__name__)
        self.logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        try:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
            # self.iter == max_iter can be used by `after_train` to
            # tell whether the training successfully finished or failed
            # due to exceptions.
            self.iter += 1
        except Exception:
            self.logger.exception("Exception during training:")
            raise
        finally:
            self.after_train()

    def register_hooks(self, hooks):
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

#
# class DiffPhyTrainer(BaseTrainer):
#     def __init__(self):
#         super(DiffPhyTrainer, self).__init__()
#         # Initialize neural network model
#         self.nn = Model(max_steps, batch_size, n_input_states, n_springs,
#                         self.input_state, self.actuation, n_hidden)
#
#     @debug
#     def simulate(self, steps,
#                  output_v=None,
#                  output_h=None,
#                  train=True,
#                  iter=0,
#                  max_speed=0.08,
#                  max_height=0.1,
#                  *args,
#                  **kwargs):
#         # clean up cache and set up control sequence
#         self.solver.clear_states(steps)
#         self.nn.clear()
#         if train:
#             self.initialize_train(iter, steps, max_speed, max_height)
#         elif not train and dim == 2:
#             self.initialize_validate(steps, output_v, output_h)
#         elif not train and dim == 3:
#             self.initialize_script(steps, 0.04, 0, 0, 0.04, -0.04, 0, 0, -0.04)
#         self.loss[None] = 0.
#         for l in self.losses:
#             l[None] = 0.
#         # start simulation
#         if train:
#             with ti.Tape(self.loss):
#                 for t in range(steps + 1):
#                     self.nn_input(t, 0, max_speed, max_height)
#                     self.nn.forward(t)
#                     self.solver.advance(t)
#                 self.get_loss(steps, *args, **kwargs)
#         else:
#             for t in range(steps + 1):
#                 self.nn_input(t, 0, max_speed, max_height)
#                 self.nn.forward(t)
#                 self.solver.advance(t)
#             self.visualizer(steps, prefix=str(output_v) + "_" + str(output_h))
#             if dim == 3:
#                 x_ = self.x.to_numpy()
#                 t = threading.Thread(target=output_mesh, args=(steps, x_, str(output_v) + '_' + str(output_h)))
#                 t.start()
#
#     def train(self, iters=100000, change_iter=5000, prefix=None, root_dir="./", \
#                  load_path=None, *args, **kwargs):
#         log_dir = os.path.join(root_dir, "logs")
#         plot_dir = os.path.join(root_dir, "plots")
#         weights_dir = os.path.join(root_dir, "weights")
#
#         if prefix is not None:
#             weights_dir = os.path.join(weights_dir, prefix)
#
#         os.makedirs(plot_dir, exist_ok=True)
#         os.makedirs(weights_dir, exist_ok=True)
#         os.makedirs(log_dir, exist_ok=True)
#
#         log_name = "training.log"
#         if prefix is not None:
#             log_name = "{}_training.log".format(prefix)
#         log_path = os.path.join(log_dir, log_name)
#
#         log_file = open(log_path, 'w')
#         log_file.close()
#
#         plot_name = "training_curve.png"
#         plot200_name = "training_curve_last_200.png"
#         if prefix is not None:
#             plot_name = "{}_training_curve.png".format(prefix)
#             plot200_name = "{}_training_curve_last_200.png".format(prefix)
#         plot_path = os.path.join(plot_dir, plot_name)
#         plot200_path = os.path.join(plot_dir, plot200_name)
#
#         weight_out = lambda x: os.path.join(weights_dir, x)
#
#         self.setup_robot()
#
#         if load_path is not None and os.path.exists(load_path):
#             print("load from {}".format(load_path))
#             self.nn.load_weights(load_path)
#         else:
#             self.nn.weights_init()
#
#         self.nn.clear_adam()
#
#         losses = []
#         best = 1e+15
#         best_finetune = 1e+15
#         train_steps = 1000
#         if dim == 3 and sys.argv[0] == "validate.py":
#             train_steps = 4000
#
#         reset_step = 2
#
#         for iter in range(iters):
#             if iter > change_iter:
#                 if iter % 500 == 0 and reset_step < max_reset_step:
#                     reset_step += 1
#                 self.rounded_train(train_steps, iter, reset_step=reset_step)
#
#             print("-------------------- {}iter #{} --------------------" \
#                   .format("" if prefix is None else "{}, ".format(prefix), iter))
#
#             self.simulate(train_steps, iter=iter, *args, **kwargs)
#
#             if iter <= change_iter and self.loss[None] < best:
#                 best = self.loss[None]
#                 self.nn.dump_weights(weight_out("best.pkl"))
#
#             if iter > change_iter + max_reset_step and self.loss[None] < best_finetune:
#                 best_finetune = self.loss[None]
#                 self.nn.dump_weights(weight_out("best_finetune.pkl"))
#
#             self.nn.dump_weights(weight_out("last.pkl"))
#             self.nn.dump_weights(os.path.join(root_dir, "weight.pkl"))
#
#             if iter % 50 == 0:
#                 self.nn.dump_weights(weight_out("iter{}.pkl".format(iter)))
#
#             total_norm_sqr = self.nn.get_TNS()
#
#             def print_logs(file=None):
#                 if iter > change_iter:
#                     print('Iter=', iter, 'Loss=', self.loss[None], 'Best_FT=', best_finetune, file=file)
#                 else:
#                     print('Iter=', iter, 'Loss=', self.loss[None], 'Best=', best, file=file)
#                 print("TNS= ", total_norm_sqr, file=file)
#                 for name, l in self.loss_dict.items():
#                     print("{}={}".format(name, l[None]), file=file)
#
#             print_logs()
#             log_file = open(log_path, "a")
#             print_logs(log_file)
#             log_file.close()
#
#             self.nn.gradient_update(iter)
#             losses.append(self.loss[None])
#
#             if iter % 100 == 0 or iter % 10 == 0 and iter < 500:
#                 plot_curve(losses, plot_path)
#                 plot_curve(losses[-200:], plot200_path)
#
#         return losses
#
#
# def output_mesh(steps, x_, fn):
#     os.makedirs('video/' + fn + '_objs', exist_ok=True)
#     for t in range(1, steps):
#         f = open('video/' + fn + f'_objs/{t:06d}.obj', 'w')
#         for i in range(n_objects):
#             f.write('v %.6f %.6f %.6f\n' % (x_[t, 0, i, 0], x_[t, 0, i, 1], x_[t, 0, i, 2]))
#         for [p0, p1, p2] in faces:
#             f.write('f %d %d %d\n' % (p0 + 1, p1 + 1, p2 + 1))
#         f.close()


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
