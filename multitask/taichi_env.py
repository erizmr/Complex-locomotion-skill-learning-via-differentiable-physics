import os
import sys
import math
import time
import itertools
import taichi as ti
import numpy as np
from collections import defaultdict
from multitask.utils import Debug, real, plot_curve, load_string, scalar, vec, mat
from multitask.solver_mass_spring import SolverMassSpring
from multitask.solver_mpm import SolverMPM


debug = Debug(False)
# ti_random_seed = int(time.time() * 1e6) % 10000
ti_random_seed = 930
ti.init(arch=ti.gpu, default_fp=real, random_seed=ti_random_seed, packed = True)


# Manage all shared parameters and taichi fields
@ti.data_oriented
class TaichiEnv:
    def __init__(self, config, train=True):
        self.config_ = config
        self.config = config.get_config()
        self.logger = config.get_logger(name=__name__)
        self.dim = self.config["robot"]["dim"]
        self.max_steps = self.config["process"]["max_steps"]  # max steps for a simulation
        self.batch_size = self.config["nn"]["batch_size"]
        self.n_hidden = self.config["nn"]["n_hidden"]
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
        self.duplicate_c = self.config["nn"]["duplicate_c"]
        self.output_vis_interval = self.config["process"]["output_vis_interval"]
        self.ground_height = self.config["simulator"]["ground_height"]

        if "n_models" in self.config["nn"].keys() and not train:
            self.n_models = self.config["nn"]["n_models"]
        else:
            self.n_models = 1
        
        self.default_model_id = 0

        self.loss_types_collections = {"velocity", "height", "pose", "actuation", "rotation", "crawl"}
        # Get the task, and ensure that the task loss is actually defined
        self.task = list(self.config["train"]["task"])
        for tsk in self.task:
            assert tsk in self.loss_types_collections

        # Remove the loss types that are not used in the task during training
        self.validate_targets = self.config["validation"]
        _to_remove = []
        for k in self.validate_targets.keys():
            if k not in self.task:
                _to_remove.append(k)
        for k in _to_remove:
            self.validate_targets.pop(k, None)

        self.validate_targets_values = defaultdict(list)
        # Construct the validation matrix i.e., combinations of all validation targets
        targets_keys = []
        targets_values = []
        for k, v in self.validate_targets.items():
            targets_keys.append(k)
            targets_values.append(v)
        validation_matrix = list(itertools.product(*targets_values))
        for element in validation_matrix:
            for i, name in enumerate(targets_keys):
                self.validate_targets_values[name].append(element[i])

        self.loss = scalar()
        self.loss_velocity = scalar()
        self.loss_height = scalar()
        self.loss_pose = scalar()
        self.loss_rotation = scalar()
        self.loss_weight = scalar()
        self.loss_act = scalar()
        self.loss_crawl = scalar()
        self.loss_dict = {'loss_velocity': self.loss_velocity,
                          'loss_height': self.loss_height,
                          'loss_pose': self.loss_pose,
                          'loss_rotation': self.loss_rotation,
                          'loss_weight': self.loss_weight,
                          'loss_actuation': self.loss_act,
                          'loss_crawl': self.loss_crawl}
        self.losses = self.loss_dict.values()

        ti.root.place(self.loss)
        ti.root.place(*self.losses)

        # For validation use
        self.loss_batch = scalar()
        self.loss_velocity_batch = scalar()
        self.loss_height_batch = scalar()
        self.loss_pose_batch = scalar()
        self.loss_rotation_batch = scalar()
        self.loss_weight_batch = scalar()
        self.loss_act_batch = scalar()
        self.loss_crawl_batch = scalar()
        self.loss_dict_batch = {'loss_velocity': self.loss_velocity_batch,
                                'loss_height': self.loss_height_batch,
                                'loss_pose': self.loss_pose_batch,
                                'loss_rotation': self.loss_rotation_batch,
                                'loss_weight': self.loss_weight_batch,
                                'loss_actuation': self.loss_act_batch,
                                'loss_crawl': self.loss_crawl_batch}

        # losses for each batch
        self.losses_batch = self.loss_dict_batch.values()
        batch_node = ti.root.dense(ti.ij, (self.n_models, self.batch_size))
        batch_node.place(self.loss_batch)
        batch_node.place(*self.losses_batch)

        # Robot objects
        self.initial_objects = vec(self.dim)
        self.initial_center = vec(self.dim)
        self.n_objects = self.config["robot"]["n_objects"]
        ti.root.dense(ti.i, self.n_objects).place(self.initial_objects)
        ti.root.place(self.initial_center)

        # NN input state
        self.input_state = scalar()
        ti.root.dense(ti.ijkl, (self.n_models, self.max_steps, self.batch_size, self.n_input_states)).place(self.input_state)

        # Target velocity, height, crawl
        self.target_v, self.target_h, self.target_c = vec(self.dim), scalar(), scalar(),
        ti.root.dense(ti.ij, (self.max_steps, self.batch_size)).place(self.target_v, self.target_h, self.target_c)

        # Record the current max height
        self.current_max_height = scalar()
        ti.root.place(self.current_max_height)

        # Initialize a simulator
        self.solver = SolverMPM() if self.simulator == "mpm" else SolverMassSpring(config)
        # self.sovler = solver
        self.solver_x = self.solver.x
        self.solver_v = self.solver.v
        self.solver_center = self.solver.center
        self.solver_height = self.solver.height
        self.solver_upper_height = self.solver.upper_height
        self.solver_rotation = self.solver.rotation
        self.solver_actuation = self.solver.actuation

        self.pool = ti.field(ti.f64, shape=(5 * self.batch_size * (1000 // self.turn_period + 1)))
        self.gui = ti.GUI(show_gui=False, background_color=0xFFFFFF)

    @ti.kernel
    def nn_input(self, t: ti.i32, offset: ti.i32, max_speed: ti.f64, max_height: ti.f64):
        if ti.static(self.n_sin_waves > 0):
            for model_id, k, j in ti.ndrange(self.n_models, self.batch_size, self.n_sin_waves):
                self.input_state[model_id, t, k, j] = ti.sin(
                    self.spring_omega * (t + offset) * self.dt + 2 * math.pi / self.n_sin_waves * j)
        for model_id, k, j in ti.ndrange(self.n_models, self.batch_size, self.n_objects):
            vec_x = self.solver_x[model_id, t, k, j] - self.solver_center[model_id, t, k]
            for d in ti.static(range(self.dim)):
                if ti.static(self.dim == 2):
                    self.input_state[model_id, t, k, j * self.dim * 2 + self.n_sin_waves + d] = vec_x[d] / 0.2
                    self.input_state[model_id, t, k, j * self.dim * 2 + self.n_sin_waves + self.dim + d] = 0
                else:
                    self.input_state[model_id, t, k, j * self.dim * 2 + self.n_sin_waves + d] = vec_x[d] * 0.04
                    self.input_state[model_id, t, k, j * self.dim * 2 + self.n_sin_waves + self.dim + d] = 0

        if ti.static(self.duplicate_v > 0):
            if ti.static(self.dim == 2):
                for model_id, k, j in ti.ndrange(self.n_models, self.batch_size, self.duplicate_v):
                    self.input_state[model_id, t, k, self.n_objects * self.dim * 2 + self.n_sin_waves + j * (self.dim - 1)] = \
                    self.target_v[t, k][0] / max_speed
            else:
                for model_id, k, j in ti.ndrange(self.n_models, self.batch_size, self.duplicate_v):
                    self.input_state[model_id, t, k, self.n_objects * self.dim * 2 + self.n_sin_waves + j * (self.dim - 1)] = \
                    self.target_v[t, k][0] * 0.15
                    self.input_state[model_id, t, k, self.n_objects * self.dim * 2 + self.n_sin_waves + j * (self.dim - 1) + 1] = \
                    self.target_v[t, k][2] * 0.15
        if ti.static(self.duplicate_h > 0):
            for model_id, k, j in ti.ndrange(self.n_models, self.batch_size, self.duplicate_h):
                self.input_state[model_id, t, k, self.n_objects * self.dim * 2 + self.n_sin_waves + self.duplicate_v * (
                            self.dim - 1) + j] = (self.target_h[
                                                      t, k] - 0.1) / max_height * 2 - 1

        if ti.static(self.duplicate_c > 0):
            for model_id, k, j in ti.ndrange(self.n_models, self.batch_size, self.duplicate_c):
                self.input_state[model_id, t, k, self.n_objects * self.dim * 2 + self.n_sin_waves + self.duplicate_v * (self.dim - 1) + self.duplicate_h + j] = self.target_c[t, k]

    @ti.kernel
    def compute_loss_velocity(self, steps: ti.template()):
        for model_id, t, k in ti.ndrange(self.n_models, (self.run_period, steps + 1), self.batch_size):
            if t % self.turn_period > self.run_period:  # and target_h[t - run_period, k] < 0.1 + 1e-4:
                if ti.static(self.dim == 2):
                    loss_x = (self.solver_center[model_id, t, k](0) - self.solver_center[model_id, t - self.run_period, k][0] - self.target_v[
                        t - self.run_period, k][0] / (1 + self.target_c[
                        t - self.run_period, k])) ** 2 / self.batch_size / steps * 100
                    self.loss_velocity[None] += loss_x
                    self.loss_velocity_batch[model_id, k] += loss_x * self.batch_size
                else:
                    loss_x = (self.solver_center[model_id, t, k](0) - self.solver_center[model_id, t - self.run_period, k](0) - self.target_v[
                        t - self.run_period, k](0)) ** 2 / self.batch_size
                    loss_y = (self.solver_center[model_id, t, k](2) - self.solver_center[model_id, t - self.run_period, k](2) - self.target_v[
                        t - self.run_period, k](2)) ** 2 / self.batch_size

                    self.loss_velocity[None] += loss_x + loss_y
                    self.loss_velocity_batch[model_id, k] += (loss_x + loss_y) * self.batch_size
        # if k == 0:
        #     print("Mark run: ", center[t, 0](0) - center[t - run_period, 0](0), target_v[t - run_period, 0](0))

    @ti.kernel
    def compute_loss_height(self, steps: ti.template()):
        for model_id, t, k in ti.ndrange(self.n_models, (1, steps + 1), self.batch_size):
            if t % self.jump_period == self.jump_period - 1 and self.target_h[t, k] > 0.1 + 1e-6:
                loss_h = (self.solver_height[model_id, t, k] - self.target_h[t, k]) ** 2 / self.batch_size / (
                            steps // self.jump_period) * 100
                self.loss_height[None] += loss_h
                self.loss_height_batch[model_id, k] += loss_h * self.batch_size
                if self.solver_height[model_id, t, k] > self.current_max_height[None]:
                    self.current_max_height[None] = self.solver_height[model_id, t, k]

    @ti.kernel
    def compute_loss_pose(self, steps: ti.template()):
        # TODO: This doesn't work for 3D
        for model_id, t, k, i in ti.ndrange(self.n_models, (1, steps + 1), self.batch_size, self.n_objects):
            if t % self.jump_period == 0:
                # dist2 = sum((x[t, k, i] - center[t, k] - initial_objects[i] + initial_center[None]) ** 2)
                dist2 = sum((self.solver_x[model_id, t, k, i] - self.initial_objects[i]) ** 2)
                loss_p = dist2 / self.batch_size / (steps // self.jump_period)
                self.loss_pose[None] += loss_p
                self.loss_pose_batch[model_id, k] += loss_p * self.batch_size

    @ti.kernel
    def compute_loss_rotation(self, steps: ti.template()):
        for model_id, t, k in ti.ndrange(self.n_models, (1, steps + 1), self.batch_size):
            loss_r = self.solver_rotation[model_id, t, k] ** 2 / self.batch_size / 5
            self.loss_rotation[None] += loss_r
            self.loss_rotation_batch[model_id, k] += loss_r * self.batch_size

    @ti.kernel
    def compute_loss_actuation(self, steps: ti.template()):
        for model_id, t, k, i in ti.ndrange(self.n_models, steps, self.batch_size, self.n_springs):
            if self.target_h[t, k] < 0.1 + 1e-4:
                loss_a = ti.max(ti.abs(self.solver_actuation[model_id, t, k, i]) - (ti.abs(self.target_v[t, k][0]) / 0.08) ** 0.5,
                                0.) / self.n_springs / self.batch_size / steps * 10
                # loss_a = ti.max(ti.abs(self.solver_actuation[t, k, i]) - (ti.abs(self.target_v[t, k][0]) / 0.16 + ti.abs(self.target_h[t, k]) /0.16 ) ** 0.5,
                #                          0.) / self.n_springs / self.batch_size / steps * 10
                self.loss_act[None] += loss_a
                self.loss_act_batch[model_id, k] += loss_a * self.batch_size

    @ti.kernel
    def compute_loss_crawl(self, steps: ti.template()):
        for model_id, t, k in ti.ndrange(self.n_models, (1, steps + 1), self.batch_size):
            if self.target_c[t, k] > 1 - 1e-4:
                loss_c = max(self.solver_upper_height[model_id, t, k] - 0.1, 0) ** 2 / self.batch_size / steps * 5.
                self.loss_crawl[None] += loss_c
                self.loss_crawl_batch[model_id, k] += loss_c * self.batch_size

    @ti.kernel
    def compute_loss_final(self, l: ti.template()):
        self.loss[None] += l[None]

    @ti.kernel
    def compute_loss_final_batch(self, l_batch: ti.template()):
        for model_id, k in ti.ndrange(self.n_models, self.batch_size):
            self.loss_batch[model_id, k] += l_batch[model_id, k]

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
        if "crawl" in loss_enable:
            self.compute_loss_crawl(steps)

        for l in self.losses:
            self.compute_loss_final(l)

        # Compute loss for each batch
        for l in self.losses_batch:
            self.compute_loss_final_batch(l)

    @ti.kernel
    def initialize_interactive(self, steps: ti.template(), output_v: ti.f64, output_h: ti.f64, output_c: ti.f64):
        for t, k in ti.ndrange(steps, self.batch_size):
            self.target_v[t, k][0] = output_v
            self.target_h[t, k] = output_h
            self.target_c[t, k] = output_c

    @ti.kernel
    def initialize_script(self, steps: ti.template(), x0: real, y0: real, x1: real, y1: real, x2: real, y2: real,
                          x3: real,
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
    def initialize_validate(self, steps: ti.template(), output_v: ti.ext_arr(), output_h: ti.ext_arr(), output_c: ti.ext_arr()):
        for t, k in ti.ndrange(steps, self.batch_size):
            self.target_v[t, k][0] = output_v[k]
            self.target_h[t, k] = output_h[k]
            self.target_c[t, k] = output_c[k]

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
                # print('Iter:', int(t), 'Step:', int(t), 'Current task:', int(target_id))
                # if len(self.task) == 1 and self.task[0] == "height":
                #     target_id = 4
                if target_id == 1:
                    # print("f&b")
                    # Move backward or forward
                    self.target_v[t, k][0] = (self.pool[q + 1] * 2 - 1) * max_velocity
                    # self.target_v[t, k][0] = 0.04
                    self.target_h[t, k] = 0.1
                    self.target_c[t, k] = 0
                elif target_id == 2:
                    # print("f&b jump")
                    # Move backward or forward & jump
                    self.target_v[t, k][0] = (self.pool[q + 1] * 2 - 1) * max_velocity
                    self.target_h[t, k] = self.pool[q + 2] * max_height + 0.1
                    self.target_c[t, k] = 0
                elif target_id == 3:
                    # print("f&b&c")
                    # Move backward or forward & crawl0
                    self.target_v[t, k][0] = (self.pool[q + 1] * 2 - 1) * max_velocity
                    self.target_h[t, k] = 0.1
                    self.target_c[t, k] = 1.
                # elif target_id == 4:
                #     # jump only
                #     self.target_v[t, k][0] = 0.
                #     self.target_h[t, k] = self.pool[q + 2] * max_height + 0.1
                #     self.target_c[t, k] = 0.
                else:
                    # print("still")
                    # Keep still
                    self.target_v[t, k][0] = 0
                    self.target_h[t, k] = 0.1
                    self.target_c[t, k] = 0
            else:
                r = ti.sqrt(self.pool[q + 1])
                angle = self.pool[q + 2] * 2 * 3.1415926
                r = 1.
                angle = 0.
                self.target_v[t, k][0] = r * ti.cos(angle) * 0.05
                self.target_v[t, k][2] = r * ti.sin(angle) * 0.05
                self.target_h[t, k] = 0.1

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
        for model_id, k, i in ti.ndrange(self.n_models, times, self.n_objects):
            self.solver_x[model_id, 0, k * step + start, i] = self.initial_objects[i]

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
        for model_id, k, i in ti.ndrange(self.n_models, self.batch_size, self.n_objects):
            self.solver_x[model_id, 0, k, i] = self.solver_x[model_id, steps, k, i]
            self.solver_v[model_id, 0, k, i] = self.solver_v[model_id, steps, k, i]

    @ti.kernel
    def refresh_xv(self):
        for model_id, i in ti.ndrange(self.n_models, self.n_objects):
            self.solver_x[model_id, 0, 0, i] = self.solver_x[model_id, 1, 0, i]
            self.solver_v[model_id, 0, 0, i] = self.solver_v[model_id, 1, 0, i]

