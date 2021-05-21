import threading
from taichi.lang.impl import reset

from utils import *
from config import *
from nn import *
from solver_mass_spring import SolverMassSpring
from solver_mpm import SolverMPM

import random
import sys
import matplotlib.pyplot as plt
import taichi as ti
import math
import numpy as np
import os

import pickle as pkl

debug = Debug(False)

ti.init(arch=ti.gpu, default_fp=real)

output_target = []
output_sim = []

loss = scalar()
loss_velocity = scalar()
loss_height = scalar()
loss_pose = scalar()
loss_rotation = scalar()
loss_weight = scalar()
loss_act = scalar()
loss_dict = {'loss_v': loss_velocity,
             'loss_h': loss_height,
             'loss_p': loss_pose,
             'loss_r': loss_rotation,
             'loss_w': loss_weight,
             'loss_a': loss_act}
losses = loss_dict.values()
ti.root.place(loss)
ti.root.place(*losses)

initial_objects = vec()
initial_center = vec()
ti.root.dense(ti.i, n_objects).place(initial_objects)
ti.root.place(initial_center)

input_state = scalar()
ti.root.dense(ti.ijk, (max_steps, batch_size, n_input_states)).place(input_state)

target_v, target_h = vec(), scalar()
ti.root.dense(ti.ij, (max_steps, batch_size)).place(target_v, target_h)

solver = SolverMPM() if simulator == "mpm" else SolverMassSpring()
x = solver.x
v = solver.v
center = solver.center
height = solver.height
rotation = solver.rotation
actuation = solver.actuation

#nn = Model(max_steps, batch_size, n_input_states, n_springs, input_state, actuation, n_hidden)

#ti.root.lazy_grad()

pool = ti.field(ti.f64, shape = (5 * batch_size * (1000 // turn_period + 1)))

@ti.kernel
def nn_input(t: ti.i32, offset: ti.i32, max_speed: ti.f64, max_height: ti.f64):
    for k, j in ti.ndrange(batch_size, n_sin_waves):
        input_state[t, k, j] = ti.sin(spring_omega * (t + offset) * dt + 2 * math.pi / n_sin_waves * j)

    for k, j in ti.ndrange(batch_size, n_objects):
        vec_x = x[t, k, j] - center[t, k]
        for d in ti.static(range(dim)):
            if ti.static(dim == 2):
                input_state[t, k, j * dim * 2 + n_sin_waves + d] = vec_x[d] / 0.05
                input_state[t, k, j * dim * 2 + n_sin_waves + dim + d] = v[t, k, j][d]
            else:
                input_state[t, k, j * dim * 2 + n_sin_waves + d] = vec_x[d] * float(sys.argv[2])
                input_state[t, k, j * dim * 2 + n_sin_waves + dim + d] = 0

    if ti.static(duplicate_v > 0):
        if ti.static(dim == 2):
            for k, j in ti.ndrange(batch_size, duplicate_v):
                input_state[t, k, n_objects * dim * 2 + n_sin_waves + j * (dim - 1)] = target_v[t, k][0] / max_speed
        else:
            for k, j in ti.ndrange(batch_size, duplicate_v):
                input_state[t, k, n_objects * dim * 2 + n_sin_waves + j * (dim - 1)] = target_v[t, k][0] * float(sys.argv[3])
                input_state[t, k, n_objects * dim * 2 + n_sin_waves + j * (dim - 1) + 1] = target_v[t, k][2] * float(sys.argv[3])
    if ti.static(duplicate_h > 0):
        for k, j in ti.ndrange(batch_size, duplicate_h):
            input_state[t, k, n_objects * dim * 2 + n_sin_waves + duplicate_v * (dim - 1) + j] = (target_h[t, k] - 0.1) / max_height * 2 - 1

@ti.kernel
def compute_loss_velocity(steps: ti.template()):
    for t, k in ti.ndrange((run_period, steps + 1), batch_size):
        if t % turn_period > run_period:# and target_h[t - run_period, k] < 0.1 + 1e-4:
            if ti.static(dim == 2):
                loss_velocity[None] += (center[t, k](0) - center[t - run_period, k](0) - target_v[t - run_period, k](0))**2 / batch_size
            else:
                loss_velocity[None] += (center[t, k](0) - center[t - run_period, k](0) - target_v[t - run_period, k](0))**2 / batch_size
                loss_velocity[None] += (center[t, k](2) - center[t - run_period, k](2) - target_v[t - run_period, k](2))**2 / batch_size
    # if k == 0:
    #     print("Mark run: ", center[t, 0](0) - center[t - run_period, 0](0), target_v[t - run_period, 0](0))


@ti.kernel
def compute_loss_height(steps: ti.template()):
    for t, k in ti.ndrange((1, steps + 1), batch_size):
        if t % jump_period == jump_period - 1 and target_h[t, k] > 0.1:
            loss_height[None] += (height[t, k] - target_h[t, k]) ** 2 / batch_size / (steps // jump_period) * 100


@ti.kernel
def compute_loss_pose(steps: ti.template()):
    # TODO: This doesn't work for 3D
    for t, k, i in ti.ndrange((1, steps + 1), batch_size, n_objects):
        if t % jump_period == 0:
            #dist2 = sum((x[t, k, i] - center[t, k] - initial_objects[i] + initial_center[None]) ** 2)
            dist2 = sum((x[t, k, i] - initial_objects[i]) ** 2)
            loss_pose[None] += dist2 / batch_size / (steps // jump_period)

@ti.kernel
def compute_loss_rotation(steps: ti.template()):
    for t, k in ti.ndrange((1, steps + 1), batch_size):
        loss_rotation[None] += rotation[t, k] ** 2 / batch_size / 5

@ti.kernel
def compute_loss_actuation(steps: ti.template()):
    for t, k, i in ti.ndrange(steps, batch_size, n_springs):
        if target_h[t, k] < 0.1 + 1e-4:
            loss_act[None] += ti.max(ti.abs(actuation[t, k, i]) - (ti.abs(target_v[t, k][0]) / 0.08) ** 0.5, 0.) / n_springs / batch_size / steps * 10

@ti.kernel
def compute_loss_final(l: ti.template()):
    loss[None] += l[None]

def get_loss(steps, loss_enable, *args, **kwargs):
    if "velocity" in loss_enable:
        compute_loss_velocity(steps)
    if "height" in loss_enable:
        compute_loss_height(steps)
    if "pose" in loss_enable:
        compute_loss_pose(steps)
    if "actuation" in loss_enable:
        compute_loss_actuation(steps)
    if "rotation" in loss_enable:
        compute_loss_rotation(steps)

    for l in losses:
        compute_loss_final(l)

gui = ti.GUI(show_gui=False, background_color=0xFFFFFF)

@ti.kernel
def initialize_interactive(steps: ti.template(), output_v: ti.f64, output_h: ti.f64):
    for t, k in ti.ndrange(steps, batch_size):
        target_v[t, k][0] = output_v
        target_h[t, k] = output_h

@ti.kernel
def initialize_script(steps: ti.template(), x0:real, y0:real, x1:real, y1:real, x2:real, y2:real, x3:real, y3:real):
    for t, k in ti.ndrange(steps, batch_size):
        if t < 1000:
            target_v[t, k][0] = x0
            target_v[t, k][2] = y0
        elif t < 2000:
            target_v[t, k][0] = x1
            target_v[t, k][2] = y1
        elif t < 3000:
            target_v[t, k][0] = x2
            target_v[t, k][2] = y2
        elif t < 4000:
            target_v[t, k][0] = x3
            target_v[t, k][2] = y3
        target_h[t, k] = 0.

@ti.kernel
def initialize_validate(steps: ti.template(), output_v: ti.f64, output_h: ti.f64):
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
    for t, k in ti.ndrange(steps, batch_size): # jump
        #if steps < 500:
            target_v[t, k][0] = output_v
            target_h[t, k] = 0.1
        #else:
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
def initialize_train(iter: ti.i32, steps: ti.template(), max_speed: ti.f64, max_height: ti.f64):
    times = steps // turn_period + 1
    for _ in range(batch_size * times * 3):
        pool[_] = ti.random()
    for t, k in ti.ndrange(steps, batch_size):
        q = (t // turn_period * batch_size + k) * 3
        if ti.static(dim == 2):
                '''
            if iter < 500:
                if k < batch_size / 2:
                    target_v[t, k][0] = (k / (batch_size / 2 - 1)) * 2 - 1 * max_speed
                    target_h[t, k] = 0.1
                else:
                    target_v[t, k][0] *= 0.
                    target_h[t, k] = ((k - batch_size / 2) / (batch_size / 2 - 1)) * max_height + 0.1
            else:
                '''
                #if pool[q + 0] < 0.5:
                target_v[t, k][0] = ((pool[q + 1] > 0.5) * 2 - 1) * max_speed
                target_h[t, k] = 0.1
                #else:
                #    target_v[t, k] *= 0.
                #    target_h[t, k] = pool[q + 1] * max_height + 0.1
        else:
            r = ti.sqrt(pool[q + 1])
            angle = pool[q + 2] * 2 * 3.1415926
            # r = 1.
            # angle = 0.
            target_v[t, k][0] = r * ti.cos(angle) * 0.05
            target_v[t, k][2] = r * ti.sin(angle) * 0.05
            target_h[t, k] = 0.

@debug
def visualizer(steps, prefix):
    interval = output_vis_interval
    os.makedirs('video/{}/'.format(prefix), exist_ok=True)

    for t in range(1, steps):
        if (t + 1) % interval == 0:
            gui.clear()
            gui.line((0, ground_height), (1, ground_height),
                    color=0x000022,
                    radius=3)
            gui.line((0, target_h[t]), (1, target_h[t]), color = 0x002200)
            solver.draw_robot(gui, t, target_v)
            gui.show('video/{}/{:04d}.png'.format(prefix, t))

def output_mesh(steps, x_, fn):
    os.makedirs('video/' + fn + '_objs', exist_ok=True)
    for t in range(1, steps):
        f = open('video/' + fn + f'_objs/{t:06d}.obj', 'w')
        for i in range(n_objects):
            f.write('v %.6f %.6f %.6f\n' % (x_[t, 0, i, 0], x_[t, 0, i, 1], x_[t, 0, i, 2]))
        for [p0, p1, p2] in faces:
            f.write('f %d %d %d\n' % (p0 + 1, p1 + 1, p2 + 1))
        f.close()

@debug
def simulate(steps, output_v=None, output_h=None, train = True, iter = 0, max_speed = 0.08, max_height = 0.1, *args, **kwargs):
    # clean up cache and set up control sequence
    solver.clear_states(steps)
    nn.clear()
    if train:
        initialize_train(iter, steps, max_speed, max_height)
    elif not train and dim == 2:
        initialize_validate(steps, output_v, output_h)
    elif not train and dim == 3:
        initialize_script(steps, 0.04, 0, 0, 0.04, -0.04, 0, 0, -0.04)
    loss[None] = 0.
    for l in losses:
        l[None] = 0.
    # start simulation
    if train:
        with ti.Tape(loss):
            for t in range(steps + 1):
                nn_input(t, 0, max_speed, max_height)
                nn.forward(t)
                solver.advance(t)
            get_loss(steps, *args, **kwargs)
    else:
        for t in range(steps + 1):
            nn_input(t, 0, max_speed, max_height)
            nn.forward(t)
            solver.advance(t)
        visualizer(steps, prefix = str(output_v) + "_" + str(output_h))
        if dim == 3:
            x_ = x.to_numpy()
            t = threading.Thread(target=output_mesh,args=(steps, x_, str(output_v) + '_' + str(output_h)))
            t.start()

@ti.kernel
def reset_robot(start: ti.template(), step: ti.template(), times: ti.template()):
    for k, i in ti.ndrange(times, n_objects):
        x[0, k * step + start, i] = initial_objects[i]

@ti.kernel
def get_center():
    for I in ti.grouped(initial_objects):
        initial_center[None] += initial_objects[I] / n_objects

def setup_robot():
    print('n_objects=', n_objects, '   n_springs=', n_springs)
    initial_objects.from_numpy(np.array(objects))
    for i in range(n_objects):
        initial_objects[i][0] += 0.4
    get_center()
    reset_robot(0, 1, batch_size)
    solver.initialize_robot()

@ti.kernel
def copy_robot(steps: ti.i32):
    for k, i in ti.ndrange(batch_size, n_objects):
        x[0, k, i] = x[steps, k, i]
        v[0, k, i] = v[steps, k, i]

def rounded_train(steps, iter, reset_step):
    copy_robot(steps)
    start = iter % reset_step
    step = reset_step
    times = (batch_size + step - start - 1) // step
    reset_robot(start, step, times)

def optimize(iters = 100000, change_iter = 5000, prefix = None, root_dir = "./",\
             load_path = None, *args, **kwargs):
    log_dir = os.path.join(root_dir, "logs")
    plot_dir = os.path.join(root_dir, "plots")
    weights_dir = os.path.join(root_dir, "weights")

    if prefix is not None:
        weights_dir = os.path.join(weights_dir, prefix)

    os.makedirs(plot_dir, exist_ok = True)
    os.makedirs(weights_dir, exist_ok = True)
    os.makedirs(log_dir, exist_ok = True)

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

    setup_robot()

    if load_path is not None and os.path.exists(load_path):
        print("load from {}".format(load_path))
        nn.load_weights(load_path)
    else:
        nn.weights_init()

    nn.clear_adam()

    losses = []
    best = 1e+15
    best_finetune = 1e+15
    train_steps = 1000
    if dim == 3 and sys.argv[0] == "validate.py":
        train_steps = 4000

    reset_step = 2

    for iter in range(iters):
        if iter > change_iter:
            if iter % 500 == 0 and reset_step < max_reset_step:
                reset_step += 1
            rounded_train(train_steps, iter, reset_step = reset_step)

        print("-------------------- {}iter #{} --------------------"\
            .format(""if prefix is None else "{}, ".format(prefix), iter))

        simulate(train_steps, iter = iter, *args, **kwargs)

        if iter <= change_iter and loss[None] < best:
            best = loss[None]
            nn.dump_weights(weight_out("best.pkl"))

        if iter > change_iter + max_reset_step and loss[None] < best_finetune:
            best_finetune = loss[None]
            nn.dump_weights(weight_out("best_finetune.pkl"))

        nn.dump_weights(weight_out("last.pkl"))
        nn.dump_weights(os.path.join(root_dir, "weight.pkl"))

        if iter % 50 == 0:
            nn.dump_weights(weight_out("iter{}.pkl".format(iter)))

        total_norm_sqr = nn.get_TNS()

        def print_logs(file = None):
            if iter > change_iter:
                print('Iter=', iter, 'Loss=', loss[None], 'Best_FT=', best_finetune, file = file)
            else:
                print('Iter=', iter, 'Loss=', loss[None], 'Best=', best, file = file)
            print("TNS= ", total_norm_sqr, file = file)
            for name, l in loss_dict.items():
                print("{}={}".format(name, l[None]), file = file)

        print_logs()
        log_file = open(log_path, "a")
        print_logs(log_file)
        log_file.close()

        nn.gradient_update(iter)
        losses.append(loss[None])

        if iter % 100 == 0 or iter % 10 == 0 and iter < 500:
            plot_curve(losses, plot_path)
            plot_curve(losses[-200:], plot200_path)

    return losses

if __name__ == '__main__':
    root_dir = "robot_{}".format(robot_id)
    load_path = os.path.join(root_dir, "weight")
    if dim == 3:
        loss_enable = ["rotation", "velocity"]
        optimize(root_dir = root_dir, loss_enable = loss_enable)
    else:
        if os.path.exists(root_dir):
            print()
            s = load_string("{} exists, continue?(Y/N)".format(root_dir), ["Y", "N"])
            if s == "N":
                exit(0)
            os.system('rm "{}" -r'.format(root_dir))
        optimize(500, 250, "stage1", root_dir, loss_enable = {"height", "pose"}, max_height = 0.01)
        #optimize(2000, 1000, "stage2", root_dir, load_path = load_path, loss_enable = {"height", "pose"}, max_height = 0.05)
        optimize(2000, 1000, "stage2", root_dir, load_path = load_path, loss_enable = {"height", "pose"})
        #optimize(2000, 1000, "stage4", root_dir, load_path = load_path, loss_enable = {"velocity", "actuation"}, max_speed = 0.08)
        optimize(100000, 5000, "final", root_dir, load_path = load_path, loss_enable = {"velocity", "height", "actuation"})
