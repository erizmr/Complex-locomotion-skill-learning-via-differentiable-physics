import threading
from taichi.lang.impl import reset

from utils import *
from config import *
from nn import Model

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

x = vec()
v = vec()
v_inc = vec()

spring_anchor_a = ti.field(ti.i32)
spring_anchor_b = ti.field(ti.i32)
spring_length = scalar()
spring_stiffness = scalar()
spring_actuation = scalar()

initial_objects = vec()
initial_center = vec()

input_state = scalar()

center = vec()
height = scalar()
rotation = scalar()
head_center = vec()
head_counter = scalar()
tail_center = vec()
tail_counter = scalar()
target_v = vec()
target_h = scalar()

actuation = scalar()

ti.root.dense(ti.ijk, (max_steps, batch_size, n_objects)).place(x, v, v_inc)
ti.root.dense(ti.i, n_springs).place(spring_anchor_a, spring_anchor_b,
                                     spring_length, spring_stiffness,
                                     spring_actuation)


actuator_id = ti.field(ti.i32)
particle_type = ti.field(ti.i32)
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()
C, F = mat(), mat()

ti.root.dense(ti.ij, (batch_size, n_particles)).place(actuator_id, particle_type)
ti.root.dense(ti.ijk, (max_steps, batch_size, n_particles)).place(C, F)
ti.root.dense(ti.ijk, (batch_size, n_grid, n_grid)).place(grid_v_in, grid_m_in, grid_v_out)

ti.root.dense(ti.ijk, (max_steps, batch_size, n_input_states)).place(input_state)

ti.root.dense(ti.i, n_objects).place(initial_objects)
ti.root.place(initial_center)

ti.root.dense(ti.ijk, (max_steps, batch_size, n_springs)).place(actuation)
ti.root.dense(ti.ij, (max_steps, batch_size)).place(center, target_v, target_h, height, rotation, head_center, head_counter, tail_center, tail_counter)
ti.root.place(loss)
ti.root.place(*losses)

nn = Model(max_steps, batch_size, n_input_states, n_springs, input_state, actuation, n_hidden)

ti.root.lazy_grad()

pool = ti.field(ti.f64, shape = (5 * batch_size))

@ti.kernel
def compute_center(t: ti.i32):
    n = ti.static(n_objects)
    for k in range(batch_size):
        center[t, k] = ti.Matrix.zero(real, dim, 1)
    for k, i in ti.ndrange(batch_size, n):
            center[t, k] += x[t, k, i] / n

@ti.kernel
def compute_height(t: ti.i32):
    for k in range(batch_size):
        h = 10.
        for i in ti.static(range(n_objects)):
            h = ti.min(h, x[t, k, i](1))
        if t % jump_period == 0:
            height[t, k] = h
        else:
            height[t, k] = ti.max(height[t - 1, k], h)

@ti.kernel
def compute_rotation(t: ti.i32):
    for k in range(batch_size):
        for i in ti.static(range(n_objects)):
            if x[0, k, i](0) < center[0, k](0):
                head_center[t, k] += x[t, k, i]
                head_counter[t, k] += 1.
            else:
                tail_center[t, k] += x[t, k, i]
                tail_counter[t, k] += 1.
        direction = -head_center[t, k] * tail_counter[t, k] + tail_center[t, k] * head_counter[t, k]
        rotation[t, k] = ti.atan2(direction[1], direction[0])

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
                input_state[t, k, j * dim * 2 + n_sin_waves + d] = [vec_x] * float(sys.argv[2])
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
def apply_spring_force(t: ti.i32):
    for k, i in ti.ndrange(batch_size, n_springs):
        a = spring_anchor_a[i]
        b = spring_anchor_b[i]
        pos_a = x[t, k, a]
        pos_b = x[t, k, b]
        dist = pos_a - pos_b
        length = dist.norm(1e-8) + 1e-4

        target_length = spring_length[i] * (1.0 + spring_actuation[i] * actuation[t, k, i])
        impulse = dt * (length - target_length) * spring_stiffness[i] / length * dist

        # Dashpot damping
        x_ij = x[t, k, a] - x[t, k, b]
        d = x_ij.normalized()
        v_rel = (v[t, k, a] - v[t, k, b]).dot(d)
        impulse += dashpot_damping * v_rel * d

        ti.atomic_add(v_inc[t, k, a], -impulse)
        ti.atomic_add(v_inc[t, k, b], impulse)


@ti.kernel
def advance_toi(t: ti.i32):
    for k, i in ti.ndrange(batch_size, n_objects):
        s = math.exp(-dt * drag_damping)
        unitY = ti.Matrix.zero(real, dim, 1)
        unitY[1] = 1.0
        old_v = s * v[t - 1, k, i] + dt * gravity * unitY + v_inc[t - 1, k, i]
        old_x = x[t - 1, k, i]
        new_x = old_x + dt * old_v
        toi = 0.0
        new_v = old_v
        if new_x[1] < ground_height and old_v[1] < -1e-4:
            toi = -(old_x[1] - ground_height) / old_v[1]
            new_v = ti.Matrix.zero(real, dim, 1)
        new_x = old_x + toi * old_v + (dt - toi) * new_v

        v[t, k, i] = new_v
        x[t, k, i] = new_x


@ti.kernel
def clear_grid():
    for k, i, j in ti.ndrange(batch_size, n_grid, n_grid):
        grid_v_in[k, i, j] = ti.Matrix.zero(real, dim, 1)
        grid_m_in[k, i, j] = 0
        grid_v_out[k, i, j] = ti.Matrix.zero(real, dim, 1)
        grid_v_in.grad[k, i, j] = ti.Matrix.zero(real, dim, 1)
        grid_m_in.grad[k, i, j] = 0
        grid_v_out.grad[k, i, j] = ti.Matrix.zero(real, dim, 1)


@ti.kernel
def p2g(f: ti.i32):
    for k, p in ti.ndrange(batch_size, n_particles):
        base = ti.cast(x[f, k, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, k, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, k, p]) @ F[f, k, p]
        J = (new_F).determinant()
        if particle_type[k, p] == 0:  # fluid
            sqrtJ = ti.sqrt(J)
            new_F = ti.Matrix([[sqrtJ, 0], [0, sqrtJ]])

        F[f + 1, k, p] = new_F
        r, s = ti.polar_decompose(new_F)

        act_id = actuator_id[k, p]

        act_applied = actuation[f, k, ti.max(0, act_id)] * act_strength
        if act_id == -1:
            act_applied = 0.0
        # ti.print(actuation)

        A = ti.Matrix([[0.0, 0.0], [0.0, 1.0]]) * act_applied
        cauchy = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        mass = 0.0
        if particle_type[k, p] == 0:
            mass = 4
            cauchy = ti.Matrix([[1.0, 0.0], [0.0, 0.1]]) * (J - 1) * E
        else:
            mass = 1
            cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + \
                     ti.Matrix.diag(2, la * (J - 1) * J)
        cauchy += new_F @ A @ new_F.transpose()
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + mass * C[f, k, p]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
                weight = w[i](0) * w[j](1)
                grid_v_in[k, base + offset] += weight * (mass * v[f, k, p] + affine @ dpos)
                grid_m_in[k, base + offset] += weight * mass


@ti.kernel
def grid_op():
    for k, i, j in grid_m_in:
        inv_m = 1 / (grid_m_in[k, i, j] + 1e-10)
        v_out = inv_m * grid_v_in[k, i, j]
        v_out[1] += dt * gravity
        if i < bound:
            v_out[0] = 0
            v_out[1] = 0
        if i > n_grid - bound:
            v_out[0] = 0
            v_out[1] = 0
        if j < bound:
            v_out[0] = 0
            v_out[1] = 0
        if j > n_grid - bound:
            v_out[0] = 0
            v_out[1] = 0
        grid_v_out[k, i, j] = v_out


@ti.kernel
def g2p(f: ti.i32):
    for k, p in ti.ndrange(batch_size, n_particles):
        base = ti.cast(x[f, k, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, k, p] * inv_dx - ti.cast(base, real)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos = ti.cast(ti.Vector([i, j]), real) - fx
                g_v = grid_v_out[k, base(0) + i, base(1) + j]
                weight = w[i](0) * w[j](1)
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        v[f + 1, k, p] = new_v
        x[f + 1, k, p] = x[f, k, p] + dt * v[f + 1, k, p]
        C[f + 1, k, p] = new_C


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
    if duplicate_v > 0:
        if "velocity" in loss_enable:
            compute_loss_velocity(steps)

    if duplicate_h > 0:
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
        target_v[t, k][0] = 0
        target_h[t, k] = output_h
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
                if pool[q + 0] < 0.5:
                    target_v[t, k][0] = (pool[q + 1] * 2 - 1) * max_speed
                    target_h[t, k] = 0.1
                else:
                    target_h[t, k] = pool[q + 1] * max_height + 0.1
                    target_v[t, k] *= 0.
        else:
            # r = pool[q + 1]
            # angle = pool[q + 2] * 2 * 3.1415926
            r = 1.
            angle = 0.
            target_v[t, k][0] = r * ti.cos(angle) * 0.05
            target_v[t, k][2] = r * ti.sin(angle) * 0.05
            target_h[t, k] = 0.

@ti.kernel
def clear_states(steps: ti.template()):
    for t, k, i in ti.ndrange(steps, batch_size, n_objects):
        x.grad[t, k, i] = ti.Matrix.zero(real, dim, 1)
        v.grad[t, k, i] = ti.Matrix.zero(real, dim, 1)
        v_inc[t, k, i] = ti.Matrix.zero(real, dim, 1)
        v_inc.grad[t, k, i] = ti.Matrix.zero(real, dim, 1)
    for t, k, i in ti.ndrange(steps, batch_size, n_particles):
        C[t, k, i] = ti.Matrix.zero(real, dim, dim)
        C.grad[t, k, i] = ti.Matrix.zero(real, dim, dim)
        if ti.static(dim == 2):
            F[t, k, i] = [[1., 0.], [0., 1.]]
        else:
            F[t, k, i] = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        F.grad[t, k, i] = ti.Matrix.zero(real, dim, dim)
    for I in ti.grouped(head_center):
        head_center[I] = ti.Matrix.zero(real, dim, 1)
        head_counter[I] = 0.
        tail_center[I] = ti.Matrix.zero(real, dim, 1)
        tail_counter[I] = 0.
        rotation[I] = 0.

@debug
def init(steps, train, output_v = None, output_h = None, iter = 0, \
         max_speed = 0.08, max_height = 0.1, *args, **kwargs):
    clear_states(steps)
    nn.clear()

    if train:
        initialize_train(iter, steps, max_speed, max_height)
    else:
        initialize_validate(steps, output_v, output_h)

    loss[None] = 0.
    for l in losses:
        l[None] = 0.


@ti.complex_kernel
def advance_mpm(s):
    clear_grid()
    p2g(s)
    grid_op()
    g2p(s)


@ti.complex_kernel_grad(advance_mpm)
def advance_mpm_grad(s):
    clear_grid()
    p2g(s)
    grid_op()

    g2p.grad(s)
    grid_op.grad()
    p2g.grad(s)


@debug
def forward(steps, train = True, max_speed = 0.08, max_height = 0.1, *args, **kwargs):
    for t in range(steps):
        compute_center(t)
        compute_height(t)
        if dim == 3:
            compute_rotation(t)
        nn_input(t, 0, max_speed, max_height)
        nn.forward(t)
        if simulator == "mpm":
            advance_mpm(t)
        else:
            apply_spring_force(t)
            advance_toi(t + 1)

    compute_center(steps)
    compute_height(steps)

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

            def circle(x, y, color):
                if simulator == "mass_spring":
                    gui.circle((x, y), ti.rgb_to_hex(color), 7)
                else:
                    gui.circle((x, y + 0.1 - dx * bound), ti.rgb_to_hex(color), 2)
                
            if simulator == "mass_spring":
                for i in range(n_springs):

                    def get_pt(x):
                        return (x[0], x[1])

                    a = actuation[t - 1, 0, i] * 0.5
                    r = 2
                    if spring_actuation[i] == 0:
                        a = 0
                        c = 0x222222
                    else:
                        r = 4
                        c = ti.rgb_to_hex((0.5 + a, 0.5 - abs(a), 0.5 - a))
                    gui.line(get_pt(x[t, 0, spring_anchor_a[i]]),
                            get_pt(x[t, 0, spring_anchor_b[i]]),
                            color=c,
                            radius=r)

            aid = actuator_id.to_numpy()
            for i in range(n_objects):
                color = (0.06640625, 0.06640625, 0.06640625)
                if simulator == "mpm" and aid[0, i] != -1:
                    act_applied = actuation[t - 1, 0, aid[0, i]]
                    color = (0.5 - act_applied, 0.5 - abs(act_applied), 0.5 + act_applied)
                circle(x[t, 0, i][0], x[t, 0, i][1], color)

            if target_v[t, 0][0] > 0:
                circle(0.5, 0.5, (1, 0, 0))
                circle(0.6, 0.5, (1, 0, 0))
            else:
                circle(0.5, 0.5, (0, 0, 1))
                circle(0.4, 0.5, (0, 0, 1))

            gui.show('video/{}/{:04d}.png'.format(prefix, t))

def output_mesh(steps, x_, fn):
    os.makedirs(fn + '_objs', exist_ok=True)
    for t in range(1, steps):
        f = open(fn + f'_objs/{t:06d}.obj', 'w')
        for i in range(n_objects):
            f.write('v %.6f %.6f %.6f\n' % (x_[t, 0, i, 0], x_[t, 0, i, 1], x_[t, 0, i, 2]))
        for [p0, p1, p2] in faces:
            f.write('f %d %d %d\n' % (p0 + 1, p1 + 1, p2 + 1))
        f.close()

@debug
def simulate(steps, output_v=None, output_h=None, train = True, iter = 0, *args, **kwargs):
    prefix = None
    if not train:
        prefix = str(output_v) + "_" + str(output_h)
    init(steps, train, output_v, output_h, iter, *args, **kwargs)
    if train:
        with ti.Tape(loss):
            forward(steps, *args, **kwargs)
            get_loss(steps, *args, **kwargs)
    else:
        forward(steps, False, *args, **kwargs)
        if dim == 3:
            x_ = x.to_numpy()
            t = threading.Thread(target=output_mesh,args=(steps, x_, str(output_v) + '_' + str(output_h)))
            t.start()

        visualizer(steps, prefix = prefix)

@ti.kernel
def copy_robot(steps: ti.i32):
    for k, i in ti.ndrange(batch_size, n_objects):
        x[0, k, i] = x[steps, k, i]
        v[0, k, i] = v[steps, k, i]

@ti.kernel
def reset_robot(start: ti.template(), step: ti.template(), times: ti.template()):
    for k, i in ti.ndrange(times, n_objects):
        x[0, k * step + start, i] = initial_objects[i]

def setup_robot():
    print('n_objects=', n_objects, '   n_springs=', n_springs)

    initial_objects.from_numpy(np.array(objects))
    for i in range(n_objects):
        initial_objects[i][0] += 0.4

    @ti.kernel
    def get_center():
        for I in ti.grouped(initial_objects):
            initial_center[None] += initial_objects[I] / n_objects

    get_center()

    if simulator == "mpm":
        reset_robot(0, 1, n_objects)
        for k in range(batch_size):
            for i in range(n_objects):
                actuator_id[k, i] = springs[i]
        particle_type.fill(1)
    else:
        reset_robot(0, 1, batch_size)
        for i in range(n_springs):
            s = springs[i]
            spring_anchor_a[i] = s[0]
            spring_anchor_b[i] = s[1]
            spring_length[i] = s[2]
            spring_stiffness[i] = s[3] / 10
            spring_actuation[i] = s[4]

def rounded_train(steps, iter):
    copy_robot(steps)
    start = iter % reset_step
    step = reset_step
    times = (batch_size + step - start) // step
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

    for iter in range(iters):

        if iter > change_iter:
            rounded_train(train_steps, iter)

        print("-------------------- {}iter #{} --------------------"\
            .format(""if prefix is None else "{}, ".format(prefix), iter))

        simulate(train_steps, iter = iter, *args, **kwargs)

        if iter <= change_iter and loss[None] < best:
            best = loss[None]
            nn.dump_weights(weight_out("best.pkl"))
            nn.dump_weights(os.path.join(root_dir, "weight.pkl"))
        
        if iter > change_iter + reset_step and loss[None] < best_finetune:
            best_finetune = loss[None]
            nn.dump_weights(weight_out("best_finetune.pkl"))
            nn.dump_weights(os.path.join(root_dir, "weight.pkl"))

        nn.dump_weights(weight_out("last.pkl"))

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
        optimize(2000, 1000, "stage2", root_dir, load_path = load_path, loss_enable = {"height", "pose"}, max_height = 0.1)
        optimize(100000, 5000, "final", root_dir, load_path = load_path, loss_enable = {"velocity", "height", "actuation"}, max_height = 0.1)