from robot_config import robots
from robot3d_config import robots3d
from robot_mpm import robots_mpm
import threading
import utils

import random
import sys
import matplotlib.pyplot as plt
import taichi as ti
import math
import numpy as np
import os

import pickle as pkl

debug = utils.Debug(False)

real = ti.f64
ti.init(arch=ti.gpu, default_fp=real)

robot_id = 5
if len(sys.argv) == 2:
    robot_id = int(sys.argv[1])
    print("Run robot", robot_id)
simulator = ""
if robot_id >= 10000:
    simulator = "mpm"
    dim = 2
    objects, springs, n_springs = robots_mpm[robot_id - 10000]()
    n_objects = len(objects)
else:
    simulator = "mass_spring"
    if robot_id < 100:
        dim = 2
        objects, springs = robots[robot_id]()
    else:
        dim = 3
        objects, springs, faces = robots3d[robot_id - 100]()
    n_objects = len(objects)
    n_springs = len(springs)

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

max_steps = 4005
vis_interval = 256
output_vis_interval = 8
train_steps = 1000
validate_steps = 4000
output_target = []
output_sim = []
output_loss = []

loss = scalar()
loss_velocity = scalar()
loss_height = scalar()
loss_pose = scalar()
loss_weight = scalar()
loss_act = scalar()
loss_dict = {'loss_v': loss_velocity,
             'loss_h': loss_height,
             'loss_p': loss_pose,
             'loss_w': loss_weight,
             'loss_a': loss_act}
losses = loss_dict.values()

total_norm_sqr = scalar()

x = vec()
v = vec()
v_inc = vec()

head_id = 10

# target_ball = 0
elasticity = 0.0
ground_height = 0.1
gravity = -1.8
friction = 2.5

spring_anchor_a = ti.field(ti.i32)
spring_anchor_b = ti.field(ti.i32)
spring_length = scalar()
spring_stiffness = scalar()
spring_actuation = scalar()

initial_objects = vec()
initial_center = vec()

input_state = scalar()

n_sin_waves = 10
weights1 = scalar()
bias1 = scalar()

n_hidden = 64
weights2 = scalar()
bias2 = scalar()
hidden_act = scalar()
hidden = scalar()

m_weights1, v_weights1 = scalar(), scalar()
m_bias1, v_bias1 = scalar(), scalar()
m_weights2, v_weights2 = scalar(), scalar()
m_bias2, v_bias2 = scalar(), scalar()

center = vec()
height = scalar()
duplicate_v = 30
duplicate_h = 30
target_v = vec()
target_h = scalar()
weight_v = 1.
weight_h = 1.

act = scalar()
act_act = scalar()

dt = 0.004 if simulator == "mass_spring" else 0.002

run_period = 100
jump_period = 500
turn_period = 500
spring_omega = 2 * math.pi / dt / run_period
print(spring_omega)
drag_damping = 0
dashpot_damping = 0.2 if dim == 2 else 0.1

batch_size = 64

reset_step = 16

#weight_decay = 0.001
learning_rate = 3e-4

adam_a = learning_rate
adam_b1=0.9
adam_b2=0.9

def get_input_states():
    return n_sin_waves + dim * 2 * n_objects + duplicate_v * (dim - 1) + duplicate_h

ti.root.dense(ti.ijk, (max_steps, batch_size, n_objects)).place(x, v, v_inc)
ti.root.dense(ti.i, n_springs).place(spring_anchor_a, spring_anchor_b,
                                     spring_length, spring_stiffness,
                                     spring_actuation)

n_particles = n_objects
n_input_states = get_input_states()
n_grid = 64
dx = 1 / n_grid
inv_dx = 1 / dx
p_vol = 1
E, mu, la = 10, 10, 10
act_strength = 4

actuator_id = ti.field(ti.i32)
particle_type = ti.field(ti.i32)
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()
C, F = mat(), mat()

ti.root.dense(ti.ij, (batch_size, n_particles)).place(actuator_id, particle_type)
ti.root.dense(ti.ijk, (max_steps, batch_size, n_particles)).place(C, F)
ti.root.dense(ti.ijk, (batch_size, n_grid, n_grid)).place(grid_v_in, grid_m_in, grid_v_out)

ti.root.dense(ti.ij, (n_hidden, n_input_states)).place(weights1)
ti.root.dense(ti.i, n_hidden).place(bias1)
ti.root.dense(ti.ij, (n_springs, n_hidden)).place(weights2)
ti.root.dense(ti.i, n_springs).place(bias2)

ti.root.dense(ti.ij, (n_hidden, n_input_states)).place(m_weights1, v_weights1)
ti.root.dense(ti.i, n_hidden).place(m_bias1, v_bias1)
ti.root.dense(ti.ij, (n_springs, n_hidden)).place(m_weights2, v_weights2)
ti.root.dense(ti.i, n_springs).place(m_bias2, v_bias2)

ti.root.dense(ti.ijk, (max_steps, batch_size, n_input_states)).place(input_state)

ti.root.dense(ti.i, n_objects).place(initial_objects)
ti.root.place(initial_center)

ti.root.dense(ti.ijk, (max_steps, batch_size, n_hidden)).place(hidden_act, hidden)
ti.root.dense(ti.ijk, (max_steps, batch_size, n_springs)).place(act_act, act)
ti.root.dense(ti.ij, (max_steps, batch_size)).place(center, target_v, target_h, height)
ti.root.place(loss, total_norm_sqr)
ti.root.place(*losses)
ti.root.lazy_grad()

pool = ti.field(ti.f32, shape = (5 * batch_size))

weights = [weights1, weights2, bias1, bias2]

def dump_weights(name = "save.pkl"):
    #print("# Save to {}".format(name))
    w_val = []
    for w in weights:
        w_val.append(w.to_numpy())
    pkl.dump(w_val, open(name, "wb"))
    #print("# Done!")

def load_weights(name = "save.pkl"):
    #print('# Load from {}'.format(name))
    w_val = pkl.load(open(name, 'rb'))
    for w, val in zip(weights, w_val):
        w.from_numpy(val)
    #print("# Done!")

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
def nn_input(t: ti.i32):
    for k, j in ti.ndrange(batch_size, n_sin_waves):
        input_state[t, k, j] = ti.sin(spring_omega * t * dt + 2 * math.pi / n_sin_waves * j)

    for k, j in ti.ndrange(batch_size, n_objects):
        offset = x[t, k, j] - center[t, k]
        for d in ti.static(range(dim)):
            input_state[t, k, j * dim * 2 + n_sin_waves + d] = offset[d] / 0.05
            input_state[t, k, j * dim * 2 + n_sin_waves + dim + d] = v[t, k, j][d]

    if ti.static(duplicate_v > 0):
        if ti.static(dim == 2):
            for k, j in ti.ndrange(batch_size, duplicate_v):
                input_state[t, k, n_objects * dim * 2 + n_sin_waves + j * (dim - 1)] = target_v[t, k][0] / 0.08
        else:
            for k, j in ti.ndrange(batch_size, duplicate_v):
                input_state[t, k, n_objects * dim * 2 + n_sin_waves + j * (dim - 1)] = target_v[t, k][0] / 0.08
                input_state[t, k, n_objects * dim * 2 + n_sin_waves + j * (dim - 1) + 1] = target_v[t, k][2] / 0.08
    if ti.static(duplicate_h > 0):
        for k, j in ti.ndrange(batch_size, duplicate_h):
            input_state[t, k, n_objects * dim * 2 + n_sin_waves + duplicate_v * (dim - 1) + j] = (target_h[t, k] - 0.15) / 0.05 - 1

@ti.kernel
def nn1(t: ti.i32):
    for k, i, j in ti.ndrange(batch_size, n_hidden, n_input_states):
        hidden[t, k, i] += weights1[i, j] * input_state[t, k, j]

    for k, i in ti.ndrange(batch_size, n_hidden):
        hidden_act[t, k, i] = ti.sin(hidden[t, k, i] + bias1[i])


@ti.kernel
def nn2(t: ti.i32):
    for k, i, j in ti.ndrange(batch_size, n_springs, n_hidden):
        act[t, k, i] += weights2[i, j] * hidden_act[t, k, j]
    for k, i in ti.ndrange(batch_size, n_springs):
        act_act[t, k, i] = ti.sin(act[t, k, i] + bias2[i])


@ti.kernel
def apply_spring_force(t: ti.i32):
    for k, i in ti.ndrange(batch_size, n_springs):
        a = spring_anchor_a[i]
        b = spring_anchor_b[i]
        pos_a = x[t, k, a]
        pos_b = x[t, k, b]
        dist = pos_a - pos_b
        length = dist.norm(1e-8) + 1e-4

        target_length = spring_length[i] * (1.0 + spring_actuation[i] * act_act[t, k, i])
        impulse = dt * (length - target_length) * spring_stiffness[i] / length * dist

        # Dashpot damping
        x_ij = x[t, k, a] - x[t, k, b]
        d = x_ij.normalized()
        v_rel = (v[t, k, a] - v[t, k, b]).dot(d)
        impulse += dashpot_damping * v_rel * d

        ti.atomic_add(v_inc[t + 1, k, a], -impulse)
        ti.atomic_add(v_inc[t + 1, k, b], impulse)


@ti.kernel
def advance_toi(t: ti.i32):
    for k, i in ti.ndrange(batch_size, n_objects):
        s = math.exp(-dt * drag_damping)
        unitY = ti.Matrix.zero(real, dim, 1)
        unitY[1] = 1.0
        old_v = s * v[t - 1, k, i] + dt * gravity * unitY + v_inc[t, k, i]
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

        act_applied = act_act[f, k, ti.max(0, act_id)] * act_strength
        if act_id == -1:
            act_applied = 0.0
        # ti.print(act_act)

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


bound = 3
coeff = 0.5


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
def compute_loss_velocity():
    for t, k in ti.ndrange((run_period, train_steps + 1), batch_size):
        if t % turn_period > run_period and target_h[t - run_period, k] < 0.1 + 1e-4:
            if ti.static(dim == 2):
                loss_velocity[None] += (center[t, k](0) - center[t - run_period, k](0) - target_v[t - run_period, k](0))**2 / batch_size
            else:
                loss_velocity[None] += (center[t, k](0) - center[t - run_period, k](0) - target_v[t - run_period, k](0))**2 / batch_size
                loss_velocity[None] += (center[t, k](2) - center[t - run_period, k](2) - target_v[t - run_period, k](2))**2 / batch_size
    # if k == 0:
    #     print("Mark run: ", center[t, 0](0) - center[t - run_period, 0](0), target_v[t - run_period, 0](0))


@ti.kernel
def compute_loss_height():
    for t, k in ti.ndrange((1, train_steps + 1), batch_size):
        if t % jump_period == jump_period - 1 and target_h[t, k] > 0.1:
            loss_height[None] += (height[t, k] - target_h[t, k]) ** 2 / batch_size / (train_steps // jump_period) * 100


@ti.kernel
def compute_loss_pose():
    # TODO: This doesn't work for 3D
    for t, k, i in ti.ndrange((1, train_steps + 1), batch_size, n_objects):
        if t % jump_period == 0:
            #dist2 = sum((x[t, k, i] - center[t, k] - initial_objects[i] + initial_center[None]) ** 2)
            dist2 = sum((x[t, k, i] - initial_objects[i]) ** 2)
            loss_pose[None] += dist2 / batch_size / (train_steps // jump_period)

@ti.kernel
def compute_loss_actuation():
    for t, k, i in ti.ndrange(train_steps, batch_size, n_springs):
        if target_h[t, k] < 0.1 + 1e-4:
            loss_act[None] += ti.max(ti.abs(act_act[t, k, i]) - (ti.abs(target_v[t, k][0]) / 0.08) ** 0.5, 0.) / n_springs / batch_size / train_steps * 10
'''
@ti.kernel
def compute_loss_crouch():
    for t, k, i in ti.ndrange(train_steps, batch_size, n_springs):

'''

@ti.kernel
def compute_loss_final(l: ti.template()):
    loss[None] += l[None]


@ti.kernel
def compute_weight_decay():
    for I in ti.grouped(weights1):
        loss[None] += weight_decay * weights1[I] ** 2
    for I in ti.grouped(weights2):
        loss[None] += weight_decay * weights2[I] ** 2


gui = ti.GUI(show_gui=False, background_color=0xFFFFFF)

@ti.kernel
def initialize_validate(output_v: ti.f32, output_h: ti.f32):
    for t, k in ti.ndrange(validate_steps, batch_size):
        q = t // turn_period
        if q % 3 == 0:
            if ti.static(dim == 2):
                target_v[t, k][0] = (q / 3 % 2 * 2 - 1) * output_v
            else:
                target_v[t, k][0] = (q / 3 % 2 * 2 - 1) * output_v
                target_v[t, k][2] = (q / 3 % 2 * 2 - 1) * output_v
            target_h[t, k] = 0.1
        elif q % 3 == 1:
            target_v[t, k][0] = 0
            target_h[t, k] = output_h
        else:
            target_v[t, k][0] = 0
            target_h[t, k] = 0.1

@ti.kernel
def initialize_train():
    times = ti.static(train_steps // turn_period)
    for _ in range(batch_size * times * 3):
        pool[_] = ti.random()
    for t, k in ti.ndrange(train_steps, batch_size):
        q = (t // turn_period * batch_size + k) * 3
        if pool[q + 0] < 0.5:
            if ti.static(dim == 2):
                target_v[t, k][0] = (pool[q + 1] * 2 - 1) * 0.08
            else:
                target_v[t, k][0] = (pool[q + 1] * 2 - 1) * 0.08
                target_v[t, k][2] = (pool[q + 2] * 2 - 1) * 0.08
            target_h[t, k] = 0.1
        else:
            target_h[t, k] = pool[q + 1] * 0.1 + 0.1
            target_v[t, k] *= 0.


@ti.kernel
def clear_states():
    for t, k, i in ti.ndrange(max_steps, batch_size, n_objects):
        x.grad[t, k, i] = ti.Matrix.zero(real, dim, 1)
        v.grad[t, k, i] = ti.Matrix.zero(real, dim, 1)
        v_inc[t, k, i] = ti.Matrix.zero(real, dim, 1)
        v_inc.grad[t, k, i] = ti.Matrix.zero(real, dim, 1)
    for t, k, i in ti.ndrange(max_steps, batch_size, n_particles):
        C[t, k, i] = ti.Matrix.zero(real, dim, dim)
        C.grad[t, k, i] = ti.Matrix.zero(real, dim, dim)
        if ti.static(dim == 2):
            F[t, k, i] = [[1., 0.], [0., 1.]]
        else:
            F[t, k, i] = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        F.grad[t, k, i] = ti.Matrix.zero(real, dim, dim)
    for I in ti.grouped(hidden):
        hidden[I] = 0.
    for I in ti.grouped(act):
        act[I] = 0.


def clear():
    clear_states()
    m_weights1.fill(0)
    v_weights1.fill(0)
    m_bias1.fill(0)
    v_bias1.fill(0)
    m_weights2.fill(0)
    v_weights2.fill(0)
    m_bias2.fill(0)
    v_bias2.fill(0)

@debug
def init(train, output_v = None, output_h = None):
    clear_states()

    if train:
        initialize_train()
    else:
        initialize_validate(output_v, output_h)

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
def forward(train = True):
    total_steps = train_steps if train else validate_steps
    for t in range(total_steps):
        compute_center(t)
        compute_height(t)
        nn_input(t)
        nn1(t)
        nn2(t)
        if simulator == "mpm":
            advance_mpm(t)
        else:
            apply_spring_force(t)
            advance_toi(t + 1)

    compute_center(total_steps)
    compute_height(total_steps)

def get_loss():
    #for t in range(train_steps):

    if duplicate_v > 0:
        compute_loss_velocity()

    if duplicate_h > 0:
        compute_loss_height()
        #compute_loss_pose()
    compute_loss_actuation()

    for l in losses:
        compute_loss_final(l)

@debug
def visualizer(train, prefix, visualize = True):
    total_steps = train_steps if train else validate_steps

    interval = vis_interval
    if not train:
        interval = output_vis_interval
        os.makedirs('mass_spring/{}/'.format(prefix), exist_ok=True)

        if visualize:
            for t in range(1, total_steps):
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

                            a = act_act[t - 1, 0, i] * 0.5
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
                            act_applied = act_act[t - 1, 0, aid[0, i]]
                            color = (0.5 - act_applied, 0.5 - abs(act_applied), 0.5 + act_applied)
                        circle(x[t, 0, i][0], x[t, 0, i][1], color)

                    if target_v[t, 0][0] > 0:
                        circle(0.5, 0.5, (1, 0, 0))
                        circle(0.6, 0.5, (1, 0, 0))
                    else:
                        circle(0.5, 0.5, (0, 0, 1))
                        circle(0.4, 0.5, (0, 0, 1))

                    gui.show('mass_spring/{}/{:04d}.png'.format(prefix, t))

    if train:
        output_loss.append(loss[None])

        if visualize:
            utils.plot_curve(output_loss, "training_curve.png")
            utils.plot_curve(output_loss[-200:], "training_curve_last_200.png")

def output_mesh(x_, fn):
    os.makedirs(fn + '_objs', exist_ok=True)
    for t in range(1, validate_steps):
        f = open(fn + f'_objs/{t:06d}.obj', 'w')
        for i in range(n_objects):
            f.write('v %.6f %.6f %.6f\n' % (x_[t, 0, i, 0], x_[t, 0, i, 1], x_[t, 0, i, 2]))
        for [p0, p1, p2] in faces:
            f.write('f %d %d %d\n' % (p0 + 1, p1 + 1, p2 + 1))
        f.close()

@debug
def simulate(output_v=None, output_h=None, visualize=True):
    train = output_v is None and output_h is None
    prefix = None
    if not train:
        prefix = str(output_v) + "_" + str(output_h)
    init(train, output_v, output_h)
    if train:
        with ti.Tape(loss):
            forward()
            get_loss()
    else:
        forward(False)
        if dim == 3:
            x_ = x.to_numpy()
            t = threading.Thread(target=output_mesh,args=(x_, str(output_v) + '_' + str(output_h)))
            t.start()

    visualizer(train = train, prefix = prefix, visualize = visualize)

def validate():
    '''
    simulate(0.08, 0.1)
    simulate(0.06, 0.1)
    simulate(0.04, 0.1)
    simulate(0.02, 0.1)
    simulate(0., 0.1)

    simulate(0, 0.15)
    simulate(0, 0.175)
    simulate(0, 0.20)
    '''
    simulate(0.06, 0.15)
    simulate(0.04, 0.15)
    simulate(0.02, 0.15)
    # simulate(0, 0.25)
    # simulate(0, 0.3)
    # simulate(0, 0)

simulate.cnt = 0

@ti.kernel
def copy_robot():
    for k, i in ti.ndrange(batch_size, n_objects):
        x[0, k, i] = x[train_steps, k, i]
        v[0, k, i] = v[train_steps, k, i]

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

def rounded_train(iter):
    copy_robot()
    start = iter % reset_step
    step = reset_step
    times = (batch_size + step - start) // step
    reset_robot(start, step, times)

@ti.kernel
def gradient_update(w: ti.template(), m: ti.template(), v: ti.template(), iter: ti.i32):
    '''
    for I in ti.grouped(w):
        w[I] -= w.grad[I] * learning_rate
    '''
    for I in ti.grouped(w):
        m[I] = adam_b1 * m[I] + (1 - adam_b1) * w.grad[I]
        v[I] = adam_b2 * v[I] + (1 - adam_b2) * w.grad[I] * w.grad[I]
        m_cap = m[I] / (1 - adam_b1 ** (iter + 1))
        v_cap = v[I] / (1 - adam_b2 ** (iter + 1))
        w[I] -= (adam_a * m_cap) / (ti.sqrt(v_cap) + 1e-8)

@ti.kernel
def compute_TNS(w: ti.template()):
    for I in ti.grouped(w):
        total_norm_sqr[None] += w.grad[I] ** 2

def optimize(output_log = "plots/training.log"):
    os.makedirs("plots", exist_ok = True)
    log_file = open(output_log, 'w')
    log_file.close()
    '''
    for i in range(n_hidden):
        for j in range(n_input_states):
            weights1[i, j] = np.random.randn() * math.sqrt(
                2 / (n_hidden + n_input_states)) * 2

    for i in range(n_springs):
        for j in range(n_hidden):
            # TODO: n_springs should be n_actuators
            weights2[i, j] = np.random.randn() * math.sqrt(
                2 / (n_hidden + n_springs)) * 2
    '''
    q1 = math.sqrt(6 / n_input_states)
    for i in range(n_hidden):
        for j in range(n_input_states):
            weights1[i, j] = (np.random.rand() * 2 - 1) * q1

    q2 = math.sqrt(6 / n_hidden)
    for i in range(n_springs):
        for j in range(n_hidden):
            weights2[i, j] = (np.random.rand() * 2 - 1) * q2

    losses = []
    # simulate('initial{}'.format(robot_id), visualize=visualize)
    best = 1e+15
    best_finetune = 1e+15

    os.makedirs("weights", exist_ok=True)

    for iter in range(10000):
        #if iter > 5000:
        #    rounded_train(iter)
            
        print("-------------------- iter #{} --------------------".format(iter))

        simulate(visualize=iter % 10 == 0)

        if iter <= 5000 and loss[None] < best:
            best = loss[None]
            dump_weights("weights/best.pkl")
        
        if iter > 5016 and loss[None] < best_finetune:
            best_finetune = loss[None]
            dump_weights("weights/best_finetune.pkl")

        dump_weights("weights/last.pkl")

        if iter % 50 == 0:
            dump_weights("weights/iter{}.pkl".format(iter))

        total_norm_sqr[None] = 0.
        compute_TNS(weights1)
        compute_TNS(bias1)
        compute_TNS(weights2)
        compute_TNS(bias2)

        def print_logs(file = None):
            if iter > 5000:
                print('Iter=', iter, 'Loss=', loss[None], 'Best_FT=', best_finetune, file = file)
            else:
                print('Iter=', iter, 'Loss=', loss[None], 'Best=', best, file = file)
            print("TNS= ", total_norm_sqr[None], file = file)
            for name, l in loss_dict.items():
                print("{}={}".format(name, l[None]), file = file)

        print_logs()
        log_file = open(output_log, "a")
        print_logs(log_file)
        log_file.close()

        gradient_update(weights1, m_weights1, v_weights1, iter)
        gradient_update(bias1, m_bias1, v_bias1, iter)
        gradient_update(weights2, m_weights2, v_weights2, iter)
        gradient_update(bias2, m_bias2, v_bias2, iter)
        losses.append(loss[None])

        # print(time.time() - t, ' 2')

        #if (iter + 1) % 200 == 0:
        #    validate()

    return losses


if __name__ == '__main__':
    setup_robot()
    optimize()
