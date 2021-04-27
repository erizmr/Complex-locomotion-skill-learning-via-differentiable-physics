from robot_config import robots
import utils

import random
import sys
import matplotlib.pyplot as plt
import taichi as ti
import math
import numpy as np
import os

debug = utils.Debug(True)

real = ti.f64
ti.init(arch=ti.gpu, default_fp=real)

max_steps = 4096
vis_interval = 256
output_vis_interval = 8
steps = 2048 // 2
train_steps = steps
validate_steps = steps * 2
output_target = []
output_sim = []
output_loss = []

assert steps * 2 <= max_steps

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(2, dtype=real)

loss = scalar()

x = vec()
v = vec()
v_inc = vec()

head_id = 10
goal = vec()

# target_ball = 0
elasticity = 0.0
ground_height = 0.1
gravity = -1.8
friction = 2.5

robot_id = int(sys.argv[1])
objects, springs = robots[robot_id]()
n_objects = len(objects)
n_springs = len(springs)

spring_anchor_a = ti.field(ti.i32)
spring_anchor_b = ti.field(ti.i32)
spring_length = scalar()
spring_stiffness = scalar()
spring_actuation = scalar()

n_sin_waves = 10
weights1 = scalar()
bias1 = scalar()

n_hidden = 64
weights2 = scalar()
bias2 = scalar()
hidden = scalar()

m_weights1, v_weights1 = scalar(), scalar()
m_bias1, v_bias1 = scalar(), scalar()
m_weights2, v_weights2 = scalar(), scalar()
m_bias2, v_bias2 = scalar(), scalar()

center = vec()
height = scalar()
duplicate_v = 15
duplicate_h = 0
target_v = vec()
target_h = scalar()
weight_v = 1.
weight_h = 1.

act = scalar()

dt = 0.004

run_period = 100
jump_period = 500
turn_period = 500
spring_omega = 2 * math.pi / dt / run_period
print(spring_omega)
drag_damping = 0
dashpot_damping = 0.2

batch_size = int(sys.argv[2])

#weight_decay = 0.001
learning_rate = 3e-4

adam_a = learning_rate
adam_b1=0.9
adam_b2=0.999

def n_input_states():
    return n_sin_waves + 4 * n_objects + 2 * duplicate_v + duplicate_h

ti.root.dense(ti.ijk, (max_steps, batch_size, n_objects)).place(x, v, v_inc)
ti.root.dense(ti.i, n_springs).place(spring_anchor_a, spring_anchor_b,
                                     spring_length, spring_stiffness,
                                     spring_actuation)
ti.root.dense(ti.ij, (n_hidden, n_input_states())).place(weights1)
ti.root.dense(ti.i, n_hidden).place(bias1)
ti.root.dense(ti.ij, (n_springs, n_hidden)).place(weights2)
ti.root.dense(ti.i, n_springs).place(bias2)

ti.root.dense(ti.ij, (n_hidden, n_input_states())).place(m_weights1, v_weights1)
ti.root.dense(ti.i, n_hidden).place(m_bias1, v_bias1)
ti.root.dense(ti.ij, (n_springs, n_hidden)).place(m_weights2, v_weights2)
ti.root.dense(ti.i, n_springs).place(m_bias2, v_bias2)

ti.root.dense(ti.ijk, (max_steps, batch_size, n_hidden)).place(hidden)
ti.root.dense(ti.ijk, (max_steps, batch_size, n_springs)).place(act)
ti.root.dense(ti.ij, (max_steps, batch_size)).place(center, target_v, target_h, height)
ti.root.place(loss, goal)
ti.root.lazy_grad()

pool = ti.field(ti.f32, shape = (100 * batch_size))


@ti.kernel
def compute_center(t: ti.i32):
    for k in range(batch_size):
        c = ti.Vector([0.0, 0.0])
        for i in ti.static(range(n_objects)):
            c += x[t, k, i]
        center[t, k] = (1.0 / n_objects) * c


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
def nn1(t: ti.i32):
    for k, i in ti.ndrange(batch_size, n_hidden):
        actuation = 0.0
        for j in ti.static(range(n_sin_waves)):
            actuation += weights1[i, j] * ti.sin(spring_omega * t * dt + 2 * math.pi / n_sin_waves * j)
        for j in ti.static(range(n_objects)):
            offset = x[t, k, j] - center[t, k]
            # use a smaller weight since there are too many of them
            actuation += weights1[i, j * 4 + n_sin_waves] * offset[0] * 0.05
            actuation += weights1[i, j * 4 + n_sin_waves + 1] * offset[1] * 0.05
            actuation += weights1[i, j * 4 + n_sin_waves + 2] * v[t, k, j][0] * 0.05
            actuation += weights1[i, j * 4 + n_sin_waves + 3] * v[t, k, j][1] * 0.05
        if ti.static(duplicate_v > 0):
            for j in ti.static(range(duplicate_v)):
                actuation += weights1[i, n_objects * 4 + n_sin_waves + j * 2] * target_v[t, k][0]
                actuation += weights1[i, n_objects * 4 + n_sin_waves + j * 2 + 1] * target_v[t, k][1]
        if ti.static(duplicate_h > 0):
            for j in ti.static(range(duplicate_h)):
                actuation += weights1[i, n_objects * 4 + n_sin_waves + duplicate_v * 2 + j] * target_h[t, k]
        actuation += bias1[i]
        actuation = ti.sin(actuation)
        hidden[t, k, i] = actuation


@ti.kernel
def nn2(t: ti.i32):
    for k, i in ti.ndrange(batch_size, n_springs):
        actuation = 0.0
        for j in ti.static(range(n_hidden)):
            actuation += weights2[i, j] * hidden[t, k, j]
        actuation += bias2[i]
        actuation = ti.sin(actuation)
        act[t, k, i] = actuation


@ti.kernel
def apply_spring_force(t: ti.i32):
    for k, i in ti.ndrange(batch_size, n_springs):
        a = spring_anchor_a[i]
        b = spring_anchor_b[i]
        pos_a = x[t, k, a]
        pos_b = x[t, k, b]
        dist = pos_a - pos_b
        length = dist.norm(1e-8) + 1e-4

        target_length = spring_length[i] * (1.0 + spring_actuation[i] * act[t, k, i])
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
        old_v = s * v[t - 1, k, i] + dt * gravity * ti.Vector([0.0, 1.0]) + v_inc[t, k, i]
        old_x = x[t - 1, k, i]
        new_x = old_x + dt * old_v
        toi = 0.0
        new_v = old_v
        if new_x[1] < ground_height and old_v[1] < -1e-4:
            toi = -(old_x[1] - ground_height) / old_v[1]
            new_v = ti.Vector([0.0, 0.0])
        new_x = old_x + toi * old_v + (dt - toi) * new_v

        v[t, k, i] = new_v
        x[t, k, i] = new_x


@ti.kernel
def compute_loss(t: ti.i32):
    for k in range(batch_size):
        loss[None] += (center[t, k](0) - center[t - run_period, k](0) - target_v[t - run_period, k](0))**2 / batch_size
    # if k == 0:
    #     print("Mark run: ", center[t, 0](0) - center[t - run_period, 0](0), target_v[t - run_period, 0](0))


@ti.kernel
def compute_loss_h(t: ti.i32):
    for k in range(batch_size):
        loss[None] += (height[t, k] - target_h[t, k]) ** 2 / batch_size * 5.
    # if k == 0:
    #     print("Mark jump:", height[t, k], target_h[t, k])


@ti.kernel
def compute_loss_pose(t: ti.i32):
    for k, i in ti.ndrange(batch_size, n_objects):
        ti.atomic_add(loss[None], (x[t, k, i](0) - center[t, k](0) - x[0, k, i](0) + center[0, k](0)) ** 2 / batch_size)
        ti.atomic_add(loss[None], (x[t, k, i](1) - center[t, k](1) - x[0, k, i](1) + center[0, k](1)) ** 2 / batch_size)

@ti.kernel
def compute_weight_decay():
    for I in ti.grouped(weights1):
        loss[None] += weight_decay * weights1[I] ** 2
    for I in ti.grouped(weights2):
        loss[None] += weight_decay * weights2[I] ** 2


gui = ti.GUI(show_gui=False)

@ti.kernel
def initialize_validate(total_steps: ti.i32, output_v: ti.f32, output_h: ti.f32):
    for t, k in ti.ndrange(total_steps, batch_size):
        target_v[t, k][0] = ((t // turn_period) % 2 * 2 - 1) * output_v
        target_h[t, k] = output_h

@ti.kernel
def initialize_train(total_steps: ti.i32):
    for _ in pool:
        pool[_] = (ti.random() - 0.5) * 2
    for t, k in ti.ndrange(total_steps, batch_size):
        target_v[t, k][0] = pool[t // turn_period + 100 * k] * 0.08
        target_h[t, k] = ti.random() * 0.25 + 0.1



@ti.kernel
def clear_states():
    for t, k, i in ti.ndrange(max_steps, batch_size, n_objects):
        x.grad[t, k, i] = ti.Vector([0.0, 0.0])
        v.grad[t, k, i] = ti.Vector([0.0, 0.0])
        v_inc[t, k, i] = ti.Vector([0.0, 0.0])
        v_inc.grad[t, k, i] = ti.Vector([0.0, 0.0])


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
    clear()
    if random.random() > 0.5:
        goal[None] = [0.9, 0.2]
    else:
        goal[None] = [0.1, 0.2]
    goal[None] = [0.9, 0.2]

    total_steps = train_steps if train else validate_steps
    
    if train:
        initialize_train(total_steps)
    else:
        initialize_validate(total_steps, output_v, output_h)

@debug
def forward(train = True, prefix = None):
    total_steps = train_steps if train else validate_steps
    for t in range(1, total_steps):
        compute_center(t - 1)
        compute_height(t - 1)
        nn1(t - 1)
        nn2(t - 1)
        apply_spring_force(t - 1)
        advance_toi(t)
    for t in range(1, total_steps):
        if duplicate_v > 0 and (t - 1) % turn_period > run_period:
            compute_loss(t - 1)
        if duplicate_h > 0 and (t - 1) % jump_period == jump_period - 1:
            for k in range(batch_size):
                compute_loss_h(t - 1)
                # compute_loss_pose(t - 1)

    # print("Speed= ", math.sqrt(loss[None] / loss_cnt))
    #compute_weight_decay()

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
                            color=0x0,
                            radius=3)

                    def circle(x, y, color):
                        gui.circle((x, y), ti.rgb_to_hex(color), 7)

                    for i in range(n_springs):

                        def get_pt(x):
                            return (x[0], x[1])

                        a = act[t - 1, 0, i] * 0.5
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

                    for i in range(n_objects):
                        color = (0.4, 0.6, 0.6)
                        if i == head_id:
                            color = (0.8, 0.2, 0.3)
                        circle(x[t, 0, i][0], x[t, 0, i][1], color)
                    # circle(goal[None][0], goal[None][1], (0.6, 0.2, 0.2))

                    if target_v[t, 0][0] > 0:
                        circle(0.5, 0.5, (1, 0, 0))
                        circle(0.6, 0.5, (1, 0, 0))
                    else:
                        circle(0.5, 0.5, (0, 0, 1))
                        circle(0.4, 0.5, (0, 0, 1))

                    gui.show('mass_spring/{}/{:04d}.png'.format(prefix, t))

    output_loss.append(loss[None])

    if visualize:
        utils.plot_curve(output_loss, "training_curve.png")
        utils.plot_curve(output_loss[-200:], "training_curve_last_200.png")

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
    else:
        forward(train = train, prefix = prefix)
    
    visualizer(train = train, prefix = prefix, visualize = visualize)

simulate.cnt = 0


def setup_robot():
    print('n_objects=', n_objects, '   n_springs=', n_springs)

    for k in range(batch_size):
        for i in range(n_objects):
            x[0, k, i] = [objects[i][0] + 0.4, objects[i][1]]

    for i in range(n_springs):
        s = springs[i]
        spring_anchor_a[i] = s[0]
        spring_anchor_b[i] = s[1]
        spring_length[i] = s[2]
        spring_stiffness[i] = s[3] / 10
        spring_actuation[i] = s[4]

@ti.kernel
def adam_update(w: ti.template(), m: ti.template(), v: ti.template(), iter: ti.i32):
    for I in ti.grouped(w):
        m[I] = adam_b1 * m[I] + (1 - adam_b1) * w.grad[I]
        v[I] = adam_b2 * v[I] + (1 - adam_b2) * w.grad[I] * w.grad[I]
        m_cap = m[I] / (1 - adam_b1 ** (iter + 1))
        v_cap = v[I] / (1 - adam_b2 ** (iter + 1))
        w[I] -= (adam_a * m_cap) / (ti.sqrt(v_cap) + 1e-8)

def optimize(visualize):
    for i in range(n_hidden):
        for j in range(n_input_states()):
            weights1[i, j] = np.random.randn() * math.sqrt(
                2 / (n_hidden + n_input_states())) * 2

    for i in range(n_springs):
        for j in range(n_hidden):
            # TODO: n_springs should be n_actuators
            weights2[i, j] = np.random.randn() * math.sqrt(
                2 / (n_hidden + n_springs)) * 3

    losses = []
    # simulate('initial{}'.format(robot_id), visualize=visualize)

    for iter in range(1000):
        print("-------------------- iter #{} --------------------".format(iter))

        simulate(visualize=iter % 10 == 0)

        print('Iter=', iter, 'Loss=', loss[None])

        total_norm_sqr = 0
        for i in range(n_hidden):
            for j in range(n_input_states()):
                total_norm_sqr += weights1.grad[i, j]**2
            total_norm_sqr += bias1.grad[i]**2

        for i in range(n_springs):
            for j in range(n_hidden):
                total_norm_sqr += weights2.grad[i, j]**2
            total_norm_sqr += bias2.grad[i]**2

        print("TNS= ", total_norm_sqr)

        adam_update(weights1, m_weights1, v_weights1, iter)
        adam_update(bias1, m_bias1, v_bias1, iter)
        adam_update(weights2, m_weights2, v_weights2, iter)
        adam_update(bias2, m_bias2, v_bias2, iter)
        losses.append(loss[None])

        # print(time.time() - t, ' 2')

        if iter % 200 == 199:
            simulate(0.07, 0)
            simulate(0.03, 0)
            simulate(0.01, 0)

    return losses


if __name__ == '__main__':
    setup_robot()
    optimize(visualize=False)
