from mass_spring_robot_config import robots
import random
import sys
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import taichi as ti
import math
import numpy as np
import os

real = ti.f64
ti.init(default_fp=real)

max_steps = 4096
vis_interval = 256
output_vis_interval = 8
steps = 2048 // 2
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

n_objects = 0
# target_ball = 0
elasticity = 0.0
ground_height = 0.1
gravity = -1.8
friction = 2.5

n_springs = 0
spring_anchor_a = ti.field(ti.i32)
spring_anchor_b = ti.field(ti.i32)
spring_length = scalar()
spring_stiffness = scalar()
spring_actuation = scalar()

n_sin_waves = 10
weights1 = scalar()
bias1 = scalar()

n_hidden = 32
weights2 = scalar()
bias2 = scalar()
hidden = scalar()

center = vec()
duplicate_v = 1
duplicate_h = 0
target_v = vec()
target_h = ti.field(ti.f64, shape=())
weight_v = 1.
weight_h = 1.

act = scalar()

dt = 0.004
learning_rate = 25

gradient_clip = 1
cycle_period = 100
turn_period = 400
spring_omega = 2 * math.pi / dt / cycle_period
print(spring_omega)
drag_damping = 0
dashpot_damping = 0.2


def n_input_states():
    return n_sin_waves + 4 * n_objects + 2 * duplicate_v + duplicate_h


@ti.layout
def place():
    ti.root.dense(ti.l, max_steps).dense(ti.i, n_objects).place(x, v, v_inc)
    ti.root.dense(ti.i, n_springs).place(spring_anchor_a, spring_anchor_b,
                                         spring_length, spring_stiffness,
                                         spring_actuation)
    ti.root.dense(ti.ij, (n_hidden, n_input_states())).place(weights1)
    ti.root.dense(ti.i, n_hidden).place(bias1)
    ti.root.dense(ti.ij, (n_springs, n_hidden)).place(weights2)
    ti.root.dense(ti.i, n_springs).place(bias2)
    ti.root.dense(ti.ij, (max_steps, n_hidden)).place(hidden)
    ti.root.dense(ti.ij, (max_steps, n_springs)).place(act)
    ti.root.dense(ti.i, max_steps).place(center, target_v)
    ti.root.place(loss, goal)
    ti.root.lazy_grad()


@ti.kernel
def compute_center(t: ti.i32):
    for _ in range(1):
        c = ti.Vector([0.0, 0.0])
        for i in ti.static(range(n_objects)):
            c += x[t, i]
        center[t] = (1.0 / n_objects) * c


@ti.kernel
def nn1(t: ti.i32):
    for i in range(n_hidden):
        actuation = 0.0
        for j in ti.static(range(n_sin_waves)):
            actuation += weights1[i, j] * ti.sin(spring_omega * t * dt +
                                                 2 * math.pi / n_sin_waves * j)
        for j in ti.static(range(n_objects)):
            offset = x[t, j] - center[t]
            # use a smaller weight since there are too many of them
            actuation += weights1[i, j * 4 + n_sin_waves] * offset[0] * 0.05
            actuation += weights1[i,
                                  j * 4 + n_sin_waves + 1] * offset[1] * 0.05
            actuation += weights1[i, j * 4 + n_sin_waves + 2] * v[t,
                                                                  j][0] * 0.05
            actuation += weights1[i, j * 4 + n_sin_waves + 3] * v[t,
                                                                  j][1] * 0.05
        if ti.static(duplicate_v > 0):
            for j in ti.static(range(duplicate_v)):
                actuation += weights1[i, n_objects * 4 + n_sin_waves + j * 2] * target_v[t][0]
                actuation += weights1[i, n_objects * 4 + n_sin_waves + j * 2 + 1] * target_v[t][1]
        if ti.static(duplicate_h > 0):
            for j in ti.static(range(duplicate_h)):
                actuation += weights1[i, n_objects * 4 + n_sin_waves + duplicate_v * 2 + j] * target_h[None]
        actuation += bias1[i]
        actuation = ti.tanh(actuation)
        hidden[t, i] = actuation


@ti.kernel
def nn2(t: ti.i32):
    for i in range(n_springs):
        actuation = 0.0
        for j in ti.static(range(n_hidden)):
            actuation += weights2[i, j] * hidden[t, j]
        actuation += bias2[i]
        actuation = ti.tanh(actuation)
        act[t, i] = actuation


@ti.kernel
def apply_spring_force(t: ti.i32):
    for i in range(n_springs):
        a = spring_anchor_a[i]
        b = spring_anchor_b[i]
        pos_a = x[t, a]
        pos_b = x[t, b]
        dist = pos_a - pos_b
        length = dist.norm(1e-8) + 1e-4

        target_length = spring_length[i] * (1.0 +
                                            spring_actuation[i] * act[t, i])
        impulse = dt * (length -
                        target_length) * spring_stiffness[i] / length * dist

        # Dashpot damping
        x_ij = x[t, a] - x[t, b]
        d = x_ij.normalized()
        v_rel = (v[t, a] - v[t, b]).dot(d)
        impulse += dashpot_damping * v_rel * d

        ti.atomic_add(v_inc[t + 1, a], -impulse)
        ti.atomic_add(v_inc[t + 1, b], impulse)


@ti.kernel
def advance_toi(t: ti.i32):
    for i in range(n_objects):
        s = math.exp(-dt * drag_damping)
        old_v = s * v[t - 1, i] + dt * gravity * ti.Vector([0.0, 1.0
                                                            ]) + v_inc[t, i]
        old_x = x[t - 1, i]
        new_x = old_x + dt * old_v
        toi = 0.0
        new_v = old_v
        if new_x[1] < ground_height and old_v[1] < -1e-4:
            toi = -(old_x[1] - ground_height) / old_v[1]
            new_v = ti.Vector([0.0, 0.0])
        new_x = old_x + toi * old_v + (dt - toi) * new_v

        v[t, i] = new_v
        x[t, i] = new_x


@ti.kernel
def compute_loss(t: ti.i32):
    # c = ti.Vector([0.0, 0.0])
    # for i in ti.static(range(n_objects)):
    #     c += v[t, i]
    # c = (1.0 / n_objects) * c
    # ti.atomic_add(loss[None], dt * weight_v * (target_v[t][0] - c[0])**2)
    # noinspection PyInterpreter
    ti.atomic_add(loss[None], (center[t](0) - center[t - cycle_period](0) - target_v[t - cycle_period](0))**2)
    print("Mark: ", center[t](0) - center[t - cycle_period](0), target_v[t - cycle_period](0))


@ti.kernel
def compute_loss_h(t: ti.i32):
    # ti.atomic_add(loss[None], weight_h * (target_h[None] - center[t][1])**2)
    ti.atomic_add(loss[None], weight_h * (target_h[None] - x[t, head_id][1])**2)


gui = ti.GUI("Mass Spring Robot", (512, 512), background_color=0xFFFFFF)


def forward(output=None, visualize=True):
    if random.random() > 0.5:
        goal[None] = [0.9, 0.2]
    else:
        goal[None] = [0.1, 0.2]
    goal[None] = [0.9, 0.2]

    interval = vis_interval
    if output:
        interval = output_vis_interval
        os.makedirs('mass_spring/{}/'.format(output), exist_ok=True)

    total_steps = steps if not output else steps * 2

    # range (-1, 1)
    pool = [(random.random() - 0.5) * 2 for _ in range(100)]
    for i in range(total_steps):
        if output:
            # target_v[i][0] = -0.03 if output < 0.04 else 0.03
            target_v[i][0] = ((i // turn_period) % 2 * 2 - 1) * 0.03
        else:
            # target_v[i][0] = (pool[0] + 1) * 0.05
            # target_v[i][0] = -0.03 if pool[i // turn_period] < 0 else 0.03
            target_v[i][0] = -0.03 if pool[0] < 0 else 0.03
        target_v[i][0] = 0.07
    if output:
        target_h[None] = 0.5
    else:
        target_h[None] = 0.4 + random.random() * 0.2
    loss_cnt = 0.
    for t in range(1, total_steps):
        compute_center(t - 1)
        nn1(t - 1)
        nn2(t - 1)
        apply_spring_force(t - 1)
        advance_toi(t)
        if duplicate_v > 0 and t - 1 > cycle_period and (t - 1) % cycle_period == 0:
            loss_cnt += 1.
            compute_loss(t - 1)
        # if duplicate_h > 0 and t % cycle_period == cycle_period // 2:
        #     compute_loss_h(t - 1)
        # output_target.append(target_v[t][0])
        # output_sim.append(v[t, head_id][0])
        # output_target.append(target_h[None])
        # output_sim.append(x[t, head_id][1])

        if (t + 1) % interval == 0 and visualize:
            gui.clear()
            gui.line((0, ground_height), (1, ground_height),
                     color=0x0,
                     radius=3)

            def circle(x, y, color):
                gui.circle((x, y), ti.rgb_to_hex(color), 7)

            for i in range(n_springs):

                def get_pt(x):
                    return (x[0], x[1])

                a = act[t - 1, i] * 0.5
                r = 2
                if spring_actuation[i] == 0:
                    a = 0
                    c = 0x222222
                else:
                    r = 4
                    c = ti.rgb_to_hex((0.5 + a, 0.5 - abs(a), 0.5 - a))
                gui.line(get_pt(x[t, spring_anchor_a[i]]),
                         get_pt(x[t, spring_anchor_b[i]]),
                         color=c,
                         radius=r)

            for i in range(n_objects):
                color = (0.4, 0.6, 0.6)
                if i == head_id:
                    color = (0.8, 0.2, 0.3)
                circle(x[t, i][0], x[t, i][1], color)
            # circle(goal[None][0], goal[None][1], (0.6, 0.2, 0.2))

            if target_v[t][0] > 0:
                circle(0.5, 0.5, (1, 0, 0))
                circle(0.6, 0.5, (1, 0, 0))
            else:
                circle(0.5, 0.5, (0, 0, 1))
                circle(0.4, 0.5, (0, 0, 1))

            if output:
                gui.show('mass_spring/{}/{:04d}.png'.format(output, t))
            else:
                gui.show()
    print("Speed= ", math.sqrt(loss[None] / loss_cnt))

    if output is None:
        output_loss.append(loss[None])
        fig = plt.figure()
        temp_loss = gaussian_filter(output_loss, 10)
        plt.plot(temp_loss)
        fig.savefig('plots/' + str(forward.cnt) + '.png', dpi=fig.dpi)
        plt.close(fig)

    # fig = plt.figure()
    # plt.plot(range(len(output_target)), output_target, 'o', markersize=2, label="target")
    # plt.plot(range(len(output_sim)), output_sim, 'o', markersize=2, label="train")
    # output_target.clear()
    # output_sim.clear()
    # plt.legend()
    # forward.cnt += 1
    # fig.savefig('plots/' + str(forward.cnt) + '.png', dpi=fig.dpi)
    # plt.close(fig)


forward.cnt = 0


@ti.kernel
def clear_states():
    for t in range(0, max_steps):
        for i in range(0, n_objects):
            x.grad[t, i] = ti.Vector([0.0, 0.0])
            v.grad[t, i] = ti.Vector([0.0, 0.0])
            v_inc[t, i] = ti.Vector([0.0, 0.0])
            v_inc.grad[t, i] = ti.Vector([0.0, 0.0])


def clear():
    clear_states()


def setup_robot(objects, springs):
    global n_objects, n_springs
    n_objects = len(objects)
    n_springs = len(springs)

    print('n_objects=', n_objects, '   n_springs=', n_springs)

    for i in range(n_objects):
        x[0, i] = [objects[i][0] + 0.4, objects[i][1]]

    for i in range(n_springs):
        s = springs[i]
        spring_anchor_a[i] = s[0]
        spring_anchor_b[i] = s[1]
        spring_length[i] = s[2]
        spring_stiffness[i] = s[3] / 10
        spring_actuation[i] = s[4]


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
    # forward('initial{}'.format(robot_id), visualize=visualize)
    for iter in range(200):
        clear()

        import time
        t = time.time()
        with ti.Tape(loss):
            forward(visualize=iter % 10 == 0)
        # print(time.time() - t, ' 1')

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

        # scale = learning_rate * min(1.0, gradient_clip / total_norm_sqr ** 0.5)
        gradient_clip = 0.1
        scale = gradient_clip / (total_norm_sqr**0.5 + 1e-6)
        for i in range(n_hidden):
            for j in range(n_input_states()):
                weights1[i, j] -= scale * weights1.grad[i, j]
            bias1[i] -= scale * bias1.grad[i]

        for i in range(n_springs):
            for j in range(n_hidden):
                weights2[i, j] -= scale * weights2.grad[i, j]
            bias2[i] -= scale * bias2.grad[i]
        losses.append(loss[None])

        # print(time.time() - t, ' 2')

    losses = gaussian_filter(losses, 10)
    return losses


robot_id = 0
if len(sys.argv) != 2:
    print("Usage: python3 mass_spring_interactive.py [robot_id=0, 1, 2, ...]")
    exit(0)
else:
    robot_id = int(sys.argv[1])

if __name__ == '__main__':
    setup_robot(*robots[robot_id]())

    optimize(visualize=False)
    clear()
    forward(0.07)
    clear()
    forward(0.03)
    clear()
    forward(0.01)
