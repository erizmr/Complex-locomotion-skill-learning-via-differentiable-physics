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
ti.init(arch=ti.gpu, default_fp=real)

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

n_hidden = 32
weights2 = scalar()
bias2 = scalar()
hidden = scalar()

m_weights1, v_weights1 = scalar(), scalar()
m_bias1, v_bias1 = scalar(), scalar()
m_weights2, v_weights2 = scalar(), scalar()
m_bias2, v_bias2 = scalar(), scalar()

center = vec()
height = scalar()
duplicate_v = 0
duplicate_h = 1
target_v = vec()
target_h = scalar()
weight_v = 1.
weight_h = 1.

act = scalar()

dt = 0.004

cycle_period = 100
turn_period = 400
spring_omega = 2 * math.pi / dt / cycle_period
print(spring_omega)
drag_damping = 0
dashpot_damping = 0.2

batch_size = int(sys.argv[2])

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
        if t % cycle_period == 0:
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
        actuation = ti.tanh(actuation)
        hidden[t, k, i] = actuation


@ti.kernel
def nn2(t: ti.i32):
    for k, i in ti.ndrange(batch_size, n_springs):
        actuation = 0.0
        for j in ti.static(range(n_hidden)):
            actuation += weights2[i, j] * hidden[t, k, j]
        actuation += bias2[i]
        actuation = ti.tanh(actuation)
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
def compute_loss(t: ti.i32, k: ti.i32):
    ti.atomic_add(loss[None], (center[t, k](0) - center[t - cycle_period, k](0) - target_v[t - cycle_period, k](0))**2 / batch_size)
    if k == 0:
        print("Mark run: ", center[t, 0](0) - center[t - cycle_period, 0](0), target_v[t - cycle_period, 0](0))


@ti.kernel
def compute_loss_h(t: ti.i32, k: ti.i32):
    ti.atomic_add(loss[None], (height[t, k] - target_h[t, k]) ** 2 / batch_size * 30.)
    if k == 0:
        print("Mark jump:", height[t, k], target_h[t, k])


gui = ti.GUI(show_gui=False)


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
    pool = [(random.random() - 0.5) * 2 for _ in range(100 * batch_size)]
    for t in range(total_steps):
        for k in range(batch_size):
            if output:
                target_v[t, k][0] = ((t // turn_period) % 2 * 2 - 1) * output
            else:
                target_v[t, k][0] = pool[t // turn_period + 100 * k] * 0.07
            target_h[t, k] = 0.5
    loss_cnt = 0.
    for t in range(1, total_steps):
        compute_center(t - 1)
        compute_height(t - 1)
        nn1(t - 1)
        nn2(t - 1)
        apply_spring_force(t - 1)
        advance_toi(t)
        if duplicate_v > 0 and t - 1 > cycle_period:
            if (t - 1) % cycle_period > 0.1 * cycle_period:
                if (t - 1) % cycle_period < 0.9 * cycle_period:
                    for k in range(batch_size):
                        loss_cnt += 1.
                        compute_loss(t - 1, k)
        if duplicate_h > 0 and (t - 1) % cycle_period == cycle_period - 1:
            for k in range(batch_size):
                loss_cnt += 1.
                compute_loss_h(t - 1, k)
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

    a = 0.01
    b_1=0.9
    b_2=0.999

    for iter in range(1000):
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

        for i in range(n_hidden):
            for j in range(n_input_states()):
                m_weights1[i, j] = b_1 * m_weights1[i, j] + (1 - b_1) * weights1.grad[i, j]
                v_weights1[i, j] = b_2 * v_weights1[i, j] + (1 - b_2) * weights1.grad[i, j] * weights1.grad[i, j]
                m_cap = m_weights1[i, j] / (1 - b_1 ** (iter + 1))
                v_cap = v_weights1[i, j] / (1 - b_2 ** (iter + 1))
                weights1[i, j] -= (a * m_cap) / (math.sqrt(v_cap) + 1e-8)
            m_bias1[i] = b_1 * m_bias1[i] + (1 - b_1) * bias1.grad[i]
            v_bias1[i] = b_2 * v_bias1[i] + (1 - b_2) * bias1.grad[i] * bias1.grad[i]
            m_cap = m_bias1[i] / (1 - b_1 ** (iter + 1))
            v_cap = v_bias1[i] / (1 - b_2 ** (iter + 1))
            bias1[i] -= (a * m_cap) / (math.sqrt(v_cap) + 1e-8)

        for i in range(n_springs):
            for j in range(n_hidden):
                m_weights2[i, j] = b_1 * m_weights2[i, j] + (1 - b_1) * weights2.grad[i, j]
                v_weights2[i, j] = b_2 * v_weights2[i, j] + (1 - b_2) * weights2.grad[i, j] * weights2.grad[i, j]
                m_cap = m_weights2[i, j] / (1 - b_1 ** (iter + 1))
                v_cap = v_weights2[i, j] / (1 - b_2 ** (iter + 1))
                weights2[i, j] -= (a * m_cap) / (math.sqrt(v_cap) + 1e-8)
            m_bias2[i] = b_1 * m_bias2[i] + (1 - b_1) * bias2.grad[i]
            v_bias2[i] = b_2 * v_bias2[i] + (1 - b_2) * bias2.grad[i] * bias2.grad[i]
            m_cap = m_bias2[i] / (1 - b_1 ** (iter + 1))
            v_cap = v_bias2[i] / (1 - b_2 ** (iter + 1))
            bias2[i] -= (a * m_cap) / (math.sqrt(v_cap) + 1e-8)
        losses.append(loss[None])

        # print(time.time() - t, ' 2')

        if iter % 200 == 199:
            clear()
            forward(0.07)
            clear()
            forward(0.03)
            clear()
            forward(0.01)

    losses = gaussian_filter(losses, 10)
    return losses


if __name__ == '__main__':
    setup_robot()
    optimize(visualize=False)
