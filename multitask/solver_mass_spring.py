import taichi as ti
from config import *
from utils import *

@ti.data_oriented
class SolverMassSpring:
    def __init__(self):
        self.x = vec()
        self.v = vec()
        self.center = vec()
        self.actuation = scalar()
        ti.root.dense(ti.ijk, (max_steps, batch_size, n_objects)).place(self.x, self.v)
        ti.root.dense(ti.ij, (max_steps, batch_size)).place(self.center)
        ti.root.dense(ti.ijk, (max_steps, batch_size, n_springs)).place(self.actuation)

        self.height = scalar()
        self.rotation = scalar()
        self.head_center = vec()
        self.head_counter = scalar()
        self.tail_center = vec()
        self.tail_counter = scalar()
        ti.root.dense(ti.ij, (max_steps, batch_size)).place(self.height, self.rotation, self.head_center,
                                                            self.head_counter, self.tail_center, self.tail_counter)

        self.spring_anchor_a = ti.field(ti.i32)
        self.spring_anchor_b = ti.field(ti.i32)
        self.spring_length = scalar()
        self.spring_stiffness = scalar()
        self.spring_actuation = scalar()
        self.v_inc = vec()
        ti.root.dense(ti.i, n_springs).place(self.spring_anchor_a, self.spring_anchor_b,
                                             self.spring_length, self.spring_stiffness,
                                             self.spring_actuation)
        ti.root.dense(ti.ijk, (max_steps, batch_size, n_objects)).place(self.v_inc)

    def initialize_robot(self):
        for i in range(n_springs):
            s = springs[i]
            self.spring_anchor_a[i] = s[0]
            self.spring_anchor_b[i] = s[1]
            self.spring_length[i] = s[2]
            self.spring_stiffness[i] = s[3] / 10
            self.spring_actuation[i] = s[4]
            if s[4] > 0:
                print("spring: ", i)

    @ti.kernel
    def clear_states(self, steps: ti.template()):
        for t, k, i in ti.ndrange(steps, batch_size, n_objects):
            self.x.grad[t, k, i] = ti.Matrix.zero(real, dim, 1)
            self.v.grad[t, k, i] = ti.Matrix.zero(real, dim, 1)
            self.v_inc[t, k, i] = ti.Matrix.zero(real, dim, 1)
            self.v_inc.grad[t, k, i] = ti.Matrix.zero(real, dim, 1)
        for t, k in ti.ndrange(steps, batch_size):
            self.head_center[t, k] = ti.Matrix.zero(real, dim, 1)
            self.head_counter[t, k] = 0.
            self.tail_center[t, k] = ti.Matrix.zero(real, dim, 1)
            self.tail_counter[t, k] = 0.
            self.rotation[t, k] = 0.

    @ti.kernel
    def apply_spring_force(self, t: ti.i32):
        for k, i in ti.ndrange(batch_size, n_springs):
            a = self.spring_anchor_a[i]
            b = self.spring_anchor_b[i]
            pos_a = self.x[t, k, a]
            pos_b = self.x[t, k, b]
            dist = pos_a - pos_b
            length = dist.norm(1e-8) + 1e-4

            target_length = self.spring_length[i] * (1.0 + self.spring_actuation[i] * self.actuation[t, k, i])
            impulse = dt * (length - target_length) * self.spring_stiffness[i] / length * dist

            # Dashpot damping
            x_ij = self.x[t, k, a] - self.x[t, k, b]
            d = x_ij.normalized()
            v_rel = (self.v[t, k, a] - self.v[t, k, b]).dot(d)
            impulse += dashpot_damping * v_rel * d

            ti.atomic_add(self.v_inc[t, k, a], -impulse)
            ti.atomic_add(self.v_inc[t, k, b], impulse)

    @ti.kernel
    def pass_actuation(self, t: ti.i32, k: ti.i32, i: ti.i32, act: ti.f32):
        self.actuation[t, k, i] = act

    @ti.kernel
    def advance_toi(self, t: ti.i32):
        for k, i in ti.ndrange(batch_size, n_objects):
            s = math.exp(-dt * drag_damping)
            unitY = ti.Matrix.zero(real, dim, 1)
            unitY[1] = 1.0
            old_v = s * self.v[t - 1, k, i] + dt * gravity * unitY + self.v_inc[t - 1, k, i]
            old_x = self.x[t - 1, k, i]
            new_x = old_x + dt * old_v
            toi = 0.0
            new_v = old_v
            if new_x[1] < ground_height and old_v[1] < -1e-4:
                toi = -(old_x[1] - ground_height) / old_v[1]
                new_v = ti.Matrix.zero(real, dim, 1)
            new_x = old_x + toi * old_v + (dt - toi) * new_v

            self.v[t, k, i] = new_v
            self.x[t, k, i] = new_x

    @ti.kernel
    def compute_center(self, t: ti.i32):
        n = ti.static(n_objects)
        for k in range(batch_size):
            self.center[t, k] = ti.Matrix.zero(real, dim, 1)
        for k, i in ti.ndrange(batch_size, n):
            self.center[t, k] += self.x[t, k, i] / n

    @ti.kernel
    def compute_height(self, t: ti.i32):
        for k in range(batch_size):
            h = 10.
            for i in ti.static(range(n_objects)):
                h = ti.min(h, self.x[t, k, i](1))
            if t % jump_period == 0:
                self.height[t, k] = h
            else:
                self.height[t, k] = ti.max(self.height[t - 1, k], h)

    @ti.kernel
    def compute_rotation(self, t: ti.i32):
        for k in range(batch_size):
            for i in ti.static(range(n_objects)):
                if self.x[0, k, i](0) < self.center[0, k](0):
                    self.head_center[t, k] += self.x[t, k, i]
                    self.head_counter[t, k] += 1.
                else:
                    self.tail_center[t, k] += self.x[t, k, i]
                    self.tail_counter[t, k] += 1.
            direction = -self.head_center[t, k] * self.tail_counter[t, k] + self.tail_center[t, k] * self.head_counter[t, k]
            self.rotation[t, k] = ti.atan2(direction[2], direction[0])

    def advance(self, t):
        self.compute_center(t)
        self.compute_height(t)
        if dim == 3:
            self.compute_rotation(t)
        self.apply_spring_force(t)
        self.advance_toi(t + 1)

    def draw_robot(self, gui, t, target_v):
        def circle(x, y, color):
            gui.circle((x, y), ti.rgb_to_hex(color), 7)
        # draw segments
        for i in range(n_springs):
            def get_pt(x):
                return (x[0], x[1])
            a = self.actuation[t - 1, 0, i] * 0.5
            r = 2
            if self.spring_actuation[i] == 0:
                a = 0
                c = 0x222222
            else:
                r = 4
                c = ti.rgb_to_hex((0.5 + a, 0.5 - abs(a), 0.5 - a))
            gui.line(get_pt(self.x[t, 0, self.spring_anchor_a[i]]),
                     get_pt(self.x[t, 0, self.spring_anchor_b[i]]),
                     color=c,
                     radius=r)
        # draw points
        for i in range(n_objects):
            color = (0.06640625, 0.06640625, 0.06640625)
            circle(self.x[t, 0, i][0], self.x[t, 0, i][1], color)
        if target_v[t, 0][0] > 0:
            circle(0.5, 0.5, (1, 0, 0))
            circle(0.6, 0.5, (1, 0, 0))
        else:
            circle(0.5, 0.5, (0, 0, 1))
            circle(0.4, 0.5, (0, 0, 1))
