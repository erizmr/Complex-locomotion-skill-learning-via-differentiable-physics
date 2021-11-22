import math
from multitask.config_sim import ConfigSim
import taichi as ti
from multitask.utils import vec, scalar, real


@ti.data_oriented
class SolverMassSpring:
    def __init__(self, config: ConfigSim):
        self.config = config
        self.max_steps = config.get_config()["process"]["max_steps"]
        self.dt = config.get_config()["process"]["dt"]
        self.ground_height = config.get_config()["simulator"]["ground_height"]
        self.gravity = config.get_config()["simulator"]["gravity"]
        self.drag_damping = config.get_config()["simulator"]["drag_damping"]
        self.dashpot_damping = config.get_config()["simulator"]["dashpot_damping"]
        self.batch_size = config.get_config()["nn"]["batch_size"]
        self.n_objects = config.get_config()["robot"]["n_objects"]
        self.n_springs = config.get_config()["robot"]["n_springs"]
        self.springs = config.get_config()["robot"]["springs"]
        self.dim = config.get_config()["robot"]["dim"]
        self.jump_period = config.get_config()["process"]["jump_period"]

        # Sticky boundary if friction coefficient is -1.0
        self.friction_coefficient = -1.0
        if "friction" in config.get_config()["simulator"]:
            self.friction_coefficient = config.get_config()["simulator"]["friction"]

        self.n_models = config.get_config()["nn"]["n_models_used"]
        print(f"n models Mass-Spring: {self.n_models}")
        self.default_model_id = 0

        self.x = vec(self.dim)
        self.v = vec(self.dim)
        self.center = vec(self.dim)
        self.actuation = scalar()
        self.act_list = []
        batch_node = ti.root.dense(ti.ijk, (self.n_models, self.max_steps, self.batch_size))
        batch_node.dense(ti.l, self.n_objects).place(self.x, self.v)
        batch_node.place(self.center)
        batch_node.dense(ti.l, self.n_springs).place(self.actuation)

        # height here is the lower height i.e., the lowest point of the robot
        self.height = scalar()
        self.upper_height = scalar()
        self.rotation = scalar()
        self.head_center = vec(self.dim)
        self.head_counter = scalar()
        self.tail_center = vec(self.dim)
        self.tail_counter = scalar()
        batch_node.place(self.height, self.rotation, self.head_center,
                                                            self.head_counter, self.tail_center, self.tail_counter,
                                                                      self.upper_height)

        self.spring_anchor_a = ti.field(ti.i32)
        self.spring_anchor_b = ti.field(ti.i32)
        self.spring_length = scalar()
        self.spring_stiffness = scalar()
        self.spring_actuation = scalar()
        self.v_inc = vec(self.dim)
        ti.root.dense(ti.i, self.n_springs).place(self.spring_anchor_a, self.spring_anchor_b,
                                             self.spring_length, self.spring_stiffness,
                                             self.spring_actuation)
        batch_node.dense(ti.l, self.n_objects).place(self.v_inc)

    def initialize_robot(self):
        for i in range(self.n_springs):
            s = self.springs[i]
            self.spring_anchor_a[i] = s[0]
            self.spring_anchor_b[i] = s[1]
            self.spring_length[i] = s[2]
            self.spring_stiffness[i] = s[3] / 10
            self.spring_actuation[i] = s[4]
            if s[4] > 0:
                self.act_list.append(i)

    @ti.kernel
    def clear_states(self, steps: ti.template()):
        for model_id, t, k, i in ti.ndrange(self.n_models, steps, self.batch_size, self.n_objects):
            # self.x.grad[model_id, t, k, i] = ti.Matrix.zero(real, self.dim, 1)
            # self.v.grad[model_id, t, k, i] = ti.Matrix.zero(real, self.dim, 1)
            self.v_inc[model_id, t, k, i] = ti.Matrix.zero(real, self.dim, 1)
            # self.v_inc.grad[model_id, t, k, i] = ti.Matrix.zero(real, self.dim, 1)
        for model_id, t, k in ti.ndrange(self.n_models, steps, self.batch_size):
            self.head_center[model_id, t, k] = ti.Matrix.zero(real, self.dim, 1)
            self.head_counter[model_id, t, k] = 0.
            self.tail_center[model_id, t, k] = ti.Matrix.zero(real, self.dim, 1)
            self.tail_counter[model_id, t, k] = 0.
            self.rotation[model_id, t, k] = 0.

    @ti.func
    def clamp(self, x):
        return ti.max(ti.min(x, 1.0), -1.0)

    @ti.kernel
    def apply_spring_force(self, t: ti.i32):
        for model_id, k, i in ti.ndrange(self.n_models, self.batch_size, self.n_springs):
            a = self.spring_anchor_a[i]
            b = self.spring_anchor_b[i]
            pos_a = self.x[model_id, t, k, a]
            pos_b = self.x[model_id, t, k, b]
            dist = pos_a - pos_b
            length = dist.norm(1e-8) + 1e-4

            # Clamp the actuation to -1 to 1
            self.actuation[model_id, t, k, i] = self.clamp(self.actuation[model_id, t, k, i])

            target_length = self.spring_length[i] * (1.0 + self.spring_actuation[i] * self.actuation[model_id, t, k, i])
            impulse = self.dt * (length - target_length) * self.spring_stiffness[i] / length * dist

            # Dashpot damping
            x_ij = self.x[model_id, t, k, a] - self.x[model_id, t, k, b]
            d = x_ij.normalized()
            v_rel = (self.v[model_id, t, k, a] - self.v[model_id, t, k, b]).dot(d)
            impulse += self.dashpot_damping * v_rel * d
            ti.atomic_add(self.v_inc[model_id, t, k, a], -impulse)
            ti.atomic_add(self.v_inc[model_id, t, k, b], impulse)

    @ti.kernel
    def pass_actuation(self, t: ti.i32, k: ti.i32, i: ti.i32, act: real):
            self.actuation[0, t, k, i] = max(min(act, 1), -1)

    @ti.kernel
    def pass_actuation_fast(self, t: ti.i32, act_spring: ti.ext_arr(), action: ti.ext_arr()):
        for k in ti.static(range(self.batch_size)):
            for i in range(act_spring.shape[0]):
                self.actuation[t, k, act_spring[i]] = action[i]

    @ti.kernel
    def advance_toi(self, t: ti.i32):
        for model_id, k, i in ti.ndrange(self.n_models, self.batch_size, self.n_objects):
            s = math.exp(-self.dt * self.drag_damping)
            unitY = ti.Matrix.zero(real, self.dim, 1)
            unitY[1] = 1.0
            old_v = s * self.v[model_id, t - 1, k, i] + self.dt * self.gravity * unitY + self.v_inc[model_id, t - 1, k, i]
            old_x = self.x[model_id, t - 1, k, i]
            new_x = old_x + self.dt * old_v
            toi = 0.0
            new_v = old_v
            if new_x[1] < self.ground_height and old_v[1] < -1e-4:
                toi = float(-(old_x[1] - self.ground_height) / old_v[1])
                new_v = ti.Matrix.zero(real, self.dim, 1)

                if self.friction_coefficient > 0:
                    # Sticky boundary if friction is -1.0, otherwise slip
                    friction = self.friction_coefficient
                    if old_v[0] < 0:
                        new_v[0] = ti.min(0., old_v[0] + friction * (-old_v[1]))
                    else:
                        new_v[0] = ti.max(0., old_v[0] - friction * (-old_v[1]))

            new_x = old_x + toi * old_v + (self.dt - toi) * new_v

            self.v[model_id, t, k, i] = new_v
            self.x[model_id, t, k, i] = new_x

    @ti.kernel
    def compute_center(self, t: ti.i32):
        n = ti.static(self.n_objects)
        for model_id, k in ti.ndrange(self.n_models, self.batch_size):
            self.center[model_id, t, k] = ti.Matrix.zero(real, self.dim, 1)
        for model_id, k, i in ti.ndrange(self.n_models, self.batch_size, n):
            self.center[model_id, t, k] += self.x[model_id, t, k, i] / n

    @ti.kernel
    def compute_height(self, t: ti.i32):
        for model_id, k in ti.ndrange(self.n_models, self.batch_size):
            h = 10.
            for i in ti.static(range(self.n_objects)):
                h = float(ti.min(h, self.x[model_id, t, k, i](1)))
            # self.height[t, k] = h
            if t % self.jump_period == 0:
                self.height[model_id, t, k] = h
            else:
                self.height[model_id, t, k] = ti.max(self.height[model_id, t - 1, k], h)

        for model_id, k in ti.ndrange(self.n_models, self.batch_size):
            h = -10.
            for i in ti.static(range(self.n_objects)):
                h = ti.max(h, self.x[model_id, t, k, i](1))
            self.upper_height[model_id, t, k] = h

    @ti.kernel
    def compute_rotation(self, t: ti.i32):
        for model_id, k in ti.ndrange(self.n_models, self.batch_size):
            for i in ti.static(range(self.n_objects)):
                if self.x[model_id, 0, k, i](0) < self.center[model_id, 0, k](0):
                    self.head_center[model_id, t, k] += self.x[model_id, t, k, i]
                    self.head_counter[model_id, t, k] += 1.
                else:
                    self.tail_center[model_id, t, k] += self.x[model_id, t, k, i]
                    self.tail_counter[model_id, t, k] += 1.
            direction = -self.head_center[model_id, t, k] * self.tail_counter[model_id, t, k] + self.tail_center[model_id, t, k] * self.head_counter[model_id, t, k]
            self.rotation[model_id, t, k] = ti.atan2(direction[2], direction[0])

    def pre_advance(self, t):
        self.compute_center(t)
        self.compute_height(t)
        if self.dim == 3:
            self.compute_rotation(t)

    def advance(self, t):
        self.apply_spring_force(t)
        self.advance_toi(t + 1)

    def draw_robot(self, gui, t, batch_rank, target_v, target_position=None):
        def circle(x, y, color):
            gui.circle((x, y), ti.rgb_to_hex(color), 7)
        # draw segments
        for i in range(self.n_springs):
            def get_pt(x):
                return x[0], x[1]
            a = self.actuation[self.default_model_id, t - 1, batch_rank, i] * 0.5
            r = 2
            if self.spring_actuation[i] == 0:
                a = 0
                c = 0x222222
            else:
                r = 4
                c = ti.rgb_to_hex((0.5 + a, 0.5 - abs(a), 0.5 - a))
            gui.line(get_pt(self.x[self.default_model_id, t, batch_rank, self.spring_anchor_a[i]]),
                     get_pt(self.x[self.default_model_id, t, batch_rank, self.spring_anchor_b[i]]),
                     color=c,
                     radius=r)
        # draw points
        for i in range(self.n_objects):
            color = (0.06640625, 0.06640625, 0.06640625)
            circle(self.x[self.default_model_id, t, batch_rank, i][0], self.x[self.default_model_id, t, batch_rank, i][1], color)
