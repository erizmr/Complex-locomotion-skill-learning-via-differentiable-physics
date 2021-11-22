import taichi as ti
from multitask.utils import vec, scalar, mat, real
from multitask.config_sim import ConfigSim


@ti.data_oriented
class SolverMPM:
    def __init__(self, config: ConfigSim):
        self.config = config
        self.simulator = 'mpm'
        # General parameters
        self.max_steps = config.get_config()["process"]["max_steps"]
        self.dt = config.get_config()["process"]["dt"]
        self.jump_period = config.get_config()["process"]["jump_period"]
        self.ground_height = config.get_config()["simulator"]["ground_height"]
        self.gravity = config.get_config()["simulator"]["gravity"]
        self.drag_damping = config.get_config()["simulator"]["drag_damping"]
        self.dashpot_damping = config.get_config()["simulator"]["dashpot_damping"]
        self.batch_size = config.get_config()["nn"]["batch_size"]
        self.n_objects = config.get_config()["robot"]["n_objects"]
        self.n_springs = config.get_config()["robot"]["n_springs"]
        self.n_squares = config.get_config()["robot"]["n_squares"]
        self.n_squ = config.get_config()["robot"]["n_squ"]
        self.springs = config.get_config()["robot"]["springs"]
        self.dim = config.get_config()["robot"]["dim"]

        # Parameters for MPM
        self.n_particles = config.get_config()["simulator"]["n_particles"]
        self.n_grid = config.get_config()["simulator"]["mpm"]["n_grid"]
        self.dx = config.get_config()["simulator"]["mpm"]["dx"]
        self.inv_dx = config.get_config()["simulator"]["mpm"]["inv_dx"]
        self.act_strength = config.get_config()["simulator"]["act_strength"]
        self.p_vol = config.get_config()["simulator"]["p_vol"]
        self.E = config.get_config()["simulator"]["E"]
        self.mu = config.get_config()["simulator"]["mu"]
        self.la = config.get_config()["simulator"]["la"]
        self.bound = config.get_config()["simulator"]["bound"]
        self.coeff = config.get_config()["simulator"]["coeff"]

        self.n_models = config.get_config()["nn"]["n_models_used"]
        print(f"n models MPM: {self.n_models}")

        self.x = vec(self.dim)
        self.v = vec(self.dim)
        self.center = vec(self.dim)

        # The center of the object to manipulate
        self.object_center = vec(self.dim)
        self.object_particle_num = ti.field(int, shape=())

        self.actuation = scalar()
        # ti.root.dense(ti.ijk, (self.max_steps, self.batch_size, self.n_objects)).place(self.x, self.v)
        batch_node = ti.root.dense(ti.ijk, (self.n_models, self.max_steps, self.batch_size))
        batch_node.dense(ti.l, self.n_objects).place(self.x, self.v)
        batch_node.place(self.center)
        batch_node.place(self.object_center)

        # ti.root.dense(ti.ij, (self.max_steps, self.batch_size)).place(self.center)
        # ti.root.dense(ti.ijk, (self.max_steps, self.batch_size, self.n_springs)).place(self.actuation)
        batch_node.dense(ti.l, self.n_springs).place(self.actuation)

        self.height = scalar()
        self.upper_height = scalar()
        self.rotation = None
        self.actuator_id = ti.field(ti.i32)
        self.particle_type = ti.field(ti.i32)
        self.C, self.F = mat(self.dim), mat(self.dim)
        self.grid_v_in, self.grid_m_in = vec(self.dim), scalar()
        self.grid_v_out = vec(self.dim)

        batch_node.place(self.height, self.upper_height)
        batch_node.dense(ti.l, self.n_particles).place(self.C, self.F)
        ti.root.dense(ti.ijkl, (self.n_models, self.batch_size, self.n_grid, self.n_grid)).place(self.grid_v_in, self.grid_m_in,
                                                                                 self.grid_v_out)
        # These properties are the same for all models
        ti.root.dense(ti.ij, (self.batch_size, self.n_particles)).place(self.actuator_id, self.particle_type)

    def initialize_robot(self):
        for k in range(self.batch_size):
            for i in range(self.n_objects):
                self.actuator_id[k, i] = self.springs[i]
        self.particle_type.fill(1)

        n = ti.static(self.n_objects)
        # Compute the number of particles on the object to manipulate
        for k, i in ti.ndrange(1, n):
            act_id = self.actuator_id[k, i]
            if act_id == -2:
                self.object_particle_num[None] += 1
        print("Object to manipulate particle number", self.object_particle_num[None])

    @ti.kernel
    def clear_states(self, steps: ti.template()):
        for model_id, t, k, i in ti.ndrange(self.n_models, steps, self.batch_size, self.n_particles):
            self.x.grad[model_id, t, k, i] = ti.Matrix.zero(real, self.dim, 1)
            self.v.grad[model_id, t, k, i] = ti.Matrix.zero(real, self.dim, 1)
            self.C[model_id, t, k, i] = ti.Matrix.zero(real, self.dim, self.dim)
            self.C.grad[model_id, t, k, i] = ti.Matrix.zero(real, self.dim, self.dim)
            if ti.static(self.dim == 2):
                self.F[model_id, t, k, i] = [[1., 0.], [0., 1.]]
            else:
                self.F[model_id, t, k, i] = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
            self.F.grad[model_id, t, k, i] = ti.Matrix.zero(real, self.dim, self.dim)

    @ti.kernel
    def clear_grid(self):
        for model_id, k, i, j in ti.ndrange(self.n_models, self.batch_size, self.n_grid, self.n_grid):
            self.grid_v_in[model_id, k, i, j] = ti.Matrix.zero(real, self.dim, 1)
            self.grid_m_in[model_id, k, i, j] = 0
            self.grid_v_out[model_id, k, i, j] = ti.Matrix.zero(real, self.dim, 1)

            self.grid_v_in.grad[model_id, k, i, j] = ti.Matrix.zero(real, self.dim, 1)
            self.grid_m_in.grad[model_id, k, i, j] = 0
            self.grid_v_out.grad[model_id, k, i, j] = ti.Matrix.zero(real, self.dim, 1)

    @ti.kernel
    def clear_grid_interactive(self):
        for model_id, k, i, j in ti.ndrange(self.n_models, self.batch_size, self.n_grid, self.n_grid):
            self.grid_v_in[model_id, k, i, j] = ti.Matrix.zero(real, self.dim, 1)
            self.grid_m_in[model_id, k, i, j] = 0
            self.grid_v_out[model_id, k, i, j] = ti.Matrix.zero(real, self.dim, 1)

    @ti.kernel
    def p2g(self, f: ti.i32):
        for model_id, k, p in ti.ndrange(self.n_models, self.batch_size, self.n_particles):
            anchor = ti.Matrix.zero(real, self.dim, 1)
            anchor[0] = ti.cast(ti.cast(self.x[model_id, f, k, 0](0) * self.inv_dx - 0.5, ti.i32) - 5, real) * self.dx
            base = ti.cast((self.x[model_id, f, k, p] - anchor) * self.inv_dx - 0.5, ti.i32)
            fx = (self.x[model_id, f, k, p] - anchor) * self.inv_dx - ti.cast(base, real)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            new_F = (ti.Matrix.diag(dim=2, val=1) + self.dt * self.C[model_id, f, k, p]) @ self.F[model_id, f, k, p]
            J = (new_F).determinant()
            if self.particle_type[k, p] == 0:  # fluid
                sqrtJ = ti.sqrt(J)
                new_F = ti.Matrix([[sqrtJ, 0], [0, sqrtJ]])

            self.F[model_id, f + 1, k, p] = new_F
            r, s = ti.polar_decompose(new_F)

            act_id = self.actuator_id[k, p]

            act_applied = self.actuation[model_id, f, k, ti.max(0, act_id)] * self.act_strength
            if act_id == -1:
                act_applied = 0.0
            # ti.print(act_applied)

            A = ti.Matrix([[0.0, 0.0], [0.0, 1.0]]) * act_applied
            cauchy = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
            mass = 0.0
            if self.particle_type[k, p] == 0:
                mass = 4
                cauchy = ti.Matrix([[1.0, 0.0], [0.0, 0.1]]) * (J - 1) * self.E
            else:
                mass = 1
                cauchy = 2 * self.mu * (new_F - r) @ new_F.transpose() + \
                         ti.Matrix.diag(2, self.la * (J - 1) * J)
            cauchy += new_F @ A @ new_F.transpose()
            stress = -(self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * cauchy
            affine = stress + mass * self.C[model_id, f, k, p]
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    offset = ti.Vector([i, j])
                    dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * self.dx
                    weight = w[i](0) * w[j](1)
                    self.grid_v_in[model_id, k, base + offset] += weight * (mass * self.v[model_id, f, k, p] + affine @ dpos)
                    self.grid_m_in[model_id, k, base + offset] += weight * mass

    @ti.kernel
    def grid_op(self):
        for model_id, k, i, j in self.grid_m_in:
            inv_m = 1 / (self.grid_m_in[model_id, k, i, j] + 1e-10)
            v_out = inv_m * self.grid_v_in[model_id, k, i, j]
            v_out[1] += self.dt * self.gravity
            if i < self.bound and v_out[0] < 0:
                v_out[0] = 0
                v_out[1] = 0
            if i > self.n_grid - self.bound and v_out[0] > 0:
                v_out[0] = 0
                v_out[1] = 0
            if j < self.bound and v_out[1] < 0:
                v_out[0] = 0
                v_out[1] = 0
                normal = ti.Vector([0.0, 1.0])
                lsq = (normal ** 2).sum()
                if lsq > 0.5:
                    if ti.static(self.coeff < 0):
                        v_out[0] = 0
                        v_out[1] = 0
                    else:
                        lin = (v_out.transpose() @ normal)(0)
                        if lin < 0:
                            vit = v_out - lin * normal
                            lit = vit.norm() + 1e-10
                            if lit + self.coeff * lin <= 0:
                                v_out[0] = 0
                                v_out[1] = 0
                            else:
                                v_out = (1 + self.coeff * lin / lit) * vit
            if j > self.n_grid - self.bound and v_out[1] > 0:
                v_out[0] = 0
                v_out[1] = 0
            self.grid_v_out[model_id, k, i, j] = v_out

    @ti.kernel
    def g2p(self, f: ti.i32):
        for model_id, k, p in ti.ndrange(self.n_models, self.batch_size, self.n_particles):
            anchor = ti.Matrix.zero(real, self.dim, 1)
            anchor[0] = ti.cast(ti.cast(self.x[model_id, f, k, 0](0) * self.inv_dx - 0.5, ti.i32) - 5, real) * self.dx
            base = ti.cast((self.x[model_id, f, k, p] - anchor) * self.inv_dx - 0.5, ti.i32)
            fx = (self.x[model_id, f, k, p] - anchor) * self.inv_dx - ti.cast(base, real)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
            new_v = ti.Vector([0.0, 0.0])
            new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    dpos = ti.cast(ti.Vector([i, j]), real) - fx
                    g_v = self.grid_v_out[model_id, k, base(0) + i, base(1) + j]
                    weight = w[i](0) * w[j](1)
                    new_v += weight * g_v
                    new_C += 4 * weight * g_v.outer_product(dpos) * self.inv_dx

            self.v[model_id, f + 1, k, p] = new_v
            self.x[model_id, f + 1, k, p] = self.x[model_id, f, k, p] + self.dt * self.v[model_id, f + 1, k, p]
            self.C[model_id, f + 1, k, p] = new_C

    @ti.kernel
    def compute_center(self, t: ti.i32):
        n = ti.static(self.n_objects)
        for model_id, k in ti.ndrange(self.n_models, self.batch_size):
            self.center[model_id, t, k] = ti.Matrix.zero(real, self.dim, 1)
            self.object_center[model_id, t, k] = ti.Matrix.zero(real, self.dim, 1)
        for model_id, k, i in ti.ndrange(self.n_models, self.batch_size, n):
            act_id = self.actuator_id[k, i]
            if act_id == -2:
                self.object_center[model_id, t, k] += self.x[model_id, t, k, i] / self.object_particle_num[None]
            else:
                self.center[model_id, t, k] += self.x[model_id, t, k, i] / (n - self.object_particle_num[None])

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

        # for model_id, k in ti.ndrange(self.n_models, self.batch_size):
        #     h = -10.
        #     for i in ti.static(range(self.n_objects)):
        #         h = ti.max(h, self.x[model_id, t, k, i](1))
        #     self.upper_height[model_id, t, k] = h

    @ti.ad.grad_replaced
    def advance_core(self, t: ti.i32):
        self.clear_grid()
        self.p2g(t)
        self.grid_op()
        self.g2p(t)

    @ti.ad.grad_for(advance_core)
    def advance_core_grad(self, t: ti.i32):
        self.clear_grid()
        self.p2g(t)
        self.grid_op()

        self.g2p.grad(t)
        self.grid_op.grad()
        self.p2g.grad(t)

    def pre_advance(self, s):
        self.compute_center(s)
        self.compute_height(s)

    def advance(self, s):
        self.advance_core(s)

    def draw_robot(self, gui, t, batch_rank, target_v, target_position=None):
        def circle(x, y, color, radius=2):
            gui.circle((x, y + 0.1 - self.dx * self.bound), ti.rgb_to_hex(color), radius)
        aid = self.actuator_id.to_numpy()
        for i in range(self.n_objects):
            color = (0.06640625, 0.06640625, 0.06640625)
            if self.simulator == "mpm" and aid[0, i] != -1:
                act_applied = self.actuation[0, t - 1, batch_rank, aid[0, i]]
                color = (0.5 - act_applied, 0.5 - abs(act_applied), 0.5 + act_applied)
            if aid[0, i] == -2:
                color = (1.0, 0.0, 1.0)
            circle(self.x[0, t, batch_rank, i][0], self.x[0, t, batch_rank, i][1], color)

        # Draw the center of the object
        circle(self.object_center[0, t, batch_rank][0], self.object_center[0, t, batch_rank][1], (0.0, 0.0, 1.0), radius=6)
        # Draw the center of the robot
        # circle(self.center[0, t, batch_rank][0], self.center[0, t, batch_rank][1], (1.0, 0.0, 0.0), radius=6)
        if target_position is not None:
            # Draw the center of the target
            circle(target_position[t, batch_rank][0], target_position[t, batch_rank][1], (0.0, 1.0, 0.0), radius=8)
            # circle(0.6, 0.6, (0.0, 1.0, 0.0), radius=8)
        # print("Target position ", target_position[t, batch_rank][0], target_position[t, batch_rank][1])

        # print("robot center", self.center[0, t, batch_rank][0], self.center[0, t, batch_rank][1])
        # print("object center", self.object_center[0, t, batch_rank][0], self.object_center[0, t, batch_rank][1])
