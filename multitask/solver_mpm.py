import taichi as ti
from config import *
from utils import *

@ti.data_oriented
class SolverMPM:
    def __init__(self):
        self.x = vec()
        self.v = vec()
        self.center = vec()
        self.actuation = scalar()
        ti.root.dense(ti.ijk, (max_steps, batch_size, n_objects)).place(self.x, self.v)
        ti.root.dense(ti.ij, (max_steps, batch_size)).place(self.center)
        ti.root.dense(ti.ijk, (max_steps, batch_size, n_springs)).place(self.actuation)

        self.height = None
        self.rotation = None
        self.actuator_id = ti.field(ti.i32)
        self.particle_type = ti.field(ti.i32)
        self.C, self.F = mat(), mat()
        self.grid_v_in, self.grid_m_in = vec(), scalar()
        self.grid_v_out = vec()
        ti.root.dense(ti.ij, (batch_size, n_particles)).place(self.actuator_id, self.particle_type)
        ti.root.dense(ti.ijk, (max_steps, batch_size, n_particles)).place(self.C, self.F)
        ti.root.dense(ti.ijk, (batch_size, n_grid, n_grid)).place(self.grid_v_in, self.grid_m_in, self.grid_v_out)

    def initialize_robot(self):
        for k in range(batch_size):
            for i in range(n_objects):
                self.actuator_id[k, i] = springs[i]
        self.particle_type.fill(1)

    @ti.kernel
    def clear_states(self, steps: ti.template()):
        for t, k, i in ti.ndrange(steps, batch_size, n_particles):
            self.x.grad[t, k, i] = ti.Matrix.zero(real, dim, 1)
            self.v.grad[t, k, i] = ti.Matrix.zero(real, dim, 1)
            self.C[t, k, i] = ti.Matrix.zero(real, dim, dim)
            self.C.grad[t, k, i] = ti.Matrix.zero(real, dim, dim)
            if ti.static(dim == 2):
                self.F[t, k, i] = [[1., 0.], [0., 1.]]
            else:
                self.F[t, k, i] = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
            self.F.grad[t, k, i] = ti.Matrix.zero(real, dim, dim)

    @ti.kernel
    def clear_grid(self):
        for k, i, j in ti.ndrange(batch_size, n_grid, n_grid):
            self.grid_v_in[k, i, j] = ti.Matrix.zero(real, dim, 1)
            self.grid_m_in[k, i, j] = 0
            self.grid_v_out[k, i, j] = ti.Matrix.zero(real, dim, 1)
            self.grid_v_in.grad[k, i, j] = ti.Matrix.zero(real, dim, 1)
            self.grid_m_in.grad[k, i, j] = 0
            self.grid_v_out.grad[k, i, j] = ti.Matrix.zero(real, dim, 1)

    @ti.kernel
    def p2g(self, f: ti.i32):
        for k, p in ti.ndrange(batch_size, n_particles):
            base = ti.cast(self.x[f, k, p] * inv_dx - 0.5, ti.i32)
            fx = self.x[f, k, p] * inv_dx - ti.cast(base, ti.i32)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            new_F = (ti.Matrix.diag(dim=2, val=1) + dt * self.C[f, k, p]) @ self.F[f, k, p]
            J = (new_F).determinant()
            if self.particle_type[k, p] == 0:  # fluid
                sqrtJ = ti.sqrt(J)
                new_F = ti.Matrix([[sqrtJ, 0], [0, sqrtJ]])

            self.F[f + 1, k, p] = new_F
            r, s = ti.polar_decompose(new_F)

            act_id = self.actuator_id[k, p]

            act_applied = self.actuation[f, k, ti.max(0, act_id)] * act_strength
            if act_id == -1:
                act_applied = 0.0
            # ti.print(act_applied)

            A = ti.Matrix([[0.0, 0.0], [0.0, 1.0]]) * act_applied
            cauchy = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
            mass = 0.0
            if self.particle_type[k, p] == 0:
                mass = 4
                cauchy = ti.Matrix([[1.0, 0.0], [0.0, 0.1]]) * (J - 1) * E
            else:
                mass = 1
                cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + \
                         ti.Matrix.diag(2, la * (J - 1) * J)
            cauchy += new_F @ A @ new_F.transpose()
            stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
            affine = stress + mass * self.C[f, k, p]
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    offset = ti.Vector([i, j])
                    dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
                    weight = w[i](0) * w[j](1)
                    self.grid_v_in[k, base + offset] += weight * (mass * self.v[f, k, p] + affine @ dpos)
                    self.grid_m_in[k, base + offset] += weight * mass

    @ti.kernel
    def grid_op(self):
        for k, i, j in self.grid_m_in:
            inv_m = 1 / (self.grid_m_in[k, i, j] + 1e-10)
            v_out = inv_m * self.grid_v_in[k, i, j]
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
            self.grid_v_out[k, i, j] = v_out

    @ti.kernel
    def g2p(self, f: ti.i32):
        for k, p in ti.ndrange(batch_size, n_particles):
            base = ti.cast(self.x[f, k, p] * inv_dx - 0.5, ti.i32)
            fx = self.x[f, k, p] * inv_dx - ti.cast(base, real)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
            new_v = ti.Vector([0.0, 0.0])
            new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    dpos = ti.cast(ti.Vector([i, j]), real) - fx
                    g_v = self.grid_v_out[k, base(0) + i, base(1) + j]
                    weight = w[i](0) * w[j](1)
                    new_v += weight * g_v
                    new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

            self.v[f + 1, k, p] = new_v
            self.x[f + 1, k, p] = self.x[f, k, p] + dt * self.v[f + 1, k, p]
            self.C[f + 1, k, p] = new_C

    @ti.kernel
    def compute_center(self, t: ti.i32):
        n = ti.static(n_objects)
        for k in range(batch_size):
            self.center[t, k] = ti.Matrix.zero(real, dim, 1)
        for k, i in ti.ndrange(batch_size, n):
            self.center[t, k] += self.x[t, k, i] / n

    @ti.complex_kernel
    def advance_core(self, t: ti.i32):
        self.clear_grid()
        self.p2g(t)
        self.grid_op()
        self.g2p(t)

    @ti.complex_kernel_grad(advance_core)
    def advance_core_grad(self, t: ti.i32):
        self.clear_grid()
        self.p2g(t)
        self.grid_op()

        self.g2p.grad(t)
        self.grid_op.grad()
        self.p2g.grad(t)

    def advance(self, s):
        self.compute_center(s)
        self.advance_core(s)

    def draw_robot(self, gui, t, target_v):
        def circle(x, y, color):
            gui.circle((x, y + 0.1 - dx * bound), ti.rgb_to_hex(color), 2)
        aid = self.actuator_id.to_numpy()
        for i in range(n_objects):
            color = (0.06640625, 0.06640625, 0.06640625)
            if simulator == "mpm" and aid[0, i] != -1:
                act_applied = self.actuation[t - 1, 0, aid[0, i]]
                color = (0.5 - act_applied, 0.5 - abs(act_applied), 0.5 + act_applied)
            circle(self.x[t, 0, i][0], self.x[t, 0, i][1], color)
        if target_v[t, 0][0] > 0:
            circle(0.5, 0.5, (1, 0, 0))
            circle(0.6, 0.5, (1, 0, 0))
        else:
            circle(0.5, 0.5, (0, 0, 1))
            circle(0.4, 0.5, (0, 0, 1))
