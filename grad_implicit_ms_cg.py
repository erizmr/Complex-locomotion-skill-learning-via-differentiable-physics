# https://www.cs.cmu.edu/~baraff/papers/sig98.pdf
import argparse

import numpy as np

import taichi as ti

from multitask.robot_design import RobotDesignMassSpring3D
from utils import data_type

@ti.data_oriented
class ImplictMassSpringSolver:
    def __init__(self, robot_builder: RobotDesignMassSpring3D, data_type=data_type, batch=1, substeps=1, dt=0.01, dim=3):
        self.dim = dim
        self.data_type = data_type
        self.dt = dt
        self.batch = batch
        self.substeps = substeps
        self.substeps_offset = substeps + 1
        print(f"Batch: {self.batch}, Substep: {self.substeps}, dt: {self.dt}, data type: {self.data_type}")
        _vertices, _springs_data, _faces = robot_builder.get_objects()
        self.vertices = np.array(_vertices) # [NV, 3]
        self.springs_data = np.array(_springs_data) # [NE, (a, b, length, stiffness, actuation)]
        self.faces = np.array(_faces) # [NF, 3]
        print(f"Vertices {self.vertices.shape}, Springs {self.springs_data.shape}, Faces {self.faces.shape}, Time step: {self.dt}")
        self.NF = self.faces.shape[0]  # number of faces
        self.NV = self.vertices.shape[0]  # number of vertices
        self.NE = self.springs_data.shape[0]  # number of edges

        self.initPos = ti.Vector.field(self.dim, self.data_type, self.NV)

        # [batch, substeps, NV, dim]
        self.pos = ti.Vector.field(self.dim, self.data_type, shape=(self.batch, self.substeps_offset, self.NV), needs_grad=True)
        self.vel = ti.Vector.field(self.dim, self.data_type, shape=(self.batch, self.substeps_offset, self.NV), needs_grad=True)
        self.dv = ti.Vector.field(self.dim, self.data_type, shape=(self.batch, self.substeps_offset, self.NV), needs_grad=True)
        self.force = ti.Vector.field(self.dim, self.data_type, shape=(self.batch, self.substeps_offset, self.NV), needs_grad=True)
        self.mass = ti.field(self.data_type, self.NV)

        # [batch, NV, dim]
        self.dv_one_step = ti.Vector.field(self.dim, self.data_type, shape=(self.batch, self.NV), needs_grad=True)

        # [batch, NE, dim]
        self.Jx = ti.Matrix.field(self.dim, self.dim, self.data_type,
                                  shape=(self.batch, self.NE))  # Jacobian with respect to position
        self.Jv = ti.Matrix.field(self.dim, self.dim, self.data_type,
                                  shape=(self.batch, self.NE))  # Jacobian with respect to velocity

        # [batch, NV, dim]
        self.b = ti.Vector.field(self.dim, self.data_type, shape=(self.batch, self.NV), needs_grad=True)
        self.Adv = ti.Vector.field(self.dim, self.data_type, shape=(self.batch, self.NV))
        self.r0 = ti.Vector.field(self.dim, self.data_type, shape=(self.batch, self.NV))
        self.r1 = ti.Vector.field(self.dim, self.data_type, shape=(self.batch, self.NV))
        self.p0 = ti.Vector.field(self.dim, self.data_type, shape=(self.batch, self.NV))
        self.p1 = ti.Vector.field(self.dim, self.data_type, shape=(self.batch, self.NV))

        # [batch] Result buffer for CG
        self.dot_ans = ti.field(dtype=self.data_type, shape=self.batch)
        self.r2 = ti.field(dtype=self.data_type, shape=self.batch)
        self.r2_init = ti.field(dtype=self.data_type, shape=self.batch)
        self.r2_new = ti.field(dtype=self.data_type, shape=self.batch)
        self.alpha = ti.field(dtype=self.data_type, shape=self.batch)
        self.beta = ti.field(dtype=self.data_type, shape=self.batch)
        
        # [batch, NE] Keep the actuation for all substeps
        self.actuation = ti.field(self.data_type, shape=(self.batch, self.NE), needs_grad=True)

        # [NV, dim]
        self.spring = ti.Vector.field(2, ti.i32, self.NE)
        self.spring_actuation_coef = ti.field(self.data_type, self.NE)
        self.rest_len = ti.field(self.data_type, self.NE)
        self.ks = 1e5  # spring stiffness

        self.gravity = ti.Vector([0.0, -2.0, 0.0])
        self.ground_height = 0.1
        self.init_pos()
        self.init_edges()

    def init_pos(self):
        self.initPos.from_numpy(np.array(self.vertices))
        pos_arr = np.expand_dims(np.array(self.vertices), axis=(0, 1))
        self.pos.from_numpy(np.tile(pos_arr, (self.batch, self.substeps_offset, 1, 1)))
        self.vel.from_numpy(np.zeros((self.batch, self.substeps_offset, self.NV, self.dim))) # assume initial velocity is 0
        self.mass.from_numpy(np.ones(self.NV)) # assume mass is 1 for all vertices
        self.dv.fill(0.0)
    
    def init_edges(self):
        self.spring.from_numpy(self.springs_data[:, :2])
        print("spring index len ", self.spring.shape)
        self.rest_len.from_numpy(self.springs_data[:, 2])
        self.spring_actuation_coef.from_numpy(self.springs_data[:, 4])
        print("spring_actuation_coef ", self.spring_actuation_coef)

    @ti.func
    def clear_force(self, step: ti.i32):
        for bs, i in ti.ndrange(self.batch, self.NV):
            self.force[bs, step, i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def compute_force(self, step: ti.i32):
        self.clear_force(step)
        for bs, i in ti.ndrange(self.batch, self.NV):
            self.force[bs, step, i] += self.gravity * self.mass[i]

        for bs, i in ti.ndrange(self.batch, self.NE):
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            pos1, pos2 = self.pos[bs, step, idx1], self.pos[bs, step, idx2]
            dis = pos1 - pos2

            target_length = self.rest_len[i] * (1.0 + self.spring_actuation_coef[i] * self.actuation[bs, i])
            force = -self.ks * (dis.norm() -
                               target_length) * dis.normalized()
            self.force[bs, step, idx1] += force
            self.force[bs, step, idx2] -= force

    @ti.kernel
    def _matrix_vector_product(self, vec: ti.template()):
        for bs, i in ti.ndrange(self.batch, self.NE):
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            val = self.Jx[bs, i]@(vec[bs, idx1] - vec[bs, idx2])
            self.Adv[bs, idx1] -= -val * self.dt**2
            self.Adv[bs, idx2] -= val * self.dt**2

    
    @ti.kernel
    def compute_jacobian(self, step: ti.i32):
        for bs, i in ti.ndrange(self.batch, self.NE):
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            pos1, pos2 = self.pos[bs, step, idx1], self.pos[bs, step, idx2]
            dx = pos1 - pos2
            I = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            dxtdx = ti.Matrix([[dx[0] * dx[0], dx[0] * dx[1], dx[0]*dx[2]],
                               [dx[1] * dx[0], dx[1] * dx[1], dx[1]*dx[2]],
                               [dx[2] * dx[0], dx[1] * dx[2], dx[2]*dx[2]]])
            l = dx.norm()
            l_inv = self.data_type(0.0)
            if l != 0.0:
                l_inv = self.data_type(1.0) / l
            # Clamp the potential negative part to make the hessian positive definite
            self.Jx[bs, i] = (ti.max(1 - self.rest_len[i] * l_inv, 0) * I + self.rest_len[i] * dxtdx * l_inv**3) * self.ks


    @ti.kernel
    def add_mass(self, vec: ti.template()):
        for bs, i in ti.ndrange(self.batch, self.NV):
            self.Adv[bs, i] = self.mass[i] * vec[bs, i]


    def matrix_vector_product(self, vec):
        self.add_mass(vec)
        self._matrix_vector_product(vec)


    @ti.kernel
    def advect(self, step: ti.i32):
        for bs, i in ti.ndrange(self.batch, self.NV):
            old_x = self.pos[bs, step, i]
            old_v = self.vel[bs, step, i] + self.dv[bs, step, i]
            new_x = self.dt * old_v + old_x
            toi = self.data_type(0.0)
            new_v = old_v
            if new_x[1] < self.ground_height and old_v[1] < -1e-4:
                toi = float(-(old_x[1] - self.ground_height) / old_v[1])
                # Inf friction
                new_v = ti.Matrix.zero(self.data_type, self.dim)
                # Reasonable friction
                new_v[1] = self.data_type(0.0)
                friction = self.data_type(0.4)
                if old_v[0] < 0:
                    new_v[0] = ti.min(0., old_v[0] + friction * (-old_v[1]))
                else:
                    new_v[0] = ti.max(0., old_v[0] - friction * (-old_v[1]))
                if old_v[2] < 0:
                    new_v[2] = ti.min(0., old_v[2] + friction * (-old_v[1]))
                else:
                    new_v[2] = ti.max(0., old_v[2] - friction * (-old_v[1]))
            new_x = old_x + toi * old_v + (self.dt - toi) * new_v
            self.vel[bs, step+1, i] = new_v
            self.pos[bs, step+1, i] = new_x
    

    @ti.kernel
    def apply_external_force(self, step: ti.int32):
        for bs, i in ti.ndrange(self.batch, self.NV):
            self.b[bs, i] = self.force[bs, step, i] * self.dt


    @ti.kernel
    def apply_hessian_vel(self, step: ti.int32):
        for bs, i in ti.ndrange(self.batch, self.NE):
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            val = self.Jx[bs, i]@(self.vel[bs, step, idx1] - self.vel[bs, step, idx2])
            self.b[bs, idx1] += -val* self.dt**2
            self.b[bs, idx2] += val * self.dt**2


    def compute_b(self, step: int):
        self.apply_external_force(step)
        self.apply_hessian_vel(step)


    @ti.kernel
    def add(self, ans: ti.template(), a: ti.template(), k: ti.f64, b: ti.template()):
        for i in ti.grouped(ans):
            ans[i] = a[i] + k * b[i]


    @ti.kernel
    def dot(self, a: ti.template(), b: ti.template()) -> ti.f64:
        ans = self.data_type(0.0)
        for i in a:
            ans += a[i].dot(b[i])
        return ans


    @ti.kernel
    def add_batched(self, ans: ti.template(), a: ti.template(), k: ti.template(), b: ti.template()):
        for bs, i in ti.ndrange(self.batch, self.NV):
            ans[bs, i] = a[bs, i] + k[bs] * b[bs, i]
    

    @ti.kernel
    def substract_batched(self, ans: ti.template(), a: ti.template(), k: ti.template(), b: ti.template()):
        for bs, i in ti.ndrange(self.batch, self.NV):
            ans[bs, i] = a[bs, i] - k[bs] * b[bs, i]


    @ti.kernel
    def dot_batched(self, ans: ti.template(), a: ti.template(), b: ti.template()):
        for bs in range(self.batch):
            ans[bs] = self.data_type(0.0)
        for bs, i in ti.ndrange(self.batch, self.NV):
            ans[bs] += a[bs, i].dot(b[bs, i])
    

    @ti.kernel
    def divide_batched(self, ans: ti.template(), a: ti.template(), b: ti.template()):
        for bs in range(self.batch):
            ans[bs] = a[bs] / (b[bs] + 1e-6)


    @ti.kernel
    def copy(self, dst: ti.template(), src: ti.template()):
        for i in ti.grouped(dst):
            dst[i] = src[i]


    @ti.kernel
    def copy_slice(self, dst: ti.template(), src: ti.template(), step: ti.i32):
        for bs, i in ti.ndrange(self.batch, self.NV):
            dst[bs, i] = src[bs, step, i]
    

    @ti.kernel
    def copy_slice_back(self, dst: ti.template(), src: ti.template(), step: ti.i32):
        for bs, i in ti.ndrange(self.batch, self.NV):
            dst[bs, step, i] = src[bs, i]


    @ti.ad.grad_replaced
    def update(self, step: int):
        self.compute_force(step)
        self.compute_jacobian(step)
        # b = (force + h * K @ vel) * h
        self.compute_b(step)

        # Get only one step slice of dv for solving
        # self.copy_slice(self.dv_one_step, self.dv, step)
        self.dv_one_step.fill(0.0)
        # Solve the linear system for dv
        self.cg_solver(self.dv_one_step)
        # Update dv using the solved result
        self.copy_slice_back(self.dv, self.dv_one_step, step)

        self.advect(step)


    @ti.ad.grad_for(update)
    def update_grad(self, step: int):
        self.advect.grad(step)
        # print("b grad before", self.b.grad)
        # Get the corresponding step slice of dv for solving
        self.copy_slice(self.dv_one_step, self.dv, step)
        self.cg_solver_grad(self.dv_one_step)
        self.apply_external_force.grad(step)
        # print("b grad up: ", self.b.grad[5], "force grad ", self.force.grad[5])
        # print("actuation before", self.actuation.grad)
        self.compute_force.grad(step)
        # print("actuation ", self.actuation.grad)


    def cg_solver(self, x):
        # x is a taichi vector field

        # print(" =============== ")
        # print("jx 20 ", self.Jx[20].to_numpy())
        # print("jx sum ", np.sqrt((self.Jx.to_numpy())**2).sum())
        self.matrix_vector_product(x) # Adv = A @ x
        # print("adv before solve ", self.Adv.to_numpy().sum())
        # print("f ", self.force.to_numpy())
        # print("b before solve ", self.b.to_numpy().sum())
        # print("b sum ", self.b.to_numpy().sum())
        self.add(self.r0, self.b, -1.0, self.Adv) # r0 = b - Adv
        self.copy(self.p0, self.r0) # p0 = r0
        self.dot_batched(self.r2, self.r0, self.r0) # r2 = r0 @ r0
        r2_init = self.r2.to_numpy() # r2_init = r2
        self.copy(self.r2_new, self.r2) # r2_new = r2
        n_iter = 24 # 24 CG iterations can achieve 1e-3 accuray gradient
        epsilon = 1e-6
        for i in range(n_iter):
            # if (i+1) % n_iter == 0:
            #     print(f"Iteration: {i} Residual: {self.r2_new.to_numpy().sum()} thresold: {epsilon * r2_init.sum()}")

            self.matrix_vector_product(self.p0) # Adv = A @ p0
            self.dot_batched(self.alpha, self.p0, self.Adv) # inv_alpha = p0 @ Adv
            self.divide_batched(self.alpha, self.r2, self.alpha) # alpha = r2 / p0 @ Adv
            # print(f"Iteration: {i} r_2; {self.r2.to_numpy().sum()} alpha: {self.alpha}, p0: {self.p0.to_numpy().sum()}, adv: {self.Adv.to_numpy().sum()}")
            self.add_batched(x, x, self.alpha, self.p0) # x = x + alpha * p0
            self.substract_batched(self.r1, self.r0, self.alpha, self.Adv) # r1 = r0 - alpha * Adv
            self.dot_batched(self.r2_new, self.r1, self.r1) # r2_new = r1 @ r1

            if self.r2_new.to_numpy().sum() < epsilon * r2_init.sum():
                break

            self.divide_batched(self.beta, self.r2_new, self.r2) # beta = r2_new / r2
            self.add_batched(self.p1, self.r1, self.beta, self.p0) # p1 = r1 + beta * p0
            self.copy(self.r0, self.r1)
            self.copy(self.p0, self.p1)
            self.copy(self.r2, self.r2_new)

        # print("adv after solve ", self.Adv.to_numpy().sum())
        # print("p0 after solve ", self.p0.to_numpy().sum())
        # print("b after solve ", self.b.to_numpy().sum())
        # print("b grad", self.b.grad.to_numpy().sum())


    def cg_solver_grad(self, x):
        # Solve the adjoint of b
        # A * adj_b = adj_x
        self.copy(self.b, x.grad)
        # Prevent divide by zero in gradcheck
        self.b.grad.fill(1e-6)
        self.cg_solver(self.b.grad)


    @ti.kernel
    def zero_grad(self):
        self.pos.grad.fill(0.0)
        self.vel.grad.fill(0.0)
        self.dv.grad.fill(0.0)
        self.force.grad.fill(0.0)
        self.b.grad.fill(0.0)
        self.actuation.grad.fill(0.0)
    
    @ti.kernel
    def copy_states(self):
        for bs, i in ti.ndrange(self.batch, self.NV):
            self.pos[bs, 0, i] = self.pos[bs, self.substeps, i]
            self.vel[bs, 0, i] = self.vel[bs, self.substeps, i]


def main():
    parser = argparse.ArgumentParser("implicit mass spring ")
    parser.add_argument('-g',
                        '--use-ggui',
                        action='store_true',
                        help='Display with GGUI')
    parser.add_argument('-a',
                        '--arch',
                        required=False,
                        default="cpu",
                        dest='arch',
                        type=str,
                        help='The arch (backend) to run this example on')
    
    parser.add_argument('--robot_design_file',
                        default='',
                        help='robot design file')
    args = parser.parse_args()
    # args, unknowns = parser.parse_known_args()
    arch = args.arch
    if arch in ["x64", "cpu", "arm64"]:
        ti.init(arch=ti.cpu, debug=True)
    elif arch in ["cuda", "gpu"]:
        ti.init(arch=ti.cuda)
    else:
        raise ValueError('Only CPU and CUDA backends are supported for now.')
    
    file_name = args.robot_design_file.split('/')[-1].split('.')[0]
    robot_design_file = args.robot_design_file
    robot_builder = RobotDesignMassSpring3D.from_file(robot_design_file)
    robot_id = robot_builder.robot_id
    vertices, springs, faces = robot_builder.build()
    # robot_builder.draw()

    h = 0.01
    BATCH_SIZE = 8
    VIS_BATCH = 0
    SUBSTEPS = 10
    pause = False
    ms_solver = ImplictMassSpringSolver(robot_builder, batch=BATCH_SIZE, substeps=SUBSTEPS)

    window = ti.ui.Window('Implicit Mass Spring System', res=(800, 800), vsync=True, show_window=True)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.make_camera()
    camera.position(0.2, 1.1, 1.1)
    camera.lookat(0.2, 0.1, 0.1)
    camera.up(0, 1, 0)

    objects, springs, faces = robot_builder.get_objects()
    indices = ti.field(ti.i32, len(faces) * 3)
    vertices = ti.Vector.field(3, ti.f32, len(objects))
    indices_ground = ti.field(ti.i32, 6)
    vertices_ground = ti.Vector.field(3, ti.f32, 4)

    vertices.from_numpy(np.array(objects))
    vertices_ground[0] = ti.Vector([-1, 0.1, -1])
    vertices_ground[1] = ti.Vector([-1, 0.1, 1])
    vertices_ground[2] = ti.Vector([1, 0.1, -1])
    vertices_ground[3] = ti.Vector([1, 0.1, 1])

    indices.from_numpy(np.array(faces).reshape(-1))
    indices_ground.from_numpy(np.array([0, 1, 2, 2, 1, 3]))

    actuation_mask = np.where(np.array(springs)[:, 4] > 0)[0]
    actuator_pos_index = np.unique(np.array(springs)[actuation_mask,:2].flatten()).astype(int)
    actuator_pos = ti.Vector.field(3, ti.f32, len(actuator_pos_index))
    print(actuator_pos_index)

    pos_vis_buffer = ti.Vector.field(3, ti.f32, shape=ms_solver.NV)

    while window.running:

        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.ESCAPE:
                break
        if window.is_pressed(ti.ui.SPACE):
            pause = not pause

        if not pause:
            # Apply a random actutation for test
            ms_solver.actuation.from_numpy(np.random.rand(ms_solver.batch, len(springs)) * 0.5)
            for i in range(ms_solver.substeps):
                ms_solver.update(i)
            ms_solver.copy_states()

        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        scene.point_light(pos=(0, 1, 0), color=(.7, .7, .7))
        scene.point_light(pos=(-1, 1, 0), color=(.7, .7, .7))
        scene.ambient_light((0.2, 0.2, 0.2))

        pos_vis = ms_solver.pos.to_numpy()
        pos_vis_buffer.from_numpy(pos_vis[VIS_BATCH, -1, :])


        actuation = pos_vis[VIS_BATCH, -1, :][actuator_pos_index]
        actuator_pos.from_numpy(actuation)

        scene.particles(actuator_pos, radius=0.005, color=(0.0, 0.0, 0.5))
        scene.mesh(pos_vis_buffer, indices=indices, color=(0.8, 0.6, 0.2))
        scene.mesh(vertices_ground, indices=indices_ground, color=(0.5, 0.5, 0.5), two_sided=True)
        canvas.scene(scene)
        window.show()

if __name__ == '__main__':
    main()
