# https://www.cs.cmu.edu/~baraff/papers/sig98.pdf
import argparse

import numpy as np

import taichi as ti

from multitask.robot_design import RobotDesignMassSpring3D
from utils import data_type

@ti.data_oriented
class ImplictMassSpringSolver:
    def __init__(self, robot_builder: RobotDesignMassSpring3D, data_type=data_type, dim=3):
        self.dim = dim
        self.data_type = data_type
        _vertices, _springs_data, _faces = robot_builder.get_objects()
        self.vertices = np.array(_vertices) # [NV, 3]
        self.springs_data = np.array(_springs_data) # [NE, (a, b, length, stiffness, actuation)]
        self.faces = np.array(_faces) # [NF, 3]
        print(f"Vertices {self.vertices.shape}, Springs {self.springs_data.shape}, Faces {self.faces.shape}")
        self.NF = self.faces.shape[0]  # number of faces
        self.NV = self.vertices.shape[0]  # number of vertices
        self.NE = self.springs_data.shape[0]  # number of edges

        self.pos = ti.Vector.field(self.dim, self.data_type, self.NV, needs_grad=True)
        self.initPos = ti.Vector.field(self.dim, self.data_type, self.NV)
        self.vel = ti.Vector.field(self.dim, self.data_type, self.NV, needs_grad=True)
        self.dv = ti.Vector.field(self.dim, self.data_type, self.NV, needs_grad=True)
        self.Adv = ti.Vector.field(self.dim, self.data_type, self.NV)

        self.force = ti.Vector.field(self.dim, self.data_type, self.NV, needs_grad=True)
        self.mass = ti.field(self.data_type, self.NV)
        self.b = ti.Vector.field(self.dim, self.data_type, self.NV, needs_grad=True)
        self.r0 = ti.Vector.field(self.dim, self.data_type, self.NV)
        self.r1 = ti.Vector.field(self.dim, self.data_type, self.NV)
        self.p0 = ti.Vector.field(self.dim, self.data_type, self.NV)
        self.p1 = ti.Vector.field(self.dim, self.data_type, self.NV)

        self.spring = ti.Vector.field(2, ti.i32, self.NE)
        self.spring_actuation_coef = ti.field(self.data_type, self.NE)
        self.actuation = ti.field(self.data_type, self.NE, needs_grad=True) # parameters to optimize
        self.spring_color = ti.Vector.field(3, self.data_type, self.NE)
        self.indices = ti.field(ti.i32, 2 * self.NE)
        self.spring_color_draw = ti.Vector.field(3, self.data_type, 2 * self.NE)
        self.Jx = ti.Matrix.field(self.dim, self.dim, self.data_type,
                                  self.NE)  # Jacobian with respect to position
        self.Jv = ti.Matrix.field(self.dim, self.dim, self.data_type,
                                  self.NE)  # Jacobian with respect to velocity
        self.rest_len = ti.field(self.data_type, self.NE)
        self.ks = 1e5  # spring stiffness
        # self.kd = 0.5  # damping constant

        self.gravity = ti.Vector([0.0, -2.0, 0.0])
        self.ground_height = 0.1
        self.center = ti.Vector.field(self.dim, self.data_type, shape=(), needs_grad=True)
        self.loss = ti.field(self.data_type, shape=(), needs_grad=True)
        self.init_pos()
        self.init_edges()

    def init_pos(self):
        self.pos.from_numpy(np.array(self.vertices))
        self.initPos.from_numpy(np.array(self.vertices))
        self.vel.from_numpy(np.zeros((self.NV, self.dim)))
        self.mass.from_numpy(np.ones(self.NV))
    
    def init_edges(self):
        self.spring.from_numpy(self.springs_data[:, :2])
        print("spring index len ", self.spring.shape)
        self.rest_len.from_numpy(self.springs_data[:, 2])
        self.spring_actuation_coef.from_numpy(self.springs_data[:, 4])
        print("spring_actuation_coef ", self.spring_actuation_coef)

    @ti.func
    def clear_force(self):
        for i in self.force:
            self.force[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def compute_force(self):
        self.clear_force()
        for i in self.force:
            self.force[i] += self.gravity * self.mass[i]

        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            pos1, pos2 = self.pos[idx1], self.pos[idx2]
            dis = pos1 - pos2

            # self.actuation[i] = ti.random()
            # print(self.actuation[i])
            # self.actuation[i] = ti.sin(i * 3.1415926 / 6)
            # self.actuation[i] = -1.0
            target_length = self.rest_len[i] * (1.0 + self.spring_actuation_coef[i] * self.actuation[i])
            force = -self.ks * (dis.norm() -
                               target_length) * dis.normalized()
            self.force[idx1] += force
            self.force[idx2] -= force

    @ti.kernel
    def _matrix_vector_product(self, h: ti.f64, vec: ti.template()):
        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            val = self.Jx[i]@(vec[idx1] - vec[idx2])
            self.Adv[idx1] -= -val * h**2
            self.Adv[idx2] -= val * h**2

    
    @ti.kernel
    def compute_jacobian(self):
        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            pos1, pos2 = self.pos[idx1], self.pos[idx2]
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
            self.Jx[i] = (ti.max(1 - self.rest_len[i] * l_inv, 0) * I + self.rest_len[i] * dxtdx * l_inv**3) * self.ks


    @ti.kernel
    def add_mass(self, vec: ti.template()):
        for i in self.Adv:
            self.Adv[i] = self.mass[i] * vec[i]


    def matrix_vector_product(self, h, vec):
        self.add_mass(vec)
        self._matrix_vector_product(h, vec)


    @ti.kernel
    def advect(self, h: ti.f64):
        for i in self.pos:
            old_x = self.pos[i]
            old_v = self.vel[i] + self.dv[i]
            new_x = h * old_v + self.pos[i]
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
            new_x = old_x + toi * old_v + (h - toi) * new_v
            self.vel[i] = new_v
            self.pos[i] = new_x
    

    @ti.kernel
    def apply_external_force(self, h: ti.f64):
        for i in self.b:
            self.b[i] = self.force[i] * h


    @ti.kernel
    def apply_hessian_vel(self, h: ti.f64):
        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            val = self.Jx[i]@(self.vel[idx1] - self.vel[idx2])
            self.b[idx1] += -val* h**2
            self.b[idx2] += val * h**2


    def compute_b(self, h):
        self.apply_external_force(h)
        self.apply_hessian_vel(h)
    

    @ti.kernel
    def compute_center(self):
        for i in self.pos:
            self.center[None] += self.pos[i] / self.NV

    @ti.kernel
    def compute_loss(self):
        for i in self.pos:
            self.loss[None] += (ti.Vector([1.0, 0.0, 0.0]) - self.center[None]).norm_sqr()   

    @ti.kernel
    def add(self, ans: ti.template(), a: ti.template(), k: ti.f64, b: ti.template()):
        for i in ans:
            ans[i] = a[i] + k * b[i]


    @ti.kernel
    def dot(self, a: ti.template(), b: ti.template()) -> ti.f64:
        ans = self.data_type(0.0)
        for i in a:
            ans += a[i].dot(b[i])
        return ans
    
    @ti.kernel
    def copy(self, dst: ti.template(), src: ti.template()):
        for i in dst:
            dst[i] = src[i]

    @ti.ad.grad_replaced
    def update(self, h):
        self.compute_force()
        self.compute_jacobian()
        # b = (force + h * K @ vel) * h
        self.compute_b(h)
        # Solve the linear system for dv
        self.cg_solver(self.dv, h)
        self.advect(h)
    
    @ti.ad.grad_for(update)
    def update_grad(self, h):
        self.advect.grad(h)
        # print("b grad before", self.b.grad)
        self.cg_solver_grad(self.dv, h)
        self.apply_external_force.grad(h)
        # print("b grad up: ", self.b.grad[5], "force grad ", self.force.grad[5])
        # print("actuation before", self.actuation.grad)
        self.compute_force.grad()
        # print("actuation ", self.actuation.grad)


    def cg_solver(self, x, h):
        # print(" =============== ")
        # print("jx 20 ", self.Jx[20].to_numpy())
        # print("jx sum ", np.sqrt((self.Jx.to_numpy())**2).sum())
        self.matrix_vector_product(h, x)
        # print("f ", self.force.to_numpy())
        # print("b ", self.b.to_numpy().flatten())
        # print("b sum ", self.b.to_numpy().sum())
        self.add(self.r0, self.b, -1.0, self.Adv)
        self.copy(self.p0, self.r0)
        r_2 = self.dot(self.r0, self.r0)
        r_2_init = r_2
        r_2_new = r_2
        n_iter = 10
        epsilon = 1e-6
        for i in range(n_iter):
            # if (i+1) % n_iter == 0:
            #     print(f"Iteration: {i} Residual: {r_2_new} thresold: {epsilon * r_2_init}")
            self.matrix_vector_product(h, self.p0)
            alpha = r_2 / self.dot(self.p0, self.Adv)
            self.add(x, x, alpha, self.p0)
            self.add(self.r1, self.r0, -alpha, self.Adv)
            r_2_new = self.dot(self.r1, self.r1)
            if r_2_new < epsilon * r_2_init:
                break
            beta = r_2_new / r_2
            self.add(self.p1, self.r1, beta, self.p0)
            self.copy(self.r0, self.r1)
            self.copy(self.p0, self.p1)
            r_2 = r_2_new


    def cg_solver_grad(self, x, h):
        # Solve the adjoint of b
        # A * adj_b = adj_x
        self.copy(self.b, x.grad)
        self.cg_solver(self.b.grad, h)


    @ti.kernel
    def zero_grad(self):
        self.pos.grad.fill(0.0)
        self.vel.grad.fill(0.0)
        self.dv.grad.fill(0.0)
        self.force.grad.fill(0.0)
        self.b.grad.fill(0.0)
        self.actuation.grad.fill(0.0)


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
        ti.init(arch=ti.cpu)
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
    pause = False
    ms_solver = ImplictMassSpringSolver(robot_builder)

    window = ti.ui.Window('Implicit Mass Spring System', res=(800, 800), vsync=True, show_window=False)
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
    while window.running:

        # if window.get_event(ti.ui.PRESS):
        #     if window.event.key == ti.ui.ESCAPE:
        #         break
        # if window.is_pressed(ti.ui.SPACE):
        #     pause = not pause

        pause = True
        if not pause:
            ms_solver.update(h)
        else:
            with ti.ad.Tape(loss=ms_solver.loss):
                ms_solver.update(h)
                ms_solver.compute_center()
                ms_solver.compute_loss()
            # print(f"Actuation Grad: {ms_solver.actuation.grad}")
            # print(f"force grad: {ms_solver.force.grad}")
            # print(f"b Grad: {ms_solver.b.grad}")
            # print(f"Position Grad: {ms_solver.pos.grad}")

        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        scene.point_light(pos=(0, 1, 0), color=(.7, .7, .7))
        scene.point_light(pos=(-1, 1, 0), color=(.7, .7, .7))
        scene.ambient_light((0.2, 0.2, 0.2))


        actuation = ms_solver.pos.to_numpy()[actuator_pos_index]
        actuator_pos.from_numpy(actuation)

        scene.particles(actuator_pos, radius=0.005, color=(0.0, 0.0, 0.5))
        scene.mesh(ms_solver.pos, indices=indices, color=(0.8, 0.6, 0.2))
        scene.mesh(vertices_ground, indices=indices_ground, color=(0.5, 0.5, 0.5), two_sided=True)
        canvas.scene(scene)
        # window.show()

if __name__ == '__main__':
    main()
