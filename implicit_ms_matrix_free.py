# https://www.cs.cmu.edu/~baraff/papers/sig98.pdf
import argparse

import numpy as np

import taichi as ti

from multitask.robot_design import RobotDesignMassSpring3D


@ti.data_oriented
class ImplictMassSpringSolver:
    def __init__(self, robot_builder: RobotDesignMassSpring3D, dim=3):
        self.dim = dim
        _vertices, _springs_data, _faces = robot_builder.get_objects()
        self.vertices = np.array(_vertices) # [NV, 3]
        self.springs_data = np.array(_springs_data) # [NE, (a, b, length, stiffness, actuation)]
        self.faces = np.array(_faces) # [NF, 3]
        print(f"Vertices {self.vertices.shape}, Springs {self.springs_data.shape}, Faces {self.faces.shape}")
        self.NF = self.faces.shape[0]  # number of faces
        self.NV = self.vertices.shape[0]  # number of vertices
        self.NE = self.springs_data.shape[0]  # number of edges

        self.pos = ti.Vector.field(self.dim, ti.f32, self.NV)
        self.initPos = ti.Vector.field(self.dim, ti.f32, self.NV)
        self.vel = ti.Vector.field(self.dim, ti.f32, self.NV)
        self.dv = ti.Vector.field(self.dim, ti.f32, self.NV)
        self.Adv = ti.Vector.field(self.dim, ti.f32, self.NV)

        self.force = ti.Vector.field(self.dim, ti.f32, self.NV)
        self.mass = ti.field(ti.f32, self.NV)
        self.vel_1D = ti.ndarray(ti.f32, self.dim * self.NV)
        self.force_1D = ti.ndarray(ti.f32, self.dim * self.NV)
        self.b = ti.field(ti.f32, self.dim * self.NV)
        self.r0 = ti.field(ti.f32, self.dim * self.NV)
        self.r1 = ti.field(ti.f32, self.dim * self.NV)
        self.p0 = ti.field(ti.f32, self.dim * self.NV)
        self.p1 = ti.field(ti.f32, self.dim * self.NV)

        self.spring = ti.Vector.field(2, ti.i32, self.NE)
        self.spring_actuation_coef = ti.field(ti.f32, self.NE)
        self.actuation = ti.field(ti.f32, self.NE) # parameters to optimize
        self.spring_color = ti.Vector.field(3, ti.f32, self.NE)
        self.indices = ti.field(ti.i32, 2 * self.NE)
        self.spring_color_draw = ti.Vector.field(3, ti.f32, 2 * self.NE)
        self.Jx = ti.Matrix.field(self.dim, self.dim, ti.f32,
                                  self.NE)  # Jacobian with respect to position
        self.Jv = ti.Matrix.field(self.dim, self.dim, ti.f32,
                                  self.NE)  # Jacobian with respect to velocity
        self.rest_len = ti.field(ti.f32, self.NE)
        self.ks = 1e7  # spring stiffness
        self.kd = 0.5  # damping constant
        self.kf = 1.0e5  # fix point stiffness

        # self.gravity = ti.Vector([0.0, -2.0, 0.0])
        self.gravity = ti.Vector([0.0, 0.0, 0.0])
        self.init_pos()
        self.init_edges()
        # self.MassBuilder = ti.linalg.SparseMatrixBuilder(
        #     self.dim * self.NV, self.dim * self.NV, max_num_triplets=10000)
        # self.DBuilder = ti.linalg.SparseMatrixBuilder(self.dim * self.NV,
        #                                               self.dim * self.NV,
        #                                               max_num_triplets=10000)
        # self.KBuilder = ti.linalg.SparseMatrixBuilder(self.dim * self.NV,
        #                                               self.dim * self.NV,
        #                                               max_num_triplets=10000)
        # self.init_mass_sp(self.MassBuilder)
        # self.M = self.MassBuilder.build()
        # self.fix_vertex = [self.N, self.NV - 1]
        # self.Jf = ti.Matrix.field(self.dim, self.dim, ti.f32, len(self.fix_vertex))  # fix constraint hessian


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

    # @ti.kernel
    # def init_mass_sp(self, M: ti.types.sparse_matrix_builder()):
    #     for i in range(self.NV):
    #         mass = self.mass[i]
    #         M[3 * i + 0, 3 * i + 0] += mass
    #         M[3 * i + 1, 3 * i + 1] += mass
    #         M[3 * i + 2, 3 * i + 2] += mass

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
            dis = pos2 - pos1

            # self.actuation[i] = ti.random()
            # print(self.actuation[i])
            self.actuation[i] = ti.sin(i * 3.1415926 / 4)
            # self.actuation[i] = -1.0
            target_length = self.rest_len[i] * (1.0 + self.spring_actuation_coef[i] * self.actuation[i])
            force = self.ks * (dis.norm() -
                               target_length) * dis.normalized()
            self.force[idx1] += force
            self.force[idx2] -= force
    

    # @ti.kernel
    # def compute_Jacobians(self):
    #     for i in self.spring:
    #         idx1, idx2 = self.spring[i][0], self.spring[i][1]
    #         pos1, pos2 = self.pos[idx1], self.pos[idx2]
    #         dx = pos1 - pos2
    #         I = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    #         dxtdx = ti.Matrix([[dx[0] * dx[0], dx[0] * dx[1], dx[0]*dx[2]],
    #                            [dx[1] * dx[0], dx[1] * dx[1], dx[1]*dx[2]],
    #                            [dx[2] * dx[0], dx[1] * dx[2], dx[2]*dx[2]]])
    #         l = dx.norm()
    #         if l != 0.0:
    #             l = 1.0 / l
    #         # self.Jx[i] = (I - self.rest_len[i] * l * (I - dxtdx * l**2)) * self.ks

    #         # Clamp the potential negative part to make the hessian positive definite
    #         self.Jx[i] = (ti.max(1 - self.rest_len[i] * l, 0) * I + self.rest_len[i] * dxtdx * l**3) * self.ks
    #         # self.Jv[i] = self.kd * I
    #         # self.Jv[i] = self.kd * dxtdx


    @ti.kernel
    def matrix_vector_product(self):
        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            pos1, pos2 = self.pos[idx1], self.pos[idx2]
            dx = pos1 - pos2
            I = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            dxtdx = ti.Matrix([[dx[0] * dx[0], dx[0] * dx[1], dx[0]*dx[2]],
                               [dx[1] * dx[0], dx[1] * dx[1], dx[1]*dx[2]],
                               [dx[2] * dx[0], dx[1] * dx[2], dx[2]*dx[2]]])
            l = dx.norm()
            if l != 0.0:
                l = 1.0 / l
            # self.Jx[i] = (I - self.rest_len[i] * l * (I - dxtdx * l**2)) * self.ks

            # Clamp the potential negative part to make the hessian positive definite
            self.Jx[i] = (ti.max(1 - self.rest_len[i] * l, 0) * I + self.rest_len[i] * dxtdx * l**3) * self.ks
            val = self.Jx[i].dot(self.dv[idx1] - self.dv[idx2])
            self.Adv[idx1] -= val + self.mass[idx1] * self.dv[idx1]
            self.Adv[idx2] += val + self.mass[idx2] * self.dv[idx2]


            # self.Jv[i] = self.kd * I
            # self.Jv[i] = self.kd * dxtdx




    # @ti.kernel
    # def assemble_K(self, K: ti.types.sparse_matrix_builder()):
    #     for i in self.spring:
    #         idx1, idx2 = self.spring[i][0], self.spring[i][1]
    #         for m, n in ti.static(ti.ndrange(self.dim, self.dim)):
    #             K[self.dim * idx1 + m, self.dim * idx1 + n] -= self.Jx[i][m, n]
    #             K[self.dim * idx1 + m, self.dim * idx2 + n] += self.Jx[i][m, n]
    #             K[self.dim * idx2 + m, self.dim * idx1 + n] += self.Jx[i][m, n]
    #             K[self.dim * idx2 + m, self.dim * idx2 + n] -= self.Jx[i][m, n]
    #     # for m, n in ti.static(ti.ndrange(self.dim, self.dim)):
    #     #     K[self.dim * self.N + m, self.dim * self.N + n] += self.Jf[0][m, n]
    #     #     K[self.dim * (self.NV - 1) + m, self.dim * (self.NV - 1) + n] += self.Jf[1][m, n]


    # @ti.kernel
    # def assemble_D(self, D: ti.types.sparse_matrix_builder()):
    #     for i in self.spring:
    #         idx1, idx2 = self.spring[i][0], self.spring[i][1]
    #         for m, n in ti.static(ti.ndrange(self.dim, self.dim)):
    #             D[self.dim * idx1 + m, self.dim * idx1 + n] -= self.Jv[i][m, n]
    #             D[self.dim * idx1 + m, self.dim * idx2 + n] += self.Jv[i][m, n]
    #             D[self.dim * idx2 + m, self.dim * idx1 + n] += self.Jv[i][m, n]
    #             D[self.dim * idx2 + m, self.dim * idx2 + n] -= self.Jv[i][m, n]

    @ti.kernel
    def updatePosVel(self, h: ti.f32):
        for i in self.pos:
            self.vel[i] += self.dv[i]
            self.pos[i] += h * self.vel[i]

    @ti.kernel
    def copy_to(self, des: ti.types.ndarray(), source: ti.template()):
        for i in range(self.NV):
            des[self.dim * i] = source[i][0]
            des[self.dim * i + 1] = source[i][1]
            des[self.dim * i + 2] = source[i][2]

    # @ti.kernel
    # def compute_b(self, b: ti.types.ndarray(), f: ti.types.ndarray(),
    #               Kv: ti.types.ndarray(), h: ti.f32):
    #     for i in range(self.dim * self.NV):
    #         b[i] = (f[i] + Kv[i] * h) * h


    @ti.kernel
    def compute_b(self, h: ti.f32):
        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            val = self.Jx[i].dot(self.vel[idx1] - self.vel[idx2]) * h
            self.b[idx1] -= (val + self.force[idx1]) * h
            self.b[idx2] += (val + self.force[idx2]) * h
    

    @ti.kernel
    def add(ans: ti.template(), a: ti.template(), k: ti.f32, b: ti.template()):
        for i in ans:
            ans[i] = a[i] + k * b[i]


    @ti.kernel
    def dot(a: ti.template(), b: ti.template()) -> ti.f32:
        ans = 0.0
        for i in a:
            ans += a[i].dot(b[i])
        return ans


    def update(self, h):
        self.compute_force()
        # Solve the linear system
        self.cg_solver(h)
        # print(solver.info())
        self.updatePosVel(h)

    
    def cg_solver(self, h):
        self.matrix_vector_product()
        # b = (force + h * K @ vel) * h
        self.compute_b(h)
        self.add(self.r0, self.Adv, -1.0, self.b)



    @ti.kernel
    def spring2indices(self):
        for i in self.spring:
            self.indices[self.dim * i + 0] = self.spring[i][0]
            self.indices[self.dim * i + 1] = self.spring[i][1]


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

    window = ti.ui.Window('Implicit Mass Spring System', res=(800, 800), vsync=True)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.make_camera()
    camera.position(0.2, 1.1, 1.1)
    camera.lookat(0.2, 0.1, 0.1)
    camera.up(0, 1, 0)

    # ms_solver.spring2indices()

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

        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.ESCAPE:
                break
        if window.is_pressed(ti.ui.SPACE):
            pause = not pause

        if not pause:
            ms_solver.update(h)

        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        scene.point_light(pos=(0, 1, 0), color=(.7, .7, .7))
        scene.point_light(pos=(-1, 1, 0), color=(.7, .7, .7))
        scene.ambient_light((0.2, 0.2, 0.2))


        actuation = ms_solver.pos.to_numpy()[actuator_pos_index]
        actuator_pos.from_numpy(actuation)

        scene.particles(actuator_pos, radius=0.005, color=(0.0, 0.0, 0.5))
        scene.mesh(ms_solver.pos, indices=indices, color=(0.8, 0.6, 0.2))
        # scene.mesh(vertices_ground, indices=indices_ground, color=(0.5, 0.5, 0.5), two_sided=True)
        canvas.scene(scene)
        window.show()

if __name__ == '__main__':
    main()
