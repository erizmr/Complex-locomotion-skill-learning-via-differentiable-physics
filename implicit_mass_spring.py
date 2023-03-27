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
        self.NE = self.springs.shape[0]  # number of edges
        self.pos = ti.Vector.field(self.dim, ti.f32, self.NV)
        self.initPos = ti.Vector.field(self.dim, ti.f32, self.NV)
        self.vel = ti.Vector.field(self.dim, ti.f32, self.NV)
        self.force = ti.Vector.field(self.dim, ti.f32, self.NV)
        self.mass = ti.field(ti.f32, self.NV)
        self.vel_1D = ti.ndarray(ti.f32, self.dim * self.NV)
        self.force_1D = ti.ndarray(ti.f32, self.dim * self.NV)
        self.b = ti.ndarray(ti.f32, self.dim * self.NV)

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
        self.ks = 1000.0  # spring stiffness
        self.kd = 0.5  # damping constant
        self.kf = 1.0e5  # fix point stiffness

        self.gravity = ti.Vector([0.0, -2.0, 0.0])
        self.init_pos()
        self.init_edges()
        self.MassBuilder = ti.linalg.SparseMatrixBuilder(
            self.dim * self.NV, self.dim * self.NV, max_num_triplets=10000)
        self.DBuilder = ti.linalg.SparseMatrixBuilder(self.dim * self.NV,
                                                      self.dim * self.NV,
                                                      max_num_triplets=10000)
        self.KBuilder = ti.linalg.SparseMatrixBuilder(self.dim * self.NV,
                                                      self.dim * self.NV,
                                                      max_num_triplets=10000)
        self.init_mass_sp(self.MassBuilder)
        self.M = self.MassBuilder.build()
        # self.fix_vertex = [self.N, self.NV - 1]
        self.Jf = ti.Matrix.field(self.dim, self.dim, ti.f32, len(
            self.fix_vertex))  # fix constraint hessian


    def init_pos(self):
        self.pos.from_numpy(np.array(self.vertices))
        self.initPos.from_numpy(np.array(self.vertices))
        self.vel.from_numpy(np.zeros((self.NV, self.dim)))
        self.mass.from_numpy(np.ones(self.NV))
    

    def init_edges(self):
        # self.spring.from
        pass

    @ti.kernel
    def init_mass_sp(self, M: ti.types.sparse_matrix_builder()):
        for i in range(self.NV):
            mass = self.mass[i]
            M[3 * i + 0, 3 * i + 0] += mass
            M[3 * i + 1, 3 * i + 1] += mass
            M[3 * i + 2, 3 * i + 2] += mass

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

            # self.actuation[i] = ti.random() * 0.5
            # self.actuation[i] = ti.sin(i * 1.0)
            # self.actuation[i] = -1.0
            target_length = self.rest_len[i] * (1.0 + self.spring_actuation_coef[i] * self.actuation[i])
            force = self.ks * (dis.norm() -
                               target_length) * dis.normalized()
            self.force[idx1] += force
            self.force[idx2] -= force
        # fix constraint gradient
        self.force[self.N] += self.kf * (self.initPos[self.N] -
                                         self.pos[self.N])
        self.force[self.NV - 1] += self.kf * (self.initPos[self.NV - 1] -
                                              self.pos[self.NV - 1])

    @ti.kernel
    def compute_Jacobians(self):
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
            self.Jx[i] = (I - self.rest_len[i] * l *
                          (I - dxtdx * l**2)) * self.ks
            # self.Jv[i] = self.kd * I
            self.Jv[i] = self.kd * dxtdx

        # fix point constraint hessian
        # self.Jf[0] = ti.Matrix([[-self.kf, 0], [0, -self.kf]])
        # self.Jf[1] = ti.Matrix([[-self.kf, 0], [0, -self.kf]])
        self.Jf[0] = ti.Matrix([[-self.kf, 0, 0], [0, -self.kf, 0], [0, 0, -self.kf]])
        self.Jf[1] = ti.Matrix([[-self.kf, 0, 0], [0, -self.kf, 0], [0, 0, -self.kf]])


    @ti.kernel
    def assemble_K(self, K: ti.types.sparse_matrix_builder()):
        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            for m, n in ti.static(ti.ndrange(self.dim, self.dim)):
                K[self.dim * idx1 + m, self.dim * idx1 + n] -= self.Jx[i][m, n]
                K[self.dim * idx1 + m, self.dim * idx2 + n] += self.Jx[i][m, n]
                K[self.dim * idx2 + m, self.dim * idx1 + n] += self.Jx[i][m, n]
                K[self.dim * idx2 + m, self.dim * idx2 + n] -= self.Jx[i][m, n]
        for m, n in ti.static(ti.ndrange(self.dim, self.dim)):
            K[self.dim * self.N + m, self.dim * self.N + n] += self.Jf[0][m, n]
            K[self.dim * (self.NV - 1) + m, self.dim * (self.NV - 1) + n] += self.Jf[1][m, n]


    @ti.kernel
    def assemble_D(self, D: ti.types.sparse_matrix_builder()):
        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            for m, n in ti.static(ti.ndrange(self.dim, self.dim)):
                D[self.dim * idx1 + m, self.dim * idx1 + n] -= self.Jv[i][m, n]
                D[self.dim * idx1 + m, self.dim * idx2 + n] += self.Jv[i][m, n]
                D[self.dim * idx2 + m, self.dim * idx1 + n] += self.Jv[i][m, n]
                D[self.dim * idx2 + m, self.dim * idx2 + n] -= self.Jv[i][m, n]

    @ti.kernel
    def updatePosVel(self, h: ti.f32, dv: ti.types.ndarray()):
        for i in self.pos:
            self.vel[i] += ti.Vector([dv[self.dim * i], dv[self.dim * i + 1]])
            self.pos[i] += h * self.vel[i]

    @ti.kernel
    def copy_to(self, des: ti.types.ndarray(), source: ti.template()):
        for i in range(self.NV):
            des[self.dim * i] = source[i][0]
            des[self.dim * i + 1] = source[i][1]
            des[self.dim * i + 2] = source[i][2]

    @ti.kernel
    def compute_b(self, b: ti.types.ndarray(), f: ti.types.ndarray(),
                  Kv: ti.types.ndarray(), h: ti.f32):
        for i in range(self.dim * self.NV):
            b[i] = (f[i] + Kv[i] * h) * h

    def update(self, h):
        self.compute_force()

        self.compute_Jacobians()
        # Assemble global system

        self.assemble_D(self.DBuilder)
        D = self.DBuilder.build()

        self.assemble_K(self.KBuilder)
        K = self.KBuilder.build()

        A = self.M - h * D - h**2 * K

        self.copy_to(self.vel_1D, self.vel)
        self.copy_to(self.force_1D, self.force)

        # b = (force + h * K @ vel) * h
        Kv = K @ self.vel_1D
        self.compute_b(self.b, self.force_1D, Kv, h)

        # Sparse solver
        solver = ti.linalg.SparseSolver(solver_type="LDLT")
        solver.analyze_pattern(A)
        solver.factorize(A)
        # Solve the linear system
        dv = solver.solve(self.b)
        self.updatePosVel(h, dv)


    @ti.kernel
    def spring2indices(self):
        for i in self.spring:
            self.indices[self.dim * i + 0] = self.spring[i][0]
            self.indices[self.dim * i + 1] = self.spring[i][1]
            self.spring_color_draw[self.dim * i + 0] = self.spring_color[i]
            self.spring_color_draw[self.dim * i + 1] = self.spring_color[i]


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
    file_name = args.robot_design_file.split('/')[-1].split('.')[0]
    robot_design_file = args.robot_design_file
    robot_builder = RobotDesignMassSpring3D.from_file(robot_design_file)
    robot_id = robot_builder.robot_id
    vertices, springs, faces = robot_builder.build()
    # robot_builder.draw()

    args, unknowns = parser.parse_known_args()
    arch = args.arch
    if arch in ["x64", "cpu", "arm64"]:
        ti.init(arch=ti.cpu)
    elif arch in ["cuda", "gpu"]:
        ti.init(arch=ti.cuda)
    else:
        raise ValueError('Only CPU and CUDA backends are supported for now.')

    h = 0.01
    pause = False
    ms_solver = ImplictMassSpringSolver(robot_builder)

    use_ggui = args.use_ggui
    window = ti.ui.Window('Implicit Mass Spring System', res=(500, 500), vsync=True)
    ms_solver.spring2indices()
    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.ESCAPE:
                break
        if window.is_pressed(ti.ui.SPACE):
            pause = not pause

        if not pause:
            cloth.update(h)

        window = ti.ui.Window("SNMT", (800, 800), vsync=True)
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
        while window.running:
            camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
            scene.set_camera(camera)

            scene.point_light(pos=(0, 1, 0), color=(.7, .7, .7))
            scene.point_light(pos=(-1, 1, 0), color=(.7, .7, .7))
            scene.ambient_light((0.2, 0.2, 0.2))

            # scene.lines(vertices, width=0.005, indices=indices, color=(0.8, 0.6, 0.2))
            scene.mesh(vertices, indices=indices, color=(0.8, 0.6, 0.2))
            # scene.mesh(vertices_ground, indices=indices_ground, color=(0.5, 0.5, 0.5), two_sided=True)
            canvas.scene(scene)
            window.show()

if __name__ == '__main__':
    main()
