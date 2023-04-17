import argparse
import torch 

import taichi as ti
import numpy as np

from multitask.robot_design import RobotDesignMassSpring3D

from torch_mass_spring import MassSpringSolver



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

    BATCH_SIZE = 8
    VIS_BATCH = 5
    SUBSTEPS = 10
    dt = 0.01
    pause = False
    ms_solver = MassSpringSolver(robot_builder=robot_builder, batch_size=BATCH_SIZE, substeps=SUBSTEPS, dt=dt)

    pause = False

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
            input_actions = torch.rand(BATCH_SIZE, ms_solver.ms_solver.NE, dtype=torch.float64, requires_grad=True)
            ms_solver(input_actions)

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