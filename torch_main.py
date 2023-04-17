import argparse
import torch 

import taichi as ti
import numpy as np
import math

from multitask.robot_design import RobotDesignMassSpring3D

from torch_mass_spring import MassSpringSolver, ActuationNet
from utils import torch_type



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
                        default='cfg3d/sim_quad.json',
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
    
    robot_design_file = args.robot_design_file
    robot_builder = RobotDesignMassSpring3D.from_file(robot_design_file)
    robot_id = robot_builder.robot_id
    vertices, springs, faces = robot_builder.build()
    # robot_builder.draw()

    BATCH_SIZE = 1
    VIS_BATCH = 0
    SUBSTEPS = 10
    STEP_NUM = 1
    dt = 0.01
    pause = False
    device = "cuda"
    ms_solver = MassSpringSolver(robot_builder=robot_builder, batch_size=BATCH_SIZE, substeps=SUBSTEPS, dt=dt).to(device)

    N_SIN_WAVES = 64
    controller = ActuationNet(input_dim=N_SIN_WAVES+ms_solver.NV*3, output_dim=ms_solver.ms_solver.NE, dtype=torch_type).to(device)
    print(f"number of elements {ms_solver.ms_solver.NE}")
    target_pos = torch.tensor([[1.0, 0.1, 0.1] for _ in range(BATCH_SIZE)], requires_grad=True).to(device)
    print("target shape ", target_pos.shape)

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

    target_pos = ti.Vector.field(3, ti.f32, shape=1)
    target_pos[0] = ti.Vector([1.0, 0.1, 0.1])

    # optimized_input_actions = torch.load("optimized_input_actions.pt")
    controller.load_state_dict(torch.load("optimized_input_actions.pt"))
    controller.eval()
    while window.running:

        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.ESCAPE:
                break
        if window.is_pressed(ti.ui.SPACE):
            pause = not pause

        if not pause:
            for s in range(STEP_NUM):
                # input_actions = torch.rand(BATCH_SIZE, ms_solver.ms_solver.NE, dtype=torch.float64, requires_grad=True)
                sin_features = torch.sin(2 * math.pi * s + 2 * math.pi / N_SIN_WAVES * torch.arange(N_SIN_WAVES, dtype=torch_type)) * torch.ones(BATCH_SIZE, N_SIN_WAVES, dtype=torch_type, requires_grad=True)
                state_features = (ms_solver.ms_solver.pos.to_torch()[:,s,:] - ms_solver.ms_solver.pos.to_torch()[:,s,:].mean(axis=1)).reshape(BATCH_SIZE, ms_solver.ms_solver.NV * 3)
                # print(f"sin_features shape: {sin_features.shape}, state_features shape: {state_features.shape}")
                input_features = torch.cat([sin_features, state_features], dim=1).to(device)
                input_actions = controller(input_features)
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
        scene.particles(target_pos, radius=0.05, color=(0.5, 0.0, 0.0))
        scene.mesh(pos_vis_buffer, indices=indices, color=(0.8, 0.6, 0.2))
        scene.mesh(vertices_ground, indices=indices_ground, color=(0.5, 0.5, 0.5), two_sided=True)
        canvas.scene(scene)
        window.show()



if __name__ == '__main__':
    main()