import argparse
import torch 
import math 

import taichi as ti
import numpy as np

from multitask.robot_design import RobotDesignMassSpring3D

from torch_mass_spring import MassSpringSolver, ActuationNet
from torch import nn
from utils import torch_type
from torchviz import make_dot


def main():
    parser = argparse.ArgumentParser("implicit mass spring ")
    parser.add_argument('-a',
                        '--arch',
                        required=False,
                        default="cuda",
                        dest='arch',
                        type=str,
                        help='The arch (backend) to run this example on')
    
    parser.add_argument('--robot_design_file',
                        default='./cfg3d/sim_quad.json',
                        help='robot design file')
    parser.add_argument('--train',
                        action='store_true',
                        help='train the model')
    args = parser.parse_args()
    # args, unknowns = parser.parse_known_args()
    arch = args.arch
    if arch in ["x64", "cpu", "arm64"]:
        ti.init(arch=ti.cpu)
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
    STEP_NUM = 2
    EPOCHS = 20
    dt = 0.01
    learning_rate = 1e-1
    pause = False
    device = args.arch
    ms_solver = MassSpringSolver(robot_builder=robot_builder, batch_size=BATCH_SIZE, substeps=SUBSTEPS, dt=dt).to(device)
    N_SIN_WAVES = 64
    controller = ActuationNet(input_dim=N_SIN_WAVES+ms_solver.NV*3, output_dim=ms_solver.ms_solver.NE, dtype=torch_type).to(device)
    print(f"number of elements {ms_solver.ms_solver.NE}")
    if args.train:
        optimizer = torch.optim.Adam(controller.parameters(), lr=learning_rate)
        target_pos = torch.tensor([[1.0, 0.1, 0.1] for _ in range(BATCH_SIZE)], requires_grad=True).to(device)
        print("target shape ", target_pos.shape)
        # Training pipeline
        for epoch in range(EPOCHS):
            intput_pos = ms_solver.ms_solver.pos.to_torch()[:,0,:].to(device)
            input_vel = ms_solver.ms_solver.vel.to_torch()[:,0,:].to(device)

            for s in range(STEP_NUM):
                sin_features = torch.sin(2 * math.pi * s + 2 * math.pi / N_SIN_WAVES * torch.arange(N_SIN_WAVES, dtype=torch_type)) * torch.ones(BATCH_SIZE, N_SIN_WAVES, dtype=torch_type, requires_grad=True)
                # print(f"sin features {sin_features}")
                state_features = (intput_pos - intput_pos.mean(axis=1)).reshape(BATCH_SIZE, ms_solver.ms_solver.NV * 3)
                # print(f"state features {state_features}")
                # print(f"sin_features shape: {sin_features.shape}, state_features shape: {state_features.shape}")
                input_features = torch.cat([sin_features.to(device), state_features], dim=1).to(device)
                # print(f"sin_features shape: {sin_features.shape}, state_features shape: {state_features.shape}, input_features shape: {input_features.shape}")
                input_actions = controller(input_features)
                output_pos, output_vel = ms_solver(input_actions, intput_pos, input_vel)
                intput_pos, input_vel = output_pos[:,-1,:], output_vel[:,-1,:]
            
            center = output_pos[:,-1,:].mean(axis=1)
            loss = torch.sqrt(((center - target_pos)**2).sum() / BATCH_SIZE)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ms_solver.zero_grad()
            print(f"loss: {loss.item()}")
            # for name, param in controller.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.data)
        torch.save(controller.state_dict(), "optimized_input_actions.pt")
    else:
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

        # Load optimized weights
        controller.load_state_dict(torch.load("optimized_input_actions.pt"))
        controller.eval()

        cnt = 0
        while window.running:

            if window.get_event(ti.ui.PRESS):
                if window.event.key == ti.ui.ESCAPE:
                    break
            if window.is_pressed(ti.ui.SPACE):
                pause = not pause

            if not pause:
                intput_pos = ms_solver.ms_solver.pos.to_torch()[:,0,:].to(device)
                input_vel = ms_solver.ms_solver.vel.to_torch()[:,0,:].to(device)
                for s in range(STEP_NUM):
                    sin_features = torch.sin(2 * math.pi * (s + STEP_NUM*cnt) + 2 * math.pi / N_SIN_WAVES * torch.arange(N_SIN_WAVES, dtype=torch_type)) * torch.ones(BATCH_SIZE, N_SIN_WAVES, dtype=torch_type, requires_grad=True)
                    # print(f"sin features {sin_features}")
                    state_features = (intput_pos - intput_pos.mean(axis=1)).reshape(BATCH_SIZE, ms_solver.ms_solver.NV * 3)
                    # print(f"state features {state_features}")
                    # print(f"sin_features shape: {sin_features.shape}, state_features shape: {state_features.shape}")
                    input_features = torch.cat([sin_features.to(device), state_features], dim=1).to(device)
                    # print(f"sin_features shape: {sin_features.shape}, state_features shape: {state_features.shape}, input_features shape: {input_features.shape}")
                    input_actions = controller(input_features)
                    output_pos, output_vel = ms_solver(input_actions, intput_pos, input_vel)
                    intput_pos, input_vel = output_pos[:,-1,:], output_vel[:,-1,:]
                cnt += 1

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