import argparse
import torch 
import math 

import taichi as ti
import numpy as np

from multitask.robot_design import RobotDesignMassSpring3D

from torch_mass_spring import MassSpringSolver, ActuationNet
from torch import nn
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
                        default='./cfg3d/sim_quad.json',
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
    
    robot_design_file = args.robot_design_file

    robot_builder = RobotDesignMassSpring3D.from_file(robot_design_file)
    robot_id = robot_builder.robot_id
    vertices, springs, faces = robot_builder.build()
    # robot_builder.draw()

    BATCH_SIZE = 1
    VIS_BATCH = 0
    SUBSTEPS = 5
    dt = 0.01
    pause = False
    device = "cuda"
    ms_solver = MassSpringSolver(robot_builder=robot_builder, batch_size=BATCH_SIZE, substeps=SUBSTEPS, dt=dt).to(device)

    N_SIN_WAVES = 64
    controller = ActuationNet(input_dim=N_SIN_WAVES+ms_solver.NV*3, output_dim=ms_solver.ms_solver.NE, dtype=torch_type).to(device)
    input_actions = torch.rand(BATCH_SIZE, ms_solver.ms_solver.NE, dtype=torch_type, requires_grad=True)
    print(f"number of elements {ms_solver.ms_solver.NE}")
    optimizer = torch.optim.Adam(controller.parameters(), lr=1e-1)
    target_pos = torch.tensor([[1.0, 0.1, 0.1] for _ in range(BATCH_SIZE)], requires_grad=True).to(device)
    print("target shape ", target_pos.shape)
    # Training pipeline
    EPOCHS = 100
    STEP_NUM = 1
    for epoch in range(EPOCHS):
        for s in range(STEP_NUM):
            sin_features = torch.sin(2 * math.pi * s + 2 * math.pi / N_SIN_WAVES * torch.arange(N_SIN_WAVES, dtype=torch_type)) * torch.ones(BATCH_SIZE, N_SIN_WAVES, dtype=torch_type, requires_grad=True)
            state_features = (ms_solver.ms_solver.pos.to_torch()[:,-1,:] - ms_solver.ms_solver.pos.to_torch()[:,-1,:].mean(axis=1)).reshape(BATCH_SIZE, ms_solver.ms_solver.NV * 3)
            # print(f"sin_features shape: {sin_features.shape}, state_features shape: {state_features.shape}")
            input_features = torch.cat([sin_features, state_features], dim=1).to(device)
            # print(f"sin_features shape: {sin_features.shape}, state_features shape: {state_features.shape}, input_features shape: {input_features.shape}")
            input_actions = controller(input_features)
            output_pos = ms_solver(input_actions)
        center = output_pos[:,-1,:].mean(axis=1)
        loss = torch.sqrt(((center - target_pos)**2).sum() / BATCH_SIZE)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"loss: {loss.item()}")
    torch.save(controller.state_dict(), "optimized_input_actions.pt")

if __name__ == '__main__':
    main()