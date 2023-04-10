import argparse
import torch 

import taichi as ti
import numpy as np

from multitask.robot_design import RobotDesignMassSpring3D

from torch_binding import MassSpringSolver
from torch_cg import CGSolver



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
    
    robot_design_file = args.robot_design_file
    robot_builder = RobotDesignMassSpring3D.from_file(robot_design_file)
    robot_id = robot_builder.robot_id
    vertices, springs, faces = robot_builder.build()
    # robot_builder.draw()

    cg_solver = CGSolver(robot_builder=robot_builder)

    input_b = torch.rand(cg_solver.ms_solver.NV, 3, dtype=torch.float64, requires_grad=True)
    cg_solver.grad_check(input_b)



if __name__ == '__main__':
    main()