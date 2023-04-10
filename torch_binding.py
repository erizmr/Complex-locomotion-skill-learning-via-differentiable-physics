import torch
import taichi as ti

from utils import (data_type, ti2torch, ti2torch_grad, ti2torch_grad_vec,
                    ti2torch_vec, torch2ti, torch2ti_grad, torch2ti_grad_vec,
                    torch2ti_vec, torch2ti_grad_vec3, ti2torch_grad_vec3, torch_type)

from multitask.robot_design import RobotDesignMassSpring3D
from grad_implicit_ms_cg import ImplictMassSpringSolver

class MassSpringSolver(torch.nn.Module):
    def __init__(self, robot_builder: RobotDesignMassSpring3D, dim=3):
        super(MassSpringSolver, self).__init__()

        self.ms_solver = ImplictMassSpringSolver(robot_builder)
        self.h = 0.01
        self.substeps = 1

        self.register_buffer(
            'output_pos',
            torch.zeros(3, self.ms_solver.NV, dtype=torch_type),
            persistent=False
        )
        self.register_buffer(
            'action_grad',
            torch.zeros(self.ms_solver.NE, dtype=torch_type),
            persistent=False
        )
        class _module_function(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input_action):

                torch2ti(self.ms_solver.actuation, input_action.contiguous())
                for i in range(self.substeps):
                    self.ms_solver.update(self.h)
                self.output_pos = self.ms_solver.pos.to_torch()
                
                return self.output_pos

            @staticmethod
            def backward(ctx, doutput):                
                # print(doutput.contiguous().shape)
                # self.zero_grad()
                torch2ti_grad_vec3(self.ms_solver.pos, doutput.contiguous())
                self.ms_solver.update_grad(self.h)
                ti2torch_grad(self.ms_solver.actuation,
                                   self.action_grad.contiguous())
                return self.action_grad

        self._module_function = _module_function

    def zero_grad(self):
        self.ms_solver.zero_grad()

    def grad_check(self, inputs):
        print("Grad checking...")
        # try:
        #     torch.autograd.gradcheck(self._module_function.apply, inputs, eps=1e-2, atol=1e-3, rtol=1.e-3, raise_exception=True)
        # except Exception as e:
        #     print(str(self._module_function.__name__) + " failed: " + str(e))
        torch.autograd.gradcheck(self._module_function.apply, inputs, eps=1e-2, atol=1e-3, rtol=1.e-3, raise_exception=True)

    def forward(self, input_action):
        return self._module_function.apply(input_action)
