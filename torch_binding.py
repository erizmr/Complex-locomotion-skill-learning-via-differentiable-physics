import torch
import taichi as ti

from utils import (data_type, ti2torch, ti2torch_grad, ti2torch_grad_vec,
                    ti2torch_vec, torch2ti, torch2ti_grad, torch2ti_grad_vec,
                    torch2ti_vec, torch2ti_grad_vec3, ti2torch_grad_vec3, torch2ti_vec3, ti2torch_vec3, torch_type)

from multitask.robot_design import RobotDesignMassSpring3D
from grad_implicit_ms_cg import ImplictMassSpringSolver

class MassSpringSolver(torch.nn.Module):
    def __init__(self, robot_builder: RobotDesignMassSpring3D, dim=3):
        super(MassSpringSolver, self).__init__()

        self.ms_solver = ImplictMassSpringSolver(robot_builder)
        self.h = 0.01
        self.substeps = 1
        self.cnt = 0

        self.register_buffer(
            'output_dv',
            torch.zeros(self.ms_solver.NV, 3, dtype=torch_type),
            persistent=False
        )
        self.register_buffer(
            'grad_rhs',
            torch.zeros(self.ms_solver.NV, 3, dtype=torch_type),
            persistent=False
        )
        class _module_function(torch.autograd.Function):

            @staticmethod
            def forward(ctx, rhs):
                self.cnt += 1
                print("forawrd num ", self.cnt)
                torch2ti_vec3(self.ms_solver.b, rhs.contiguous())
                # An initial guess set to zero
                self.ms_solver.dv.fill(0.0)
                self.ms_solver.compute_force()
                self.ms_solver.compute_jacobian()
                self.ms_solver.cg_solver(self.ms_solver.dv, self.h)
                ti2torch_vec3(self.ms_solver.dv, self.output_dv.contiguous())
                
                return self.output_dv

            @staticmethod
            def backward(ctx, grad_output_dv):
                self.cnt -= 1
                print("backward num ", self.cnt)           
                # print(doutput.contiguous().shape)
                self.zero_grad()
                # print("grad rhs shape ", self.grad_rhs.contiguous().shape)
                torch2ti_grad_vec3(self.ms_solver.dv, grad_output_dv.contiguous())
                self.ms_solver.cg_solver_grad(self.ms_solver.dv, self.h)
                ti2torch_grad_vec3(self.ms_solver.b, self.grad_rhs.contiguous())
                print("grad_rhs grad ", self.grad_rhs)
                return self.grad_rhs

        self._module_function = _module_function

    def zero_grad(self):
        self.ms_solver.zero_grad()

    def grad_check(self, inputs):
        print("Grad checking...")
        # try:
        #     torch.autograd.gradcheck(self._module_function.apply, inputs, eps=1e-2, atol=1e-3, rtol=1.e-3, raise_exception=True)
        # except Exception as e:
        #     print(str(self._module_function.__name__) + " failed: " + str(e))
        return torch.autograd.gradcheck(self._module_function.apply, inputs, eps=1e-6, atol=1e-3, rtol=1.e-3, raise_exception=True)

    def forward(self, input_action):
        return self._module_function.apply(input_action)
