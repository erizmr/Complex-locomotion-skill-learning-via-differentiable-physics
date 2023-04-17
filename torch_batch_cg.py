import torch
import taichi as ti

from utils import (data_type, ti2torch, ti2torch_grad, ti2torch_grad_vec,
                    ti2torch_vec, torch2ti, torch2ti_grad, torch2ti_grad_vec,
                    torch2ti_vec, torch2ti_grad_vec3, ti2torch_grad_vec3, torch2ti_vec3, ti2torch_vec3, torch_type)

from multitask.robot_design import RobotDesignMassSpring3D
from grad_implicit_ms_cg import ImplictMassSpringSolver

class BatchedCGSolver(torch.nn.Module):
    def __init__(self, robot_builder: RobotDesignMassSpring3D, batch=1, dim=3):
        super(BatchedCGSolver, self).__init__()

        self.batch_size = batch
        self.ms_solver = ImplictMassSpringSolver(robot_builder, batch=self.batch_size)
        self.h = 0.01
        self.substeps = 1
        self.cnt = 0

        self.register_buffer(
            'output_dv',
            torch.zeros(self.batch_size, self.ms_solver.NV, 3, dtype=torch_type),
            persistent=False
        )
        self.register_buffer(
            'grad_rhs',
            torch.zeros(self.batch_size, self.ms_solver.NV, 3, dtype=torch_type),
            persistent=False
        )
        class _module_function(torch.autograd.Function):

            @staticmethod
            def forward(ctx, rhs: torch.Tensor):
                self.cnt += 1
                if self.cnt % 100 == 0:
                    print("forawrd num ", self.cnt)
                torch2ti_vec3(self.ms_solver.b, rhs.contiguous())
                print(f"rhs {rhs}")
                # An initial guess set to zero
                self.ms_solver.dv_one_step.fill(0.0)
                self.ms_solver.compute_force(0)
                self.ms_solver.compute_jacobian(0)
                self.ms_solver.cg_solver(self.ms_solver.dv_one_step)
                ti2torch_vec3(self.ms_solver.dv_one_step, self.output_dv.contiguous())
                print("output_dv ", self.output_dv)
                return self.output_dv

            @staticmethod
            def backward(ctx, grad_output_dv: torch.Tensor):
                self.cnt -= 1
                if self.cnt % 100 == 0:
                    print("backward num ", self.cnt)           
                self.zero_grad()
                # print("grad rhs shape ", self.grad_rhs.contiguous().shape)
                torch2ti_grad_vec3(self.ms_solver.dv_one_step, grad_output_dv.contiguous())
                self.ms_solver.cg_solver_grad(self.ms_solver.dv_one_step)
                ti2torch_grad_vec3(self.ms_solver.b, self.grad_rhs.contiguous())
                print("grad_rhs grad ", self.ms_solver.b.grad)
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
