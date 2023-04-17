import torch
import taichi as ti

from utils import (data_type, ti2torch, ti2torch_grad, ti2torch_grad_vec,
                    ti2torch_vec, torch2ti, torch2ti_grad, torch2ti_grad_vec,
                    torch2ti_vec, torch2ti_grad_vec3, ti2torch_grad_vec3, torch2ti_vec3, ti2torch_vec3, torch_type)

from multitask.robot_design import RobotDesignMassSpring3D
from grad_implicit_ms_cg import ImplictMassSpringSolver

class MassSpringSolver(torch.nn.Module):
    def __init__(self, robot_builder: RobotDesignMassSpring3D, batch_size=1, substeps=1, dt=0.01, dim=3):
        super(MassSpringSolver, self).__init__()
        self.batch_size = batch_size
        self.dt = dt
        self.substeps = substeps
        self.cnt = 0
        self.ms_solver = ImplictMassSpringSolver(robot_builder, batch=batch_size, substeps=self.substeps, dt=dt, dim=dim)
        self.NV = self.ms_solver.NV
        self.NE = self.ms_solver.NE
        self.pos = self.ms_solver.pos

        self.register_buffer(
            'output_pos',
            torch.zeros(self.batch_size, self.substeps, self.NV, 3, dtype=torch_type),
            persistent=False
        )
        self.register_buffer(
            'grad_input_actions',
            torch.zeros(self.batch_size, self.NE, dtype=torch_type),
            persistent=False
        )
        class _module_function(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input_actions):
                self.cnt += 1
                # print("forawrd num ", self.cnt)
                torch2ti(self.ms_solver.actuation, input_actions.contiguous())
                # An initial guess set to zero
                # self.ms_solver.init_pos()
                for i in range(self.substeps):
                    self.ms_solver.update(i)
                self.ms_solver.copy_states()
                ti2torch_vec3(self.ms_solver.pos, self.output_pos.contiguous())
                
                return self.output_pos

            @staticmethod
            def backward(ctx, grad_output_pos):
                self.cnt -= 1
                # print("backward num ", self.cnt)          
                # print(doutput.contiguous().shape)
                self.zero_grad()
                # print("grad rhs shape ", self.grad_rhs.contiguous().shape)
                torch2ti_grad_vec3(self.ms_solver.pos, grad_output_pos.contiguous())
                for i in reversed(range(self.substeps)):
                    self.ms_solver.update_grad(i)
                ti2torch_grad(self.ms_solver.actuation, self.grad_input_actions.contiguous())
                # print("grad_input_actions", self.grad_input_actions)
                return self.grad_input_actions

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
