import torch
import taichi as ti

from utils import (data_type, ti2torch, ti2torch_grad, ti2torch_grad_vec,
                    ti2torch_vec, torch2ti, torch2ti_grad, torch2ti_grad_vec,
                    torch2ti_vec, torch2ti_grad_vec3, ti2torch_grad_vec3, torch2ti_vec3, ti2torch_vec3, torch_type)

from multitask.robot_design import RobotDesignMassSpring3D
from grad_implicit_ms_cg import ImplictMassSpringSolver
from torch import nn


class ActuationNet(nn.Module):
    def __init__(self, input_dim, output_dim, dtype=torch.float32):
        super(ActuationNet, self).__init__()
        # input_dim = n_sin_wave + n_vertices + target_pos
        self.input_dim = input_dim
        # output_dim = n_edges
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, 64, dtype=dtype)
        self.fc2 = nn.Linear(64, self.output_dim, dtype=dtype)
        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()

    def forward(self, x):
        # x: [batch, input_dim]
        x = self.fc1(x)
        x = torch.sin(x)
        x = self.fc2(x)
        x = torch.sin(x)
        return x


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
            torch.zeros(self.batch_size, self.substeps+1, self.ms_solver.NV, 3, dtype=torch_type),
            persistent=False
        )
        self.register_buffer(
            'output_vel',
            torch.zeros(self.batch_size, self.substeps+1, self.ms_solver.NV, 3, dtype=torch_type),
            persistent=False
        )
        self.register_buffer(
            'grad_input_actions',
            torch.zeros(self.batch_size, self.ms_solver.NE, dtype=torch_type),
            persistent=False
        )
        self.register_buffer(
            'grad_input_pos',
            torch.zeros(self.batch_size, self.ms_solver.NV, 3, dtype=torch_type),
            persistent=False
        )
        self.register_buffer(
            'grad_input_vel',
            torch.zeros(self.batch_size, self.ms_solver.NV, 3, dtype=torch_type),
            persistent=False
        )
        class _module_function(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input_actions: torch.Tensor, input_pos: torch.Tensor, input_vel: torch.Tensor):
                self.cnt += 1
                # print("forawrd num ", self.cnt)
                torch2ti(self.ms_solver.actuation, input_actions.contiguous())
                self.ms_solver.initialize(input_pos=input_pos, input_vel=input_vel)
                for i in range(self.substeps):
                    self.ms_solver.update(i)                
                ti2torch_vec3(self.ms_solver.pos, self.output_pos.contiguous())
                ti2torch_vec3(self.ms_solver.vel, self.output_vel.contiguous())

                # Save the whole trajectory for backward use
                ctx.save_for_backward(self.output_pos, self.output_vel, self.ms_solver.dv.to_torch())

                return self.output_pos, self.output_vel

            @staticmethod
            def backward(ctx, grad_output_pos: torch.Tensor, grad_output_vel: torch.Tensor):
                self.cnt -= 1
                # print("backward num ", self.cnt)          
                self.zero_grad()

                # Restore the saved trajectory
                cached_pos, cached_vel, cached_dv = ctx.saved_tensors
                torch2ti_vec3(self.ms_solver.pos, cached_pos.contiguous())
                torch2ti_vec3(self.ms_solver.vel, cached_vel.contiguous())
                torch2ti_vec3(self.ms_solver.dv, cached_dv.contiguous())

                print(f"cnt {self.cnt} grad output pos {grad_output_pos[:, -1, :].sum()}")
                # print("grad output pos shape ", grad_output_pos.shape, " grad output vel shape ", grad_output_vel.shape)
                # Restore the gradient computed from last torch function
                torch2ti_grad_vec3(self.ms_solver.pos, grad_output_pos.contiguous())
                torch2ti_grad_vec3(self.ms_solver.vel, grad_output_vel.contiguous())
                
                for i in reversed(range(self.substeps)):
                    # We only need the actuation gradient of the initial step
                    self.ms_solver.actuation.grad.fill(0.0)
                    self.ms_solver.update_grad(i)
                
                ti2torch_grad(self.ms_solver.actuation, self.grad_input_actions.contiguous())
                # Copy the gradient that has been backpropated to the initial step to outputs
                self.ms_solver.copy_grad(self.grad_input_pos, self.grad_input_vel)
                # print("grad_input_actions", self.grad_input_actions)

                print(f"cnt {self.cnt} grad input pos {self.grad_input_pos.sum()}")
                return self.grad_input_actions, self.grad_input_pos, self.grad_input_vel

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

    def forward(self, input_action: torch.Tensor, input_pos: torch.Tensor, input_vel: torch.Tensor):
        return self._module_function.apply(input_action, input_pos, input_vel)
