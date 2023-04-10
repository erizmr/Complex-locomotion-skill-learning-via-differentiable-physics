import taichi as ti
import torch


data_type = ti.f64
torch_type = torch.float64

@ti.kernel
def torch2ti(field: ti.template(), data: ti.types.ndarray()):
    for I in ti.grouped(data):
        field[I] = data[I]


@ti.kernel
def ti2torch(field: ti.template(), data: ti.types.ndarray()):
    for I in ti.grouped(data):
        data[I] = field[I]


@ti.kernel
def ti2torch_grad(field: ti.template(), grad: ti.types.ndarray()):
    for I in ti.grouped(grad):
        grad[I] = field.grad[I]


@ti.kernel
def torch2ti_grad(field: ti.template(), grad: ti.types.ndarray()):
    for I in ti.grouped(field):
        field.grad[I] = grad[I]


@ti.kernel
def torch2ti_vec(field: ti.template(), data: ti.types.ndarray()):
    for I in range(data.shape[0] // 2):
        field[I] = ti.Vector([data[I * 2], data[I * 2 + 1]])


@ti.kernel
def ti2torch_vec(field: ti.template(), data: ti.types.ndarray()):
    for i, j in ti.ndrange(data.shape[0], data.shape[1] // 2):
        data[i, j * 2] = field[i, j][0]
        data[i, j * 2 + 1] = field[i, j][1]


@ti.kernel
def ti2torch_grad_vec(field: ti.template(), grad: ti.types.ndarray()):
    for I in range(grad.shape[0] // 2):
        grad[I * 2] = field.grad[I][0]
        grad[I * 2 + 1] = field.grad[I][1]


@ti.kernel
def torch2ti_grad_vec(field: ti.template(), grad: ti.types.ndarray()):
    for i, j in ti.ndrange(grad.shape[0], grad.shape[1] // 2):
        field.grad[i, j][0] = grad[i, j * 2]
        field.grad[i, j][1] = grad[i, j * 2 + 1]


@ti.kernel
def torch2ti_vec3(field: ti.template(), data: ti.types.ndarray()):
    for i in field:
        field[i][0] = data[i, 0]
        field[i][1] = data[i, 1]
        field[i][2] = data[i, 2]


@ti.kernel
def ti2torch_vec3(field: ti.template(), data: ti.types.ndarray()):
    for i in field:
        data[i, 0] = field[i][0]
        data[i, 1] = field[i][1]
        data[i, 2] = field[i][2]


@ti.kernel
def torch2ti_grad_vec3(field: ti.template(), grad: ti.types.ndarray()):
    for i in field:
        field.grad[i][0] = grad[i, 0]
        field.grad[i][1] = grad[i, 1]
        field.grad[i][2] = grad[i, 2]


@ti.kernel
def ti2torch_grad_vec3(field: ti.template(), grad: ti.types.ndarray()):
    for i in field:
        grad[i, 0] = field.grad[i][0]
        grad[i, 1] = field.grad[i][1]
        grad[i, 2] = field.grad[i][2]

