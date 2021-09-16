import os
import sys
import glob
import taichi as ti
from multitask.arguments import get_args
from multitask.config_sim import ConfigSim
from multitask.diffphy_trainer import DiffPhyTrainer
# from solver_mass_spring import SolverMassSpring
# from solver_mpm import SolverMPM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

offset = 0
def set_target(window):
    if window.get_event(ti.ui.PRESS):
        e = window.event
        if e.key == ti.ui.UP:
            set_target.target_v += 0.01
        if e.key == ti.ui.DOWN:
            set_target.target_v -= 0.01
        print("!!!!!!!!!!!!!!!!!!!!!!! ", set_target.target_v)
    trainer.taichi_env.initialize_interactive(1, set_target.target_v, set_target.target_h, set_target.target_c)
set_target.target_v = 0
set_target.target_h = 0.1
set_target.target_c = 0.0


all_holder = []
lower_height_holder = []
velocity_holder = []
upper_height_holder = []
target_v_holder = []
center_holder = []
def make_decision():
    trainer.nn.clear_single(0)
    trainer.taichi_env.solver.compute_center(0)
    trainer.taichi_env.nn_input(0, offset, 0.08, 0.1)
    if offset % int(control_length) == 0:
        trainer.nn.forward(0)

    holder = []
    v_sub_holder = []
    for i in range(trainer.taichi_env.solver.n_objects):
        holder.append(round(trainer.taichi_env.solver.actuation[0, 0, i], 3))
        v_sub_holder.append(trainer.taichi_env.v[0, 0, i][0])
    # if offset % int(control_length) == 0:
    #     print(f"Frame: {offset}, control signal: {holder}")
    all_holder.append(holder)

    if set_target.target_c == 1.0:
        target_v_holder.append(set_target.target_v / 2)
    else:
        target_v_holder.append(set_target.target_v)
    # velocity_holder.append(np.mean(v_sub_holder))

    center_holder.append(trainer.taichi_env.center[0, 0][0])
    velocity_holder.append(trainer.taichi_env.center[0, 0][0] - center_holder[max(offset, 100) - 100])
    lower_height_holder.append(trainer.taichi_env.height[0, 0])
    upper_height_holder.append(trainer.taichi_env.upper_height[0, 0])

def forward_mass_spring():
    trainer.taichi_env.solver.apply_spring_force(0)
    trainer.taichi_env.solver.advance_toi(1)
    trainer.taichi_env.solver.clear_states(1)

@ti.kernel
def refresh_xv():
    for i in range(trainer.taichi_env.n_objects):
        trainer.taichi_env.x[0, 0, i] = trainer.taichi_env.x[1, 0, i]
        trainer.taichi_env.v[0, 0, i] = trainer.taichi_env.v[1, 0, i]

if __name__ == "__main__":
    ti.init(arch=ti.gpu, default_fp=ti.f32, random_seed=555)
    args = get_args()
    print('args', args)
    config_file = args.config_file
    config = ConfigSim.from_file(config_file, if_mkdir=False)
    config_name = config_file.split('/')[-1].split('.')[0]
    robot_id = config.get_config()["robot"]["robot_id"]
    os.makedirs(f'./video/interactive/robot_{robot_id}/{config_name}', exist_ok=True)
    control_length = config.get_config()["robot"]["control_length"]

    indices = ti.field(ti.i32, len(config.get_config()["robot"]["faces"]) * 3)
    vertices = ti.Vector.field(3, ti.f32, len(config.get_config()["robot"]["objects"]))
    indices_ground = ti.field(ti.i32, 6)
    vertices_ground = ti.Vector.field(3, ti.f32, 4)

    # Enforce the batch size to 1
    config._config["nn"]["batch_size"] = 1
    trainer = DiffPhyTrainer(args, config=config)

    trainer.taichi_env.setup_robot()
    model_paths = glob.glob(os.path.join("saved_results", config_file.split('/')[1].split('.json')[0], "DiffTaichi_DiffPhy/*/models"), recursive=True)
    model_path = sorted(model_paths, key=os.path.getmtime)[-1]
    print("load from : ", model_path)

    trainer.nn.load_weights(os.path.join(model_path, "weight.pkl"))
    # trainer.nn.load_weights(os.path.join(model_path, "best.pkl"))

    window = ti.ui.Window("Difftaichi2", (800, 800), vsync=True)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.make_camera()

    indices.from_numpy(np.array(config.get_config()["robot"]["faces"]).reshape(-1))
    indices_ground.from_numpy(np.array([0, 1, 2, 2, 1, 3]))
    @ti.kernel
    def update_verts():
        vertices_ground[0] = ti.Vector([-1, 0.1, -1])
        vertices_ground[1] = ti.Vector([-1, 0.1, 1])
        vertices_ground[2] = ti.Vector([1, 0.1, -1])
        vertices_ground[3] = ti.Vector([1, 0.1, 1])
        for i in range(trainer.taichi_env.n_objects):
            vertices[i] = trainer.taichi_env.x[0, 0, i]

    while window.running:
        update_verts()
        for i in range(10):
            set_target(window)
            make_decision()
            forward_mass_spring()
            refresh_xv()
            offset += 1

        camera.position(0.2, 1.1, 1.1)
        camera.lookat(0.2, 0.1, 0.1)
        camera.up(0, 1, 0)
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        scene.point_light(pos=(0, 1, 0), color=(.7, .7, .7))
        scene.point_light(pos=(-1, 1, 0), color=(.7, .7, .7))
        scene.ambient_light((0.2, 0.2, 0.2))

        scene.mesh(vertices, indices=indices, color=(0.2, 0.6, 0.2))
        scene.mesh(vertices_ground, indices=indices_ground, color=(0.5, 0.5, 0.5), two_sided=True)
        canvas.scene(scene)
        window.show()
