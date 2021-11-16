import os
import glob
import time
import taichi as ti
from multitask.arguments import get_args, get_args_parser
from multitask.config_sim import ConfigSim
from multitask.diffphy_trainer import DiffPhyTrainer
# from solver_mass_spring import SolverMassSpring
# from solver_mpm import SolverMPM
import matplotlib.pyplot as plt
import pandas as pd
from multitask.utils import real

ROBOT_NAMES = {2:"Alpaca", 3:"Monster", 4:"HugeStool", 5:"Stool", 6:"Snake", 7:"Snake-A", 21: "MPM-Stool"}

def set_target():
    for e in gui.get_events():
        if '0' <= e.key <= '9':
            set_target.target_v = (ord(e.key) - ord('0')) * 0.01
            set_target.target_h = 0.1
        elif 'a' <= e.key <= 'z':
            set_target.target_v = (ord(e.key) - ord('a')) * -0.01
            set_target.target_h = 0.1
        elif e.key == gui.SPACE:
            set_target.target_v = 0.
            set_target.target_h = 0.2
        elif e.key == 'Control_L':
            set_target.target_c = 1.0
        elif e.key == 'Control_R':
            set_target.target_c = 0.0
        elif e.key == gui.UP:
            set_target.target_h += 0.01
        elif e.key == gui.DOWN:
            set_target.target_h -= 0.01
        elif e.key == gui.LEFT:
            set_target.target_v -= 0.01
        elif e.key == gui.RIGHT:
            set_target.target_v += 0.01
        elif e.key == gui.BACKSPACE:
            set_target.target_v = 0.
            set_target.target_h = 0.1
    # print("Model Path {} Status: v {:.4f} h {:.4f} c {:.4f}".format(model_path, set_target.target_v, set_target.target_h, set_target.target_c))
    trainer.taichi_env.initialize_interactive(1, set_target.target_v, set_target.target_h, set_target.target_c)

offset = 0
set_target.target_v = 0.0
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
        holder.append(round(trainer.taichi_env.solver.actuation[0, 0, 0, i], 3))
        v_sub_holder.append(trainer.taichi_env.solver_v[0, 0, 0, i][0])
    # if offset % int(control_length) == 0:
    #     print(f"Frame: {offset}, control signal: {holder}")
    all_holder.append(holder)

    if set_target.target_c == 1.0:
        target_v_holder.append(set_target.target_v / 2)
    else:
        target_v_holder.append(set_target.target_v)
    # velocity_holder.append(np.mean(v_sub_holder))

    center_holder.append(trainer.taichi_env.solver_center[0, 0, 0][0])
    velocity_holder.append(trainer.taichi_env.solver_center[0, 0, 0][0] - center_holder[max(offset, 100) - 100])
    lower_height_holder.append(trainer.taichi_env.solver_height[0, 0, 0])
    upper_height_holder.append(trainer.taichi_env.solver_upper_height[0, 0, 0])


def forward_mass_spring():
    trainer.taichi_env.solver.apply_spring_force(0)
    trainer.taichi_env.solver.advance_toi(1)
    trainer.taichi_env.solver.clear_states(1)


def forward_mpm():
    trainer.taichi_env.solver.p2g(0)
    trainer.taichi_env.solver.grid_op()
    trainer.taichi_env.solver.g2p(0)
    trainer.taichi_env.solver.clear_grid_interactive()


@ti.kernel
def refresh_xv():
    for i in range(trainer.taichi_env.n_objects):
        trainer.taichi_env.solver_x[0, 0, 0, i] = trainer.taichi_env.solver_x[0, 1, 0, i]
        trainer.taichi_env.solver_v[0, 0, 0, i] = trainer.taichi_env.solver_v[0, 1, 0, i]
        if ti.static(is_mpm):
            trainer.taichi_env.solver.C[0, 0, 0, i] = trainer.taichi_env.solver.C[0, 1, 0, i]
            trainer.taichi_env.solver.F[0, 0, 0, i] = trainer.taichi_env.solver.F[0, 1, 0, i]


# TODO: clean up
gui = ti.GUI(background_color=0xFFFFFF)


def visualizer(output=False):
    gui.clear()
    gui.line((0, trainer.taichi_env.ground_height), (1, trainer.taichi_env.ground_height),
             color=0x000022,
             radius=3)
    gui.text(f"Agent Name: {ROBOT_NAMES[robot_id]}", (0.05, 0.9), color=0x000022, font_size=20)
    gui.text(f"Control length: {control_length}", (0.05, 0.85), color=0x000022, font_size=20)
    gui.text(f"Targets v: {set_target.target_v:.2f}, h: {set_target.target_h:.2f}, c: {set_target.target_c:.2f}", (0.05, 0.80), color=0x000022, font_size=20)
    trainer.taichi_env.solver.draw_robot(gui=gui, t=1, batch_rank=1, target_v=trainer.taichi_env.target_v)
    if output:
        gui.show('video/interactive/robot_{}/{}/{:04d}.png'.format(robot_id, config_name, visualizer.frame))
    else:
        gui.show()
    visualizer.frame += 1

visualizer.frame = 0

if __name__ == "__main__":
    # args = get_args()
    args_parser = get_args_parser()
    args_parser.add_argument(
        '--save',
        action='store_true',
        help='whether save frames')
    args_parser.add_argument(
        '--control-seq',
        action='store_true',
        help='whether use default control sequence')
    args_parser.add_argument(
        '--use-best',
        action='store_true',
        help='use the model with smallest training loss')
    args = args_parser.parse_args()
    # No tensoboard for interactive
    args.no_tensorboard_train = True
    print('args', args)

    # Init taichi
    # Init taichi
    if args.random:
        args.seed = int(time.time() * 1e6) % 10000
    ti.init(arch=ti.gpu, default_fp=real, random_seed=args.seed, packed=args.packed, device_memory_GB=args.memory,
            debug=args.debug)

    config_file = args.config_file
    config = ConfigSim.from_args_and_file(args, config_file, if_mkdir=False)
    config_name = config_file.split('/')[-1].split('.')[0]
    robot_id = config.get_config()["robot"]["robot_id"]
    os.makedirs(f'./video/interactive/robot_{robot_id}/{config_name}', exist_ok=True)
    control_length = config.get_config()["robot"]["control_length"]
    solver_type = config.get_config()["robot"]["simulator"]
    if solver_type == "mpm":
        is_mpm = True

    # Enforce the batch size to 1
    config._config["nn"]["batch_size"] = 1
    config._config["nn"]["n_models"] = 1
    trainer = DiffPhyTrainer(args, config=config)

    trainer.taichi_env.setup_robot()
    model_paths = glob.glob(os.path.join("saved_results", config_file.split('/')[1].split('.json')[0], "DiffTaichi_DiffPhy/*/models"), recursive=True)
    model_path = sorted(model_paths, key=os.path.getmtime)[-1]
    print("load from : ", model_path)

    if args.use_best:
        trainer.nn.load_weights(os.path.join(model_path, "best.pkl"))
    else:
        trainer.nn.load_weights(os.path.join(model_path, "weight.pkl"))

    visualizer(output=args.save)
    is_mpm = False
    default_control_sequence = args.control_seq
    while gui.running:
        for i in range(6):
            set_target()
            make_decision()
            if solver_type == "mass_spring":
                forward_mass_spring()
            elif solver_type == "mpm":
                forward_mpm()
            else:
                raise NotImplementedError(f"Solver {solver_type} not implemented.")
            refresh_xv()
            offset += 1
        visualizer(output=args.save)

        if default_control_sequence:
            if offset < 100:
                set_target.target_v = 0.0
                set_target.target_h = 0.1
                set_target.target_c = 0.0
            if offset > 100 and offset <= 500:
                set_target.target_v = 0.06
                set_target.target_h = 0.1
                set_target.target_c = 0.0
            if offset > 500 and offset < 2000:
                set_target.target_v = 0.06
                set_target.target_h = 0.1
                set_target.target_c = 0.0
            if offset >= 2000 and offset < 3500:
                set_target.target_v = -0.06
                set_target.target_h = 0.1
                set_target.target_c = 0.0
            if offset >= 3500 and offset < 4000:
                set_target.target_v = 0.0
                set_target.target_h = 0.15
                set_target.target_c = 0.0
            if offset >= 4000 and offset < 4500:
                set_target.target_v = 0.0
                set_target.target_h = 0.18
                set_target.target_c = 0.0
            if offset >= 4500 and offset < 6000:
                set_target.target_v = 0.06
                set_target.target_h = 0.1
                set_target.target_c = 1.0

            if offset == 6000:
                break


    # Draw control signal
    # signal_num = np.array(all_holder).shape[1]
    # for i in range(0, 1):
    #     plt.plot([x for x in range(offset)], np.array(all_holder)[:, i], '-x', label="control signal "+str(i))
    # plt.legend()
    # plt.show()
    # print(velocity_holder)
    # print(lower_height_holder)
    # print(upper_height_holder)
    plt.plot([x for x in range(len(velocity_holder))], velocity_holder, label="v", color='tab:blue')
    plt.plot([x for x in range(len(velocity_holder))], target_v_holder, label="target v", color='tab:blue', alpha=0.5)
    # plt.plot([x for x in range(len(velocity_holder))], lower_height_holder, label="h")
    # plt.plot([x for x in range(len(velocity_holder))], upper_height_holder, label="c")
    # plt.legend()
    # plt.show()

    if args.save:
        data_dict = {"v": velocity_holder, "target_v": target_v_holder}
        data_df = pd.DataFrame(data_dict)
        os.makedirs(f"./stats/robot_{robot_id}/{config_name}", exist_ok=True)
        task_name = "long"
        data_df.to_csv(f"./stats/robot_{robot_id}/{config_name}/stats_{task_name}.csv")


