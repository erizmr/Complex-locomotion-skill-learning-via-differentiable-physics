import os
import sys
import glob
import taichi as ti
from arguments import get_args
from config_sim import ConfigSim
from multitask_obj import DiffPhyTrainer
from solver_mass_spring import SolverMassSpring
from solver_mpm import SolverMPM

offset = 0
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
            set_target.target_h = 0.30
        elif e.key == gui.BACKSPACE:
            set_target.target_v = 0.
            set_target.target_h = 0.1
    print("Status: ", set_target.target_v, set_target.target_h)
    trainer.initialize_interactive(1, set_target.target_v, set_target.target_h)
set_target.target_v = 0
set_target.target_h = 0.1

def make_decision():
    trainer.nn.clear_single(0)
    trainer.solver.compute_center(0)
    trainer.nn_input(0, offset, 0.08, 0.1)
    trainer.nn.forward(0)

def forward_mass_spring():
    trainer.solver.apply_spring_force(0)
    trainer.solver.advance_toi(1)
    trainer.solver.clear_states(1)

@ti.kernel
def refresh_xv():
    for i in range(trainer.n_objects):
        trainer.x[0, 0, i] = trainer.x[1, 0, i]
        trainer.v[0, 0, i] = trainer.v[1, 0, i]

# TODO: clean up
gui = ti.GUI(background_color=0xFFFFFF)
def visualizer():
    gui.clear()
    gui.line((0, trainer.ground_height), (1, trainer.ground_height),
             color=0x000022,
             radius=3)
    trainer.solver.draw_robot(gui=gui, batch_rank=1, t=1, target_v=trainer.target_v)
    gui.show('video/interactive/{:04d}.png'.format(visualizer.frame))
    visualizer.frame += 1
visualizer.frame = 0

if __name__ == "__main__":
    ti.init(arch=ti.gpu, default_fp=ti.f64, random_seed=555)
    args = get_args()
    print('args', args)
    config_file = args.config_file
    config = ConfigSim.from_file(config_file, if_mkdir=False)

    # Enforce the batch size to 1
    config._config["nn"]["batch_size"] = 1
    trainer = DiffPhyTrainer(args, config=config)

    trainer.setup_robot()
    model_path = glob.glob(os.path.join("saved_results", config_file.split('/')[1].split('.json')[0], "DiffTaichi_DiffPhy/*/models"), recursive=True)[0]
    print("load from : ", model_path)
    # With actuation, can be still when v = 0 but can not jump
    # trainer.nn.load_weights("saved_results/weight.pkl")

    # No actuation, can jump but the monster is very active...
    # trainer.nn.load_weights("saved_results/reference/weights/last.pkl")

    # With actuation, looks good
    # trainer.nn.load_weights(
    #     "saved_results/sim_config_DiffPhy_with_actuation/DiffTaichi_DiffPhy/0713_155908/models/weight.pkl")

    # With actuation, looks good
    # trainer.nn.load_weights(
    #     "saved_results/sim_config_DiffPhy_with_actuation_robot5/DiffTaichi_DiffPhy/0713_220913/models/weight.pkl")

    # trainer.nn.load_weights("saved_results/sim_config_DiffPhy_batch_test/DiffTaichi_DiffPhy/0712_174022/models/weight.pkl")
    # trainer.nn.load_weights("remote_results/robot_5/weight.pkl")
    # trainer.nn.load_weights("saved_results/sim_config_DiffPhy_with_actuation_large_h_loss_act_h_v/DiffTaichi_DiffPhy/0713_173036/models/iter5000.pkl")
    # trainer.nn.load_weights(
    #     "saved_results/sim_config_DiffPhy_with_actuation_large_h_loss/DiffTaichi_DiffPhy/0713_180151/models/iter7800.pkl")

    # trainer.nn.load_weights("saved_results/sim_config_DiffPhy_with_actuation_loss_act_h_v/DiffTaichi_DiffPhy/0713_190631/models/weight.pkl")

    # With actuation, looks good
    trainer.nn.load_weights(os.path.join(model_path, "weight.pkl"))

    # trainer.nn.load_weights(
    #     "saved_results/sim_config_DiffPhy_with_actuation_robot3_half_lr/DiffTaichi_DiffPhy/0713_031320/models/weight.pkl")

    print(trainer.x.to_numpy()[0, :, :])
    visualizer()
    while gui.running:
        for i in range(10):
            set_target()
            make_decision()
            forward_mass_spring()
            refresh_xv()
            offset += 1
        visualizer()
