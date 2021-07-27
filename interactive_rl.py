import os
import sys
import glob
import torch
import taichi as ti
from multitask.arguments import get_args
from multitask.config_sim import ConfigSim
from ppo.RL_trainer import RLTrainer
from ppo.a2c_ppo_acktr.envs import make_vec_envs
from ppo.a2c_ppo_acktr.utils import get_vec_normalize

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
            set_target.target_h = 0.2
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
    print("Modlel name {} Status: {:.4f} {:.4f} {:.4f}".format(model_name, set_target.target_v, set_target.target_h, set_target.target_c))
    eval_envs.env_method("set_targets", set_target.target_v, set_target.target_h, set_target.target_c, indices=0)

set_target.target_v = 0
set_target.target_h = 0.1
set_target.target_c = 0.


# def make_decision():
#     trainer.nn.clear_single(0)
#     trainer.solver.compute_center(0)
#     trainer.nn_input(0, offset, 0.08, 0.1)
#     trainer.nn.forward(0)
#
#
# def forward_mass_spring():
#     trainer.solver.apply_spring_force(0)
#     trainer.solver.advance_toi(1)
#     trainer.solver.clear_states(1)

def refresh_xv():
    eval_envs.env_method("refresh_xv", indices=0)


# TODO: clean up
gui = ti.GUI(background_color=0xFFFFFF)


def visualizer():
    gui.clear()
    gui.line((0, ground_height), (1, ground_height),
             color=0x000022,
             radius=3)
    eval_envs.env_method("draw_robot", gui, indices=0)
    # trainer.solver.draw_robot(gui=gui, batch_rank=1, t=1, target_v=trainer.target_v)
    # gui.show('video/interactive/{:04d}.png'.format(visualizer.frame))
    gui.show()
    visualizer.frame += 1


visualizer.frame = 0


if __name__ == "__main__":
    args = get_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print('args', args)
    config_file = args.config_file
    print(config_file)
    config = ConfigSim.from_file(config_file, if_mkdir=False)
    # Enforce the batch size to 1
    config._config["nn"]["batch_size"] = 1
    device = torch.device("cuda:0" if args.cuda else "cpu")
    # device = "cpu"
    eval_envs = make_vec_envs(config,
                              args.env_name,
                              args.seed + 1000,
                              1,
                              None,
                              device, False, training=False, interactive=True)

    model_paths = glob.glob(os.path.join("saved_results", config_file.split('/')[1].split('.json')[0], "DiffTaichi_RL/*/models"), recursive=True)
    model_path = sorted(model_paths, key=os.path.getmtime)[-1]
    model_names = glob.glob(os.path.join(model_path, "*.pt"))
    model_name = sorted(model_names, key=os.path.getmtime)[-1]
    print("load from : ", model_name)
    actor_critic, obs_rms = torch.load(model_name)

    vec_norm = get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    ground_height = eval_envs.get_attr("ground_height", indices=0)[0]
    set_target()
    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(1, 1, device=device)

    visualizer()
    while gui.running:
        set_target()
        # Make decision
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)
        # Forward
        obs, _, done, infos = eval_envs.step(action)
        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)
        # refresh_xv()
        # offset += 1
        visualizer()
    eval_envs.close()



    # while gui.running:
    #     for i in range(10):
    #         set_target()
    #         make_decision()
    #         forward_mass_spring()
    #         refresh_xv()
    #         offset += 1
    #     visualizer()
