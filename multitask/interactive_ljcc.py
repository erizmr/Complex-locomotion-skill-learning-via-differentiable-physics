import config
config.batch_size = 1
import multitask
import taichi as ti
import os

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
        elif e.key == gui.BACKSPACE:
            set_target.target_v = 0.
            set_target.target_h = 0.1
    print("Status: ", set_target.target_v, set_target.target_h)
    multitask.initialize_interactive(1, set_target.target_v, set_target.target_h)
set_target.target_v = 0
set_target.target_h = 0.1

def make_decision():
    multitask.nn.clear_single(0)
    multitask.solver.compute_center(0)
    multitask.nn_input(0, offset, 0.08, 0.1)
    multitask.nn.forward(0)

def forward_mass_spring():
    multitask.solver.apply_spring_force(0)
    multitask.solver.advance_toi(1)
    multitask.solver.clear_states(1)

@ti.kernel
def refresh_xv():
    for i in range(multitask.n_objects):
        multitask.x[0, 0, i] = multitask.x[1, 0, i]
        multitask.v[0, 0, i] = multitask.v[1, 0, i]

# TODO: clean up
gui = ti.GUI(background_color=0xFFFFFF)
def visualizer():
    gui.clear()
    gui.line((0, multitask.ground_height), (1, multitask.ground_height),
             color=0x000022,
             radius=3)
    multitask.solver.draw_robot(gui, 1, multitask.target_v)
    gui.show('video/interactive/{:04d}.png'.format(visualizer.frame))
    visualizer.frame += 1
visualizer.frame = 0

if __name__ == "__main__":
    robot_id = 5
    os.makedirs("video/interactive", exist_ok = True)
    multitask.setup_robot()
    multitask.nn.load_weights("saved_results/weight.pkl")
    print(multitask.x.to_numpy()[0, :, :])
    visualizer()
    while gui.running:
        for i in range(10):
            set_target()
            make_decision()
            forward_mass_spring()
            refresh_xv()
            offset += 1
        visualizer()
