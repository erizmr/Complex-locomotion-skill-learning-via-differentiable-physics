import multitask
import taichi as ti

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
set_target.target_v = 0.
set_target.target_h = 0.1

def make_decision():
    multitask.nn.clear_single(0)
    multitask.compute_center(0)
    multitask.nn_input(0)
    multitask.nn.forward(0)

def forward_mass_spring():
    multitask.apply_spring_force(0)
    multitask.advance_toi(1)
    multitask.clear_states(1)

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
    def circle(x, y, color):
        gui.circle((x, y), ti.rgb_to_hex(color), 7)
    for i in range(multitask.n_springs):
        def get_pt(x):
            return (x[0], x[1])
        a = multitask.actuation[0, 0, i] * 0.5
        r = 2
        if multitask.spring_actuation[i] == 0:
            a = 0
            c = 0x222222
        else:
            r = 4
            c = ti.rgb_to_hex((0.5 + a, 0.5 - abs(a), 0.5 - a))
        gui.line(get_pt(multitask.x[0, 0, multitask.spring_anchor_a[i]]),
                 get_pt(multitask.x[0, 0, multitask.spring_anchor_b[i]]),
                 color=c,
                 radius=r)
    for i in range(multitask.n_objects):
        color = (0.06640625, 0.06640625, 0.06640625)
        circle(multitask.x[0, 0, i][0], multitask.x[0, 0, i][1], color)
    gui.show('mass_spring/{:04d}.png'.format(visualizer.frame))
    visualizer.frame += 1
visualizer.frame = 0

if __name__ == "__main__":
    multitask.setup_robot()
    multitask.nn.load_weights("weights/best.pkl")
    print(multitask.x.to_numpy()[0, :, :])
    visualizer()
    while gui.running:
        set_target()
        make_decision()
        forward_mass_spring()
        refresh_xv()
        visualizer()
