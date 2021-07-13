from robot_config import robots
from robot3d_config import robots3d
from robot_mpm import robots_mpm, n_grid, dx

import sys
import math

# robot------------------------------------------------------------
robot_id = 2

if len(sys.argv) >= 2:
    robot_id = 5
    print("Run robot", robot_id)
simulator = ""
if robot_id >= 10000:
    simulator = "mpm"
    dim = 2
    objects, springs, n_springs = robots_mpm[robot_id - 10000]()
    n_objects = len(objects)
else:
    simulator = "mass_spring"
    if robot_id < 100:
        dim = 2
        objects, springs = robots[robot_id]()
    else:
        dim = 3
        objects, springs, faces = robots3d[robot_id - 100]()
    n_objects = len(objects)
    n_springs = len(springs)

# process------------------------------------------------------------
max_steps = 1000
if dim == 3 and sys.argv[0] == "validate.py":
    max_steps = 4050
vis_interval = 256
output_vis_interval = 8

run_period = 100
jump_period = 500
turn_period = 500

max_speed = 0.08
max_height = 0.1

dt = 0.004 if simulator == "mass_spring" else 0.001

spring_omega = 2 * math.pi / dt / run_period

# simulator----------------------------------------------------------
# target_ball = 0
ground_height = 0.1
gravity = -1.8 if simulator == "mass_spring" else -10.

drag_damping = 0
dashpot_damping = 0.2 if dim == 2 else 0.1

n_particles = n_objects
inv_dx = 1 / dx
p_vol = 1
E, mu, la = 40, 40, 40
act_strength = 40

bound = 3
coeff = 0.5

# nn------------------------------------------------------------
n_sin_waves = 10
n_hidden = 64
duplicate_v = 30
duplicate_h = 30
if dim == 3:
    duplicate_v = 1
    duplicate_h = 0

n_input_states = n_sin_waves + dim * 2 * n_objects + duplicate_v * (dim - 1) + duplicate_h

batch_size = 1
if sys.argv[0] in ["interactive_ljcc.py", "multitask_rl.py"]:
    batch_size = 1

max_reset_step = 4

#weight_decay = 0.001
learning_rate = 0.5 * 3e-4

adam_a = learning_rate
adam_b1=0.90
adam_b2=0.90
