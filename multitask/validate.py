import multitask
import os
from config import *

os.system("rm video/* -r")

root_dir = "robot_{}".format(robot_id)
multitask.setup_robot()
multitask.nn.load_weights(root_dir + "/weight.pkl")

def validate(steps):
    if multitask.dim == 2:
        '''
        multitask.simulate(steps, 0.08, 0.1, train = False)
        multitask.simulate(steps, 0.06, 0.1, train = False)
        multitask.simulate(steps, 0.04, 0.1, train = False)
        multitask.simulate(steps, 0.02, 0.1, train = False)
        multitask.simulate(steps, 0., 0.1, train = False)
        '''

        multitask.simulate(steps, 0, 0.15, train = False)
        multitask.simulate(steps, 0, 0.125, train = False)
        multitask.simulate(steps, 0, 0.10, train = False)
        '''
        multitask.simulate(steps, 0.06, 0.15, train = False)
        multitask.simulate(steps, 0.04, 0.15, train = False)
        multitask.simulate(steps, 0.02, 0.15, train = False)
        '''

        # multitask.simulate(steps, 0, 0.25)
        # multitask.simulate(steps, 0, 0.3)
        # multitask.simulate(steps, 0, 0)
    else:
        multitask.simulate(steps, 0, 0, train = False)

validate(4000)

import video_gen
