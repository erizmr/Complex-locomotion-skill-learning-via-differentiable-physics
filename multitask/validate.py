import multitask
import os

os.system("rm video/* -r")

multitask.setup_robot()
multitask.nn.load_weights("weights.pkl")

def validate(steps):
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

validate(1000)

import video_gen
