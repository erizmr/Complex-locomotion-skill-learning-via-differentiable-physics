import gym
from gym import spaces

from config import *
from nn import *
from solver_mass_spring import SolverMassSpring

import sys
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import taichi as ti

class MassSpringEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(MassSpringEnv, self).__init__()