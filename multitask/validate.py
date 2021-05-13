import multitask
import os

os.system("rm mass_spring/* -r")

multitask.setup_robot()
multitask.nn.load_weights("weights/best.pkl")
multitask.validate(4000)

import video_gen
