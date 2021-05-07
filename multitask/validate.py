import multitask
import os

os.system("rm mass_spring/* -r")

multitask.setup_robot()
multitask.load_weights("weights/best.pkl")
multitask.validate()

import video_gen
