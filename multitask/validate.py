import multitask

multitask.setup_robot()
multitask.load_weights("weights/best.pkl")
multitask.validate()

import video_gen