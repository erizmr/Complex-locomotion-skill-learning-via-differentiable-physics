import taichi as ti
import gym
from stable_baselines3 import PPO
import torch
from tqdm import tqdm

import config
config.max_steps = 4050

import multitask_rl
import multitask

gui = ti.GUI(background_color=0xFFFFFF)
def visualizer(t):
    gui.clear()
    gui.line((0, multitask.ground_height), (1, multitask.ground_height),
             color=0x000022,
             radius=3)
    multitask.solver.draw_robot(gui, t, multitask.target_v)
    gui.show()

import sys
config.robot_id = sys.argv[1]
multitask.setup_robot()
env = multitask_rl.MassSpringEnv(multitask.solver.act_list, "../../")
model = PPO.load("../../../best_model.zip", env)

multitask.initialize_validate(1000, 0.04, 0.05)
obs = env.reset()
for i in tqdm(range(1000)):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    #if done:
     #   obs = env.reset()
for i in range(1000):
    if i % 1 == 0:
        visualizer(i)
env.close()
