import taichi as ti
import gym
from stable_baselines3 import PPO
import torch

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

env = multitask_rl.MassSpringEnv()
model = PPO.load("./log/best_model.zip", env)

multitask.initialize_validate(1000, 0.08, 0.1)
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if i % 1 == 0:
        visualizer(env.t)
    if done:
        obs = env.reset()
env.close()
