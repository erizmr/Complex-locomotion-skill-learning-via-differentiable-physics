import config
config.max_steps = 4050
import multitask
import multitask_rl
from stable_baselines3 import PPO
import os
import taichi as ti
import pickle as pkl
import sys
import matplotlib.pyplot as plt
import time

total_task_num = 27
v_num = 9
p_num = 3

def initialize_task(task_id):
    velocities = [-0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
    heights = [0.1, 0.15, 0.1]
    crawlings = [0., 0., 1.]
    multitask.initialize_validate(config.max_steps, velocities[task_id % v_num], heights[task_id // v_num])

if __name__ == "__main__":
    interval = 50
    task_iter = [[] for _ in range(total_task_num)]
    task_loss = [[] for _ in range(total_task_num)]
    cnt = 0
    start_time = time.time()

    if len(sys.argv) == 2:
        # os.system('rm "{}" -r'.format('validate_plots/' + str(config.robot_id)))
        for stage in ['walk', 'jump', 'crawl']:
            for iter_num in range(50, 40000, 50):
                weight_path = '../../remote_result_rl/rl_robot_' + str(config.robot_id) + '/model_' + str(iter_num) +'.zip'
                if not os.path.exists(weight_path):
                    continue
                multitask.setup_robot()
                print("load from {}".format(weight_path))
                env = multitask_rl.MassSpringEnv(multitask.solver.act_list, '../../remote_result_rl_video/')
                model = PPO.load(weight_path, env, device = "cuda")

                for task_id in range(total_task_num):
                    initialize_task(task_id)
                    multitask.solver.clear_states(config.max_steps)

                    multitask.loss[None] = 0.
                    for l in multitask.losses:
                        l[None] = 0.

                    obs = env.reset()
                    for t in range(0, config.max_steps - 1):
                        action, _states = model.predict(obs, deterministic=True)
                        obs, reward, done, info = env.step(action)

                    multitask.get_loss(config.max_steps + 1, loss_enable = {"crawl", "walk", "jump"})
                    cnt += 1
                    print('{:.2f}%  ({:.2f})'.format(cnt / 3 / 200 / total_task_num * 100, time.time() - start_time), stage, iter_num, task_id, multitask.loss[None])
                    task_iter[task_id].append(iter_num)
                    task_loss[task_id].append(multitask.loss[None])
        os.makedirs('validate_plots/' + str(config.robot_id), exist_ok = True)
        pkl.dump([task_iter, task_loss], open('validate_plots/' + str(config.robot_id) + '/data.bin', 'wb'))
    else:
        [task_iter, task_loss] = pkl.load(open('validate_plots/' + str(config.robot_id) + '/data.bin', 'rb'))
        task_iter = []
        for iter_num in range(0, 30000, 50):
            task_iter.append(iter_num)
        fig, ax = plt.subplots(nrows=p_num, ncols=v_num, figsize=(v_num * 3, p_num * 3))
        for i in range(p_num):
            for j in range(v_num):
                velocities = [-0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
                heights = [0.1, 0.15, 0.1]
                crawlings = [0., 0., 1.]
                label = str(velocities[j]) + "_" + str(heights[i]) + "_" + str(crawlings[i])
                ax[i, j].plot(task_iter, task_loss[i * v_num + j], label=label)
                ax[i, j].legend()
        plt.show()

        avg_loss = [0 for _ in range(len(task_loss[0]))]
        for i in range(len(task_loss[0])):
            for j in range(total_task_num):
                avg_loss[i] += task_loss[j][i]
        plt.plot(task_iter, avg_loss)
        plt.show()