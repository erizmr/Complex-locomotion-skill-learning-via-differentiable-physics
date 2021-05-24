import config
config.batch_size = 1
import multitask
import os
import taichi as ti
import pickle as pkl
import sys
import matplotlib.pyplot as plt

total_task_num = 27
v_num = 9
p_num = 3

@ti.kernel
def initialize_task(task_id: ti.template()):
    velocities = ti.Vector([-0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04])
    heights = ti.Vector([0.1, 0.15, 0.1])
    crawlings = ti.Vector([0., 0., 1.])
    for t, k in ti.ndrange(config.max_steps, config.batch_size):
        multitask.target_v[t, k][0] = velocities[task_id % v_num]
        multitask.target_h[t, k] = heights[(task_id // v_num) % p_num]
        multitask.target_c[t, k] = crawlings[(task_id // v_num) % p_num]

if __name__ == "__main__":
    interval = 50
    task_iter = [[] for _ in range(total_task_num)]
    task_loss = [[] for _ in range(total_task_num)]
    cnt = 0

    if len(sys.argv) == 2:
        # os.system('rm "{}" -r'.format('validate_plots/' + str(config.robot_id)))
        for stage in ['walk', 'jump', 'crawl']:
            for iter_num in range(0, 10000, 50):
                weight_path = '../../remote_result/robot_' + str(config.robot_id) + '/weights/' + stage + '/iter' + str(iter_num) +'.pkl'
                multitask.setup_robot()
                print("load from {}".format(weight_path))
                multitask.nn.load_weights(weight_path)

                for task_id in range(total_task_num):
                    initialize_task(task_id)
                    multitask.nn.clear()
                    multitask.solver.clear_states(config.max_steps)

                    multitask.loss[None] = 0.
                    for l in multitask.losses:
                        l[None] = 0.

                    for t in range(0, config.max_steps - 1):
                        multitask.nn_input(t, 0, config.max_speed, config.max_height)
                        multitask.nn.forward(t)
                        multitask.solver.advance(t)

                    multitask.get_loss(config.max_steps + 1, loss_enable = {"crawl", "walk", "jump"})
                    cnt += 1
                    print('{:.2f}%  '.format(cnt / 3 / 200 / total_task_num * 100), stage, iter_num, task_id, multitask.loss[None])
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