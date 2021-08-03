#!/bin/bash
for i in 2 3 4 5;
do
echo $(expr $i);
rm -r /tmp/taichi-*
python3 main_rl.py --env-name "RL_Multitask" --algo ppo --use-gae --use-linear-lr-decay --evaluate --evaluate_path saved_results/sim_config_RL_robot${i}_vh/DiffTaichi_RL/ --no-tensorboard
done

rm -r /tmp/taichi-*






