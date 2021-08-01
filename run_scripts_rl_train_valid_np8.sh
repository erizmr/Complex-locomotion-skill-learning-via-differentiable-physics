#!/bin/bash
for i in $(seq 2 5)
do
echo $(expr $i);
rm -r /tmp/taichi-*
python3 main_rl.py --config_file cfg/sim_config_RL_robot${i}_vh.json --env-name "RL_Multitask" --algo ppo --use-gae --use-linear-lr-decay --train --num-processes 8
done


for i in 2 3 4 5;
do
echo $(expr $i);
rm -r /tmp/taichi-*
python3 main_rl.py --env-name "RL_Multitask" --algo ppo --use-gae --use-linear-lr-decay --evaluate --evaluate_path saved_results/sim_config_RL_robot${i}_vh/DiffTaichi_RL/ --no-tensorboard
done






