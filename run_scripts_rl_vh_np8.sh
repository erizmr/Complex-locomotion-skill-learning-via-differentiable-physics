#!/bin/bash
for i in $(seq 2 5)
do
echo $(expr $i);
rm -r /tmp/taichi-*
python3 main_rl.py --config_file cfg/sim_config_RL_robot${i}_vh.json --env-name "RL_Multitask_vh" --algo ppo --use-gae --use-linear-lr-decay --train --num-processes 8

done






