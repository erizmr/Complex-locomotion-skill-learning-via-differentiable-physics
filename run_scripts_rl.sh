#!/bin/bash
for i in $(seq 2 5)
do
echo $(expr $i);
python3 main_rl.py --config_file cfg/sim_config_RL_robot${i}.json --env-name "RL_Multitask" --algo ppo --use-gae --use-linear-lr-decay --train

done






