#!/bin/bash
for i in 5 4 3 2;
do
echo $(expr $i);
python3 main_rl.py --config_file cfg/sim_config_RL_robot${i}.json --env-name "RL_Multitask" --algo ppo --use-gae --use-linear-lr-decay --evaluate --evaluate_path saved_results/sim_config_RL_robot${i}/DiffTaichi_RL --no-tensorboard

done






