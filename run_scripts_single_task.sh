#!/bin/bash


for i in 2 3 4 5;
do
echo $(expr $i);
  for t in 'v' 'h' ''c;
  do
    echo $(expr $t)
    python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot${i}_${t}_l10.json --train
  done
done


for i in 2 3 4 5;
do
echo $(expr $i);
  for t in 'v' 'h' ''c;
  do
    echo $(expr $t)
    python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot${i}_${t}_l10.json --evaluate --no-tensorboard --evaluate_path saved_results/sim_config_DiffPhy_robot${i}_${t}_l10/DiffTaichi_DiffPhy
  done
done


