#!/bin/bash


for i in 2 3 4 5;
do
echo $(expr $i);
python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot${i}_vhc_5_l01_naive_loss.json --train
done

for i in 2 3 4 5;
do
echo $(expr $i);
python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot${i}_vhc_5_l01_naive_loss.json --evaluate --no-tensorboard-train --evaluate_path saved_results/sim_config_DiffPhy_robot${i}_vhc_5_l01_naive_loss/DiffTaichi_DiffPhy
done

