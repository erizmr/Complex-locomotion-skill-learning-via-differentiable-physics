#!/bin/bash
for i in $(seq 2 5)  
do  
echo $(expr $i);
python3 multitask/main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot${i}_max_h.json --train
done


for i in $(seq 2 5)
do
echo $(expr $i);
python3 multitask/main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot${i}_jump_test_max_h.json --train
done


for i in $(seq 2 5)
do
echo $(expr $i);
python3 multitask/main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot${i}_max_h.json --evaluate_path saved_results/sim_config_DiffPhy_robot${i}_max_h/DiffTaichi_DiffPhy --evaluate --no-tensorboard
python3 multitask/main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot${i}_jump_test_max_h.json --evaluate_path saved_results/sim_config_DiffPhy_robot${i}_jump_test_max_h/DiffTaichi_DiffPhy --evaluate --no-tensorboard
done

