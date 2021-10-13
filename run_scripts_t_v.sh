#!/bin/bash

for i in 2 3 4 5;
do
echo $(expr $i);
python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot${i}_vh.json --train
python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot${i}_vh.json --evaluate --no-tensorboard --evaluate_path saved_results/sim_config_DiffPhy_robot${i}_vh/DiffTaichi_DiffPhy
done

