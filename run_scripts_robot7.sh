#!/bin/bash

# Good one
for i in 1 2 3;
do
python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot7_vhc_5_l01.json --train
python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot7_vhc_5_l01.json --evaluate --no-tensorboard-train --evaluate_path saved_results/sim_config_DiffPhy_robot7_vhc_5_l01/DiffTaichi_DiffPhy
done

