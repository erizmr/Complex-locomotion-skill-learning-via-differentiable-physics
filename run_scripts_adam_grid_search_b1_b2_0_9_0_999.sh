#!/bin/bash
for i in 1 2 3;
do
python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot2_vhc_5_l01_adam_grid_search_b1_0_82_b2_0_9.json --train --gpu-id 0 --memory 2.0 --random
python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot2_vhc_5_l01_adam_grid_search_b1_0_82_b2_0_9.json --evaluate --gpu-id 0 --memory 8.0 --no-tensorboard-train --evaluate_path saved_results/sim_config_DiffPhy_robot2_vhc_5_l01_adam_grid_search_b1_0_82_b2_0_9/DiffTaichi_DiffPhy
done

for i in 1 2 3;
do
python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot2_vhc_5_l01_adam_grid_search_b1_0_82_b2_0_999.json --train --gpu-id 0 --memory 2.0 --random
python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot2_vhc_5_l01_adam_grid_search_b1_0_82_b2_0_999.json --evaluate --gpu-id 0 --memory 8.0 --no-tensorboard-train --evaluate_path saved_results/sim_config_DiffPhy_robot2_vhc_5_l01_adam_grid_search_b1_0_82_b2_0_999/DiffTaichi_DiffPhy
done

for i in 1 2 3;
do
python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot2_vhc_5_l01_adam_grid_search_b1_0_68_b2_0_9.json --train --gpu-id 0 --memory 2.0 --random
python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot2_vhc_5_l01_adam_grid_search_b1_0_68_b2_0_9.json --evaluate --gpu-id 0 --memory 8.0 --no-tensorboard-train --evaluate_path saved_results/sim_config_DiffPhy_robot2_vhc_5_l01_adam_grid_search_b1_0_68_b2_0_9/DiffTaichi_DiffPhy
done

for i in 1 2 3;
do
python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot2_vhc_5_l01_adam_grid_search_b1_0_68_b2_0_999.json --train --gpu-id 0 --memory 2.0 --random
python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot2_vhc_5_l01_adam_grid_search_b1_0_68_b2_0_999.json --evaluate --gpu-id 0 --memory 8.0 --no-tensorboard-train --evaluate_path saved_results/sim_config_DiffPhy_robot2_vhc_5_l01_adam_grid_search_b1_0_68_b2_0_999/DiffTaichi_DiffPhy
done

for i in 1 2 3;
do
python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot2_vhc_5_l01_adam_grid_search_b1_0_43_b2_0_9.json --train --gpu-id 0 --memory 2.0 --random
python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot2_vhc_5_l01_adam_grid_search_b1_0_43_b2_0_9.json --evaluate --gpu-id 0 --memory 8.0 --no-tensorboard-train --evaluate_path saved_results/sim_config_DiffPhy_robot2_vhc_5_l01_adam_grid_search_b1_0_43_b2_0_9/DiffTaichi_DiffPhy
done

for i in 1 2 3;
do
python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot2_vhc_5_l01_adam_grid_search_b1_0_43_b2_0_999.json --train --gpu-id 0 --memory 2.0 --random
python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot2_vhc_5_l01_adam_grid_search_b1_0_43_b2_0_999.json --evaluate --gpu-id 0 --memory 8.0 --no-tensorboard-train --evaluate_path saved_results/sim_config_DiffPhy_robot2_vhc_5_l01_adam_grid_search_b1_0_43_b2_0_999/DiffTaichi_DiffPhy
done

