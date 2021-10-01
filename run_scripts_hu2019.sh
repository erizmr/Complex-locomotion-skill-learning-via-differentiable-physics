
#!/bin/bash

python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot2_vhc_5_l01_sgd.json --train
python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot2_vhc_5_l01_sgd.json --evaluate --no-tensorboard--train --evaluate_path saved_results/sim_config_DiffPhy_robot2_vhc_5_l01_sgd/DiffTaichi_DiffPhy

python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot2_vhc_5_l01_sgd.json --train
python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot2_vhc_5_l01_sgd.json --evaluate --no-tensorboard--train --evaluate_path saved_results/sim_config_DiffPhy_robot2_vhc_5_l01_sgd/DiffTaichi_DiffPhy

python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot2_vhc_5_l01_sgd.json --train
python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot2_vhc_5_l01_sgd.json --evaluate --no-tensorboard--train --evaluate_path saved_results/sim_config_DiffPhy_robot2_vhc_5_l01_sgd/DiffTaichi_DiffPhy
