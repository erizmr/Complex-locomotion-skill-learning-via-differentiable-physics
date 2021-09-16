# DiffTaichi2


## RL

### Train
`python3 main_rl.py --config_file cfg/sim_config_RL_robot2_vh.json --env-name "RL_Multitask" --algo ppo --use-gae --use-linear-lr-decay --train`

### Evaluate 
`python3 main_rl.py --env-name "RL_Multitask" --algo ppo --use-gae --use-linear-lr-decay --evaluate --evaluate_path "saved_results/sim_config_RL_robot2_vh/DiffTaichi_RL/" `

## DiffPhy

### Train
`python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot2_vh.json --train`

### Evaluate
`python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot2_vh.json --evaluate --no-tensorboard --evaluate_path saved_results/sim_config_DiffPhy_robot2_vh/DiffTaichi_DiffPhy`

###  Interactive
`python3 interactive.py --config_file cfg/sim_config_DiffPhy_robot2_vh.json --no-tensorboard`


## Output plots

### DiffPhy Only

#### Multi-Plots
`python3 scripts/export_tensorboard_data.py --our_file_path saved_results/sim_config_DiffPhy_robot4_vh/DiffTaichi_DiffPhy/ --no-rl --error-bar --task tvh`

#### Single-Plot
`python3 scripts/export_tensorboard_data.py --our_file_path saved_results/sim_config_DiffPhy_robot4_vh/DiffTaichi_DiffPhy/ --no-rl --error-bar --task tvh --draw-single`

### DiffPhy and RL

#### Multi-Plots
`python3 scripts/export_tensorboard_data.py --our_file_path saved_results/sim_config_DiffPhy_robot4_vh/DiffTaichi_DiffPhy/ --rl_file_path saved_results/sim_config_RL_robot4_vh/DiffTaichi_RL/ --task tvh`

#### Single-Plot
`python3 scripts/export_tensorboard_data.py --our_file_path saved_results/sim_config_DiffPhy_robot4_vh/DiffTaichi_DiffPhy/ --rl_file_path saved_results/sim_config_RL_robot4_vh/DiffTaichi_RL/ --task tvh --draw-single --error-bar`


## 3D
python3 main_diff_phy.py --config_file cfg3d/sim_config_DiffPhy_robot2_vh.json --train
python3 main_diff_phy.py --config_file cfg3d/sim_config_DiffPhy_robot2_vh.json --evaluate --no-tensorboard --evaluate_path saved_results/sim_config_DiffPhy_robot2_vh/DiffTaichi_DiffPhy
python3 interactive.py --config_file cfg3d/sim_config_DiffPhy_robot2_vh.json --no-tensorboard