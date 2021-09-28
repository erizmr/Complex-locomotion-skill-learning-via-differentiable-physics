
#!/bin/bash


for i in 1 8 16;
do
echo $(expr $i);
python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot2_vhc_5_l10_batch_${i}.json --train
done

for i in 1 8 16;
do
echo $(expr $i);
python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot2_vhc_5_l10_batch_${i}.json --evaluate --no-tensorboard --evaluate_path saved_results/sim_config_DiffPhy_robot2_vhc_5_l10_batch_${i}/DiffTaichi_DiffPhy
done

