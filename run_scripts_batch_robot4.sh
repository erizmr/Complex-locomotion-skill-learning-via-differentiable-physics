
#!/bin/bash

# 1
for i in 1 8 32 64;
do
echo $(expr $i);
python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot4_vh_batch_${i}.json --train
done

for i in 1 8 32 64;
do
echo $(expr $i);
python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot4_vh_batch_${i}.json --evaluate --no-tensorboard-train --evaluate_path saved_results/sim_config_DiffPhy_robot4_vh_batch_${i}/DiffTaichi_DiffPhy
done

# 2
for i in 1 8 32 64;
do
echo $(expr $i);
python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot4_vh_batch_${i}.json --train
done

for i in 1 8 32 64;
do
echo $(expr $i);
python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot4_vh_batch_${i}.json --evaluate --no-tensorboard-train --evaluate_path saved_results/sim_config_DiffPhy_robot4_vh_batch_${i}/DiffTaichi_DiffPhy
done

# 3
for i in 1 8 32 64;
do
echo $(expr $i);
python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot4_vh_batch_${i}.json --train
done

for i in 1 8 32 64;
do
echo $(expr $i);
python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot4_vh_batch_${i}.json --evaluate --no-tensorboard-train --evaluate_path saved_results/sim_config_DiffPhy_robot4_vh_batch_${i}/DiffTaichi_DiffPhy
done