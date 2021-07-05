# DiffTaichi2



## Train

`python3 pytorch-a2c-ppo-acktr-gail/main_rl.py --env-name "RL_Multitask" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 1 --num-step 1000 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --save-interval 50 --train`

## Evaluate 
` python3 pytorch-a2c-ppo-acktr-gail/main_rl.py --env-name "RL_Multitask" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 1 --num-step 1000 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --save-interval 50 --evaluate --evaluate_path "saved_results/sim_config_RL/DiffTaichi_RL/0702_011543" `


