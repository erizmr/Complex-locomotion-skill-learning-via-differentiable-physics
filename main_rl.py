import os
import sys
import time
sys.path.append("multitask")
from multitask.utils import real
import taichi as ti
from ppo.a2c_ppo_acktr.arguments import get_args
from multitask.config_sim import ConfigSim
from ppo.RL_trainer import RLTrainer

if __name__ == "__main__":
    # random_seed = int(time.time() * 1e6) % 10000
    # ti.init(arch=ti.gpu, default_fp=real, random_seed=random_seed)
    args = get_args()
    print('args', args)
    config_file = args.config_file

    if args.train:
        config = ConfigSim.from_args_and_file(args, config_file)
        print(config)
        rl_trainer = RLTrainer(args, config=config)
        rl_trainer.train(start_iter=0, max_iter=10000)
    if args.evaluate:
        config = ConfigSim.from_args_and_file(args, config_file, if_mkdir=False)
        process_required = 1
        for k, v in config.get_config()["validation"].items():
            if k not in config.get_config()["train"]["task"]:
                continue
            process_required *= len(v)
        print(f"Processes required {process_required}")
        config._config["train"]["num_processes"] = process_required // 2
        args.num_processes = process_required // 2
        print(config)
        rl_trainer = RLTrainer(args, config=config)
        eval_path = args.evaluate_path
        rl_trainer.evaluate(eval_path)
