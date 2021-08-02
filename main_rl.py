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
        rl_trainer.envs.close()
    if args.evaluate:
        import glob
        exp_folders = glob.glob(os.path.join(args.evaluate_path, "*"))
        exp_folders = sorted(exp_folders, key=os.path.getmtime)
        print(exp_folders)
        paths_to_evaluate = []
        # Check whether this experiment has been evaluated before
        for ef in exp_folders:
            if len(os.listdir(os.path.join(ef, "validation"))) == 0:
                paths_to_evaluate.append(ef)
        print(f"All experiments to evaluate {paths_to_evaluate}")

        for ef in paths_to_evaluate:
            config_file = os.path.join(ef, "config.json")
            config = ConfigSim.from_args_and_file(args, config_file, if_mkdir=False)

            # Reset the number of processes to 1
            args.num_processes = 1
            process_required = 1
            for k, v in config.get_config()["validation"].items():
                if k not in config.get_config()["train"]["task"]:
                    continue
                process_required *= len(v)
            print(f"Processes required {process_required}")
            args.num_processes = process_required
            print(config)
            # global_buffer.initialize(config)
            rl_trainer = RLTrainer(args, config=config)
            rl_trainer.evaluate(ef)
