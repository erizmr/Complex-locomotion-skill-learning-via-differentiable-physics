from multitask.utils import real
import taichi as ti
import time
from multitask.arguments import get_args
from multitask.config_sim import ConfigSim
from multitask.diffphy_trainer import DiffPhyTrainer


if __name__ == "__main__":
    # random_seed = int(time.time() * 1e6) % 10000
    # ti.init(arch=ti.gpu, default_fp=real, random_seed=random_seed)
    args = get_args()
    print('args', args)
    config_file = args.config_file
    if args.train:
        config = ConfigSim.from_file(config_file)
        print(config)
        diffphy_trainer = DiffPhyTrainer(args, config=config)
        ti.root.lazy_grad()
        diffphy_trainer.train(start_iter=0, max_iter=10000)
        # diffphy_trainer.optimize(loss_enable={"velocity", "height"}, root_dir="/home/mingrui/difftaichi/difftaichi2/saved_results")
        # diffphy_trainer.optimize(iters=35000, loss_enable={"velocity", "height", "actuation"}, root_dir="/home/mingrui/difftaichi/difftaichi2/saved_results")
    if args.evaluate:
        load_path = args.evaluate_path
        config = ConfigSim.from_file(config_file, if_mkdir=False)
        print(config)
        batch_required = 1
        for k, v in config.get_config()["validation"].items():
            if k not in config.get_config()["train"]["task"]:
                continue
            batch_required *= len(v)
        print(f"Batch required {batch_required}")
        config._config["nn"]["batch_size"] = batch_required
        diffphy_trainer = DiffPhyTrainer(args, config=config)
        diffphy_trainer.evaluate(load_path=load_path, custom_loss_enable={"velocity", "height", "crawl"})
        # diffphy_trainer.evaluate(load_path=load_path, custom_loss_enable={"velocity", "height"})
