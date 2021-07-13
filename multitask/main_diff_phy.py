from utils import real
import taichi as ti
from arguments import get_args
from config_sim import ConfigSim
from multitask_obj import DiffPhyTrainer


if __name__ == "__main__":
    ti.init(arch=ti.gpu, default_fp=real, random_seed=555)
    args = get_args()
    print('args', args)
    config_file = args.config_file
    config = ConfigSim.from_file(config_file)
    diffphy_trainer = DiffPhyTrainer(args, config=config)
    ti.root.lazy_grad()
    print(config)
    if args.train:
        diffphy_trainer.train(start_iter=0, max_iter=10000 * config.get_config()["nn"]["batch_size"])
        # diffphy_trainer.optimize(loss_enable={"velocity", "height"}, root_dir="/home/mingrui/difftaichi/difftaichi2/saved_results")
        # diffphy_trainer.optimize(iters=35000, loss_enable={"velocity", "height", "actuation"}, root_dir="/home/mingrui/difftaichi/difftaichi2/saved_results")

    # if args.validate:
    #     diffphy_trainer .validate()
    # if args.evaluate:
    #     eval_path = args.evaluate_path
    #     diffphy_trainer .evaluate(eval_path)
