import os
from a2c_ppo_acktr.arguments import get_args
from multitask.config_sim import ConfigSim
from RL_trainer import RLTrainer

if __name__ == "__main__":
    args = get_args()
    curr_path = os.path.abspath(os.getcwd())
    config_file_name = os.path.join(curr_path, "cfg/sim_config_RL.json")
    config = ConfigSim.from_file(config_file_name)
    rl_trainer = RLTrainer(args, config=config)
    rl_trainer.train(start_iter=0, max_iter=100)

