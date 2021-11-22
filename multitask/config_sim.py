import os
import json
import time
import math
import logging

from datetime import datetime
from pathlib import Path
from multitask.robot_config import robots
from multitask.robot3d_config import robots3d
from multitask.robot_mpm import robots_mpm, n_grid, dx, n_squ
from multitask.robot_design import RobotDesignBase, RobotDesignMassSpring, RobotDesignMassSpring3D, RobotDesignMPM
from util import read_json, write_json
from logger import setup_logging


class ConfigSim:
    def __init__(self, config, args=None, file_name=None, if_mkdir=True):
        self._config = config
        self._args = args
        self.robot_builder = RobotDesignBase(cfg=None)
        self.file_name = file_name.split("/")[-1].split(".")[0]
        self._add_adaptive_configs()
        save_dir = self._config["train"]["save_dir"]
        save_dir = os.path.join(save_dir, self.file_name)
        self._config["train"]["save_dir"] = save_dir
        # os.makedirs(self._config["train"]["save_dir"], exist_ok=True)

        exper_name = self._config["train"]["name"]
        save_dir = Path(save_dir)
        run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        self._save_dir = save_dir / exper_name / run_id
        self._model_dir = self.save_dir / 'models'
        self._video_dir = self._save_dir / 'video'
        self._validation_dir = self._save_dir / 'validation'
        self._monitor_dir = self._save_dir / 'monitor'
        self._log_dir = self._save_dir / 'log'

        if if_mkdir:
            # make directory for saving checkpoints and log.
            exist_ok = run_id == ''
            self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
            self._model_dir.mkdir(parents=True, exist_ok=exist_ok)
            self.log_dir.mkdir(parents=True, exist_ok=exist_ok)
            self.monitor_dir.mkdir(parents=True, exist_ok=exist_ok)
            self.video_dir.mkdir(parents=True, exist_ok=exist_ok)
            self.validation_dir.mkdir(parents=True, exist_ok=exist_ok)

            # save updated config file to the checkpoint dir
            write_json(self.config, self.save_dir / 'config.json')
            # save robot design to the checkpoint dir
            self.robot_builder.dump_to_json(file=self.save_dir / 'robot_design.json')

        # configure logging module
        # setup_logging(save_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }
        self.logger = None

    def _add_adaptive_configs(self):
        # Robot
        robot_id = self._config["robot"]["robot_id"]
        faces = []
        robot_dim = self._config["robot"]["dim"]
        robot_design_file = self._config["robot"]["design_file"]
        solver_type = self._config["robot"]["simulator"]
        if solver_type == "mass_spring" and robot_dim == 2:
            self.robot_builder = RobotDesignMassSpring.from_file(file_name=robot_design_file)
            objects, springs = self.robot_builder.build()
        elif solver_type == "mass_spring" and robot_dim == 3:
            self.robot_builder = RobotDesignMassSpring3D.from_file(file_name=robot_design_file)
            objects, springs, faces = self.robot_builder.build()
        elif solver_type == "mpm":
            self.robot_builder = RobotDesignMPM.from_file(file_name=robot_design_file)
            objects, springs, n_springs = self.robot_builder.build()
            # objects, springs, n_springs = robots_mpm[0]()
        else:
            raise NotImplementedError(f"{solver_type} not implemented.")

        # if robot_id >= 10000:
        #     self._config["robot"]["simulator"] = "mpm"
        #     self._config["robot"]["dim"] = 2
        #     objects, springs, n_springs = robots_mpm[robot_id - 10000]()
        #     n_objects = len(objects)
        # else:
        #     self._config["robot"]["simulator"] = "mass_spring"
        #     if robot_id < 100:
        #         self._config["robot"]["dim"] = 2
        #         objects, springs = robots[robot_id]()
        #
        #     else:
        #         self._config["robot"]["dim"] = 3
        #         objects, springs, faces = robots3d[robot_id - 100]()
        #     n_objects = len(objects)
        #     n_springs = len(springs)

        n_objects = len(objects)
        n_springs = len(springs)
        self._config["robot"]["objects"] = objects
        self._config["robot"]["springs"] = springs
        self._config["robot"]["faces"] = faces
        self._config["robot"]["n_objects"] = n_objects
        self._config["robot"]["n_springs"] = n_springs

        if solver_type == "mpm":
            self._config["robot"]["n_squ"] = n_squ
            self._config["robot"]["n_squares"] = n_objects // (n_squ ** 2)

        # Process
        self._config["process"]["dt"] = 0.004 if self._config["robot"][
            "simulator"] == "mass_spring" else 0.001
        self._config["process"]["spring_omega"] = 2 * math.pi / self._config[
            "process"]["dt"] / self._config["process"]["run_period"]

        # Simulator
        self._config["simulator"]["gravity"] = -1.8 if self._config["robot"][
            "simulator"] == "mass_spring" else -10.
        self._config["simulator"]["dashpot_damping"] = 0.2 if self._config[
            "robot"]["dim"] == 2 else 0.1

        self._config["simulator"]["n_particles"] = self._config["robot"][
            "n_objects"]

        self._config["simulator"]["mpm"][
            "dx"] = 1 / self._config["simulator"]["mpm"]["n_grid"]
        self._config["simulator"]["mpm"][
            "inv_dx"] = 1 / self._config["simulator"]["mpm"]["dx"]

        # NN
        if self._config["robot"]["dim"] == 3:
            self._config["nn"]["duplicate_v"] = 1
            self._config["nn"]["duplicate_h"] = 0

        n_sin_waves = self._config["nn"]["n_sin_waves"]
        dim = self._config["robot"]["dim"]
        n_objects = self._config["robot"]["n_objects"]
        duplicate_h = self._config["nn"]["duplicate_h"]
        duplicate_v = self._config["nn"]["duplicate_v"]
        duplicate_c = self._config["nn"]["duplicate_c"]
        duplicate_o = self._config["nn"]["duplicate_o"] if "duplicate_o" in self._config["nn"] else 0

        if solver_type == "mass_spring":
            self._config["nn"][
                "n_input_states"] = n_sin_waves + dim * 2 * n_objects + duplicate_v * (
                    dim - 1) + duplicate_h + duplicate_c
        elif solver_type == "mpm":
            n_squares = self._config["robot"]["n_squares"]
            # self._config["nn"]["n_input_states"] = n_sin_waves + dim * 2 * n_squares + duplicate_v * (dim - 1) + duplicate_h + duplicate_c
            self._config["nn"]["n_input_states"] = n_sin_waves + dim * 2 * n_squares + duplicate_v * (
                        dim - 1) + duplicate_h + duplicate_c + duplicate_o * dim
        else:
            raise NotImplementedError(f"Solver {solver_type} not implemented.")

        self._config["nn"]["adam_a"] = self._config["nn"]["learning_rate"]

        if self._args:
            self._config["train"]["random_seed"] = self._args.seed
            self._config["train"]["num_processes"] = self._args.num_processes

    @classmethod
    def from_file(cls, file_name, if_mkdir=True):
        config = read_json(file_name)
        return cls(config, file_name=file_name, if_mkdir=if_mkdir)

    @classmethod
    def from_args_and_file(cls, args, file_name, if_mkdir=True):
        config = read_json(file_name)
        return cls(config, args=args, file_name=file_name, if_mkdir=if_mkdir)

    def get_config(self):
        return self._config

    def get_logger(self, name, verbosity=2):
        if self.logger:
            return self.logger
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(
            verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity

        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s'
        )

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)
        logger.propagate = False
        self.logger = logger
        return self.logger

    def print_config(self):
        return json.dumps(self._config, indent=4, sort_keys=True)

    def __str__(self):
        return self.print_config()

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def model_dir(self):
        return self._model_dir

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def monitor_dir(self):
        return self._monitor_dir

    @property
    def video_dir(self):
        return self._video_dir

    @property
    def validation_dir(self):
        return self._validation_dir
