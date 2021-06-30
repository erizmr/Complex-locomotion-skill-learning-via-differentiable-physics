import os

from multitask.robot_config import robots
from multitask.robot3d_config import robots3d
from multitask.robot_mpm import robots_mpm, n_grid, dx

from util import read_json, write_json
import sys
import math


class ConfigSim:
    def __init__(self, config, file_name=None):
        self._config = config
        self.file_name = file_name.split("/")[-1].split(".")[0]
        self._add_adaptive_configs()
        save_dir = self._config["train"]["save_dir"]
        save_dir = os.path.join(save_dir, self.file_name)
        self._config["train"]["save_dir"] = save_dir
        os.makedirs(self._config["train"]["save_dir"], exist_ok=True)
        write_json(self._config, os.path.join(save_dir, "config.json"))

    def _add_adaptive_configs(self):
        # Robot
        robot_id = self._config["robot"]["robot_id"]
        faces = []
        if robot_id > 10000:
            self._config["robot"]["simulator"] = "mpm"
            self._config["robot"]["dim"] = 2
            objects, springs, n_springs = robots_mpm[robot_id - 10000]()
            n_objects = len(objects)
        else:
            self._config["robot"]["simulator"] = "mass_spring"
            if robot_id < 100:
                self._config["robot"]["dim"] = 2
                objects, springs = robots[robot_id]()
            else:
                self._config["robot"]["dim"] = 3
                objects, springs, faces = robots3d[robot_id - 100]()
            n_objects = len(objects)
            n_springs = len(springs)

        self._config["robot"]["objects"] = objects
        self._config["robot"]["springs"] = springs
        self._config["robot"]["faces"] = faces
        self._config["robot"]["n_objects"] = n_objects
        self._config["robot"]["n_springs"] = n_springs

        # Process
        self._config["process"]["dt"] = 0.004 if self._config["robot"]["simulator"] == "mass_spring" else 0.001
        self._config["process"]["spring_omega"] = 2 * math.pi / self._config["process"]["dt"] / self._config["process"]["run_period"]

        # Simulator
        self._config["simulator"]["gravity"] = -1.8 if self._config["robot"]["simulator"] == "mass_spring" else -10.
        self._config["simulator"]["dashpot_damping"] = 0.2 if self._config["robot"]["dim"] == 2 else 0.1

        self._config["simulator"]["n_particles"] = self._config["robot"]["n_objects"]
        self._config["simulator"]["inv_dx"] = 1 / self._config["simulator"]["mpm"]["n_grid"]

        # NN
        if self._config["robot"]["dim"] == 3:
            self._config["nn"]["duplicate_v"] = 1
            self._config["nn"]["duplicate_h"] = 0

        n_sin_waves = self._config["nn"]["n_sin_waves"]
        dim = self._config["robot"]["dim"]
        n_objects = self._config["robot"]["n_objects"]
        duplicate_h = self._config["nn"]["duplicate_h"]
        duplicate_v = self._config["nn"]["duplicate_v"]

        self._config["nn"]["n_input_states"] = n_sin_waves + dim * 2 * n_objects + duplicate_v * (dim - 1) + duplicate_h

        self._config["nn"]["adam_a"] = self._config["nn"]["learning_rate"]

    @classmethod
    def from_file(cls, file_name):
        config = read_json(file_name)
        return cls(config, file_name)

    def get_config(self):
        return self._config
