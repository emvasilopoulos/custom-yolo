import dataclasses
import pathlib

import custom_yolo_lib.config.utils


@dataclasses.dataclass
class OptimizerConfiguration:
    name: str
    learning_rate: float
    momentum: float = 0.9
    weight_decay: float = 0.0

    @staticmethod
    def from_file(file_path: pathlib.Path) -> "OptimizerConfiguration":
        optimizer_configuration = custom_yolo_lib.config.utils.read_yaml(file_path)
        return OptimizerConfiguration.from_dict(optimizer_configuration)

    @staticmethod
    def from_dict(optimizer_configuration: dict) -> "OptimizerConfiguration":
        return OptimizerConfiguration(
            optimizer_configuration["name"], optimizer_configuration["learning_rate"]
        )
