import dataclasses
import pathlib

import custom_yolo_lib.image_size
import custom_yolo_lib.config.utils


@dataclasses.dataclass
class HyperparametersConfiguration:
    image_size: custom_yolo_lib.image_size.ImageSize
    batch_size: int
    epochs: int
    optimizer: str
    learning_rate: float

    @staticmethod
    def from_file(file_path: pathlib.Path) -> "HyperparametersConfiguration":
        hyperparameters_configuration = custom_yolo_lib.config.utils.read_yaml(
            file_path
        )
        return HyperparametersConfiguration.from_dict(hyperparameters_configuration)

    @staticmethod
    def from_dict(
        hyperparameters_configuration: dict,
    ) -> "HyperparametersConfiguration":
        return HyperparametersConfiguration(
            custom_yolo_lib.image_size.ImageSize.from_dict(
                hyperparameters_configuration["input_image_size"]
            ),
            hyperparameters_configuration["batch_size"],
            hyperparameters_configuration["epochs"],
            hyperparameters_configuration["optimizer"],
            hyperparameters_configuration["learning_rate"],
        )
