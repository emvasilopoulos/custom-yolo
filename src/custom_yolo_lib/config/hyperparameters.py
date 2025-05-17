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


import custom_yolo_lib.experiments.model_factory
import custom_yolo_lib.experiments.optimizer_factory
import custom_yolo_lib.experiments.loss_factory
import custom_yolo_lib.experiments.dataloaders_factory
import custom_yolo_lib.experiments.schedulers_factory


@dataclasses.dataclass
class ThreeAnchorsHyperparameters:
    experiment_name: str
    model_type: custom_yolo_lib.experiments.model_factory.ModelType
    optimizer_type: custom_yolo_lib.experiments.optimizer_factory.OptimizerType
    loss_type: custom_yolo_lib.experiments.loss_factory.LossType
    dataset_type: custom_yolo_lib.experiments.dataloaders_factory.DatasetType
    scheduler_type: custom_yolo_lib.experiments.schedulers_factory.SchedulerType
    batch_size: int
    momentum: float
    weight_decay: float
    warmup_epochs: int
    epochs: int
    dataset_num_classes: int
    image_size: custom_yolo_lib.image_size.ImageSize
    class_loss_gain: float
    objectness_loss_gain: float
    box_loss_gain: float
    objectness_loss_small_map_gain: float
    objectness_loss_medium_map_gain: float
    objectness_loss_large_map_gain: float

    BASE_LR: float = 0.01 / 64

    def __post_init__(self):
        self.learning_rate = self.BASE_LR * self.batch_size
