import pathlib

import torch

import custom_yolo_lib.experiments.dataloaders_factory
import custom_yolo_lib.experiments.loss_factory
import custom_yolo_lib.experiments.model_factory
import custom_yolo_lib.experiments.optimizer_factory
import custom_yolo_lib.experiments.schedulers_factory
import custom_yolo_lib.experiments.train_val_sessions.three_scales
import custom_yolo_lib.image_size
import custom_yolo_lib.config.hyperparameters
import custom_yolo_lib.experiments.train_val_sessions.args

torch.manual_seed(42)

# do a fast forward model to check if augmentations do extra work
MODEL_TYPE = custom_yolo_lib.experiments.model_factory.ModelType.YOLO
OPTIMIZER_TYPE = (
    custom_yolo_lib.experiments.optimizer_factory.OptimizerType.SPLIT_GROUPS_ADAMW
)
LOSS_TYPE = custom_yolo_lib.experiments.loss_factory.LossType.THREESCALE_YOLO_ORD_v3
DATASET_TYPE = (
    custom_yolo_lib.experiments.dataloaders_factory.DatasetType.COCO_SAMA_AUGMENT
)
SCHEDULER_TYPE = (
    custom_yolo_lib.experiments.schedulers_factory.SchedulerType.WARMUP_COSINE_10_CYCLES  # TOO MANY CYCLES
)
BASE_LR = 0.01 / 64
EXPERIMENT_NAME = "exp6"
WARMUP_EPOCHS = 3
EPOCHS = 100
NUM_CLASSES = 80
BATCH_SIZE = 16
LR = BASE_LR * BATCH_SIZE
MOMENTUM = 0.9
DECAY = 5e-4
IMAGE_SIZE = custom_yolo_lib.image_size.ImageSize(640, 640)
CLASS_LOSS_GAIN = 0.3
OBJECTNESS_LOSS_GAIN = 0.7
BOX_LOSS_GAIN = 0.05
OBJECTNESS_LOSS_SMALL_MAP_GAIN = 4.0  # bigger grid 80x80 results in smaller loss if BCE
OBJECTNESS_LOSS_MEDIUM_MAP_GAIN = 1.0
OBJECTNESS_LOSS_LARGE_MAP_GAIN = 0.4

if __name__ == "__main__":
    hyperparameters = (
        custom_yolo_lib.config.hyperparameters.ThreeAnchorsHyperparameters(
            experiment_name=EXPERIMENT_NAME,
            model_type=MODEL_TYPE,
            optimizer_type=OPTIMIZER_TYPE,
            loss_type=LOSS_TYPE,
            dataset_type=DATASET_TYPE,
            scheduler_type=SCHEDULER_TYPE,
            warmup_epochs=WARMUP_EPOCHS,
            epochs=EPOCHS,
            dataset_num_classes=NUM_CLASSES,
            batch_size=BATCH_SIZE,
            momentum=MOMENTUM,
            weight_decay=DECAY,
            image_size=IMAGE_SIZE,
            class_loss_gain=CLASS_LOSS_GAIN,
            objectness_loss_gain=OBJECTNESS_LOSS_GAIN,
            box_loss_gain=BOX_LOSS_GAIN,
            objectness_loss_small_map_gain=OBJECTNESS_LOSS_SMALL_MAP_GAIN,
            objectness_loss_medium_map_gain=OBJECTNESS_LOSS_MEDIUM_MAP_GAIN,
            objectness_loss_large_map_gain=OBJECTNESS_LOSS_LARGE_MAP_GAIN,
        )
    )
    args = custom_yolo_lib.experiments.train_val_sessions.args.parse_args()
    dataset_path = pathlib.Path(args.dataset_path)
    if args.experiment_path is None:
        file_path = pathlib.Path(__file__)
        experiment_path = file_path.parent
    else:
        experiment_path = pathlib.Path(args.experiment_path)
    custom_yolo_lib.experiments.train_val_sessions.three_scales.main(
        dataset_path, experiment_path, hyperparameters
    )
