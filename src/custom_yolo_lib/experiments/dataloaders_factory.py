import enum

import pathlib

import custom_yolo_lib.image_size
import custom_yolo_lib.process.image.e2e
import custom_yolo_lib.dataset.coco.tasks.instances
import custom_yolo_lib.dataset.coco.tasks.loader
import custom_yolo_lib.dataset.augmentation_types

"""
when you introduce a new dataset other than COCO create a submodule in the current directory.
for each dataset create a separate factory
"""


class DatasetType(enum.Enum):
    COCO_ORIGINAL = enum.auto()
    COCO_SAMA = enum.auto()
    COCO_SAMA_AUGMENT = enum.auto()
    COCO_ORIGINAL_THREE_FEATURE_MAPS = enum.auto()


def init_dataloaders(
    dataset_type: DatasetType,
    dataset_path: pathlib.Path,
    num_classes: int,
    image_size: custom_yolo_lib.image_size.ImageSize,
    batch_size: int,
) -> tuple:
    classes = [i for i in range(num_classes)]
    e2e_preprocessor = custom_yolo_lib.process.image.e2e.E2EPreprocessor(
        expected_image_size=image_size,
    )
    if dataset_type == DatasetType.COCO_ORIGINAL:
        is_sama = False
        dataloader = custom_yolo_lib.dataset.coco.tasks.loader.COCODataLoader
        augmentations = []
    elif dataset_type == DatasetType.COCO_SAMA:
        is_sama = True
        dataloader = custom_yolo_lib.dataset.coco.tasks.loader.COCODataLoader
        augmentations = []
    elif dataset_type == DatasetType.COCO_ORIGINAL_THREE_FEATURE_MAPS:
        is_sama = False
        dataloader = (
            custom_yolo_lib.dataset.coco.tasks.loader.COCODataLoaderThreeFeatureMaps
        )
        augmentations = []
    elif dataset_type == DatasetType.COCO_SAMA_AUGMENT:
        is_sama = True
        dataloader = custom_yolo_lib.dataset.coco.tasks.loader.COCODataLoader
        augmentations = [
            custom_yolo_lib.dataset.augmentation_types.AugmentationType.FLIP_X,
            custom_yolo_lib.dataset.augmentation_types.AugmentationType.FLIP_Y,
            custom_yolo_lib.dataset.augmentation_types.AugmentationType.SLIGHT_RESIZE,
        ]
    train_dataset = custom_yolo_lib.dataset.coco.tasks.instances.COCOInstances2017(
        dataset_path,
        "train",
        expected_image_size=image_size,
        classes=classes,
        is_sama=is_sama,
        e2e_preprocessor=e2e_preprocessor,
        augmentations=augmentations,
    )
    training_loader = dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_dataset = custom_yolo_lib.dataset.coco.tasks.instances.COCOInstances2017(
        dataset_path,
        "val",
        expected_image_size=image_size,
        classes=classes,
        is_sama=is_sama,
        e2e_preprocessor=e2e_preprocessor,
    )
    validation_loader = dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return training_loader, validation_loader
