import enum

import torch

import custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based


class ModelType(enum.Enum):
    YOLO = enum.auto()
    YOLOFPN = enum.auto()


def init_model(
    model_type: ModelType,
    device: torch.device,
    num_classes: int,
) -> custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based.YOLOModel:
    if model_type == ModelType.YOLO:
        model = custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based.YOLOModel_FAILURE(
            num_classes=num_classes, training=True
        )
    elif model_type == ModelType.YOLOFPN:
        model = custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based.YOLOModel(
            num_classes=num_classes, training=True
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    model.to(device)
    return model
