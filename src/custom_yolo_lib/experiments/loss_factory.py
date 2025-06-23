import enum
from typing import Type

import torch

import custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based
import custom_yolo_lib.model.e2e.anchor_based.training_utils
import custom_yolo_lib.image_size
import custom_yolo_lib.model.e2e.anchor_based.losses.loss
import custom_yolo_lib.model.e2e.anchor_based.losses.base_v3
import custom_yolo_lib.model.e2e.anchor_based.losses.loss_v3


class LossType(enum.Enum):
    THREESCALE_YOLO = enum.auto()
    THREESCALE_YOLO_ORD = enum.auto()
    THREESCALE_YOLO_ORD_v3 = enum.auto()
    THREESCALE_YOLO_ORD_v3_GAUSSIAN_RADIAL_DECAY = enum.auto()


def _init_v2(
    predictions_s: torch.Tensor,
    predictions_m: torch.Tensor,
    predictions_l: torch.Tensor,
    num_classes: int,
    is_ordinal_objectness: bool,
    small_map_anchors: torch.Tensor,
    medium_map_anchors: torch.Tensor,
    large_map_anchors: torch.Tensor,
):

    loss_s = custom_yolo_lib.model.e2e.anchor_based.losses.loss.YOLOLossPerFeatureMapV2(
        num_classes=num_classes,
        feature_map_anchors=small_map_anchors,
        grid_size_h=predictions_s.shape[3],
        grid_size_w=predictions_s.shape[4],
        is_ordinal_objectness=is_ordinal_objectness,
    )
    loss_m = custom_yolo_lib.model.e2e.anchor_based.losses.loss.YOLOLossPerFeatureMapV2(
        num_classes=num_classes,
        feature_map_anchors=medium_map_anchors,
        grid_size_h=predictions_m.shape[3],
        grid_size_w=predictions_m.shape[4],
        is_ordinal_objectness=is_ordinal_objectness,
    )
    loss_l = custom_yolo_lib.model.e2e.anchor_based.losses.loss.YOLOLossPerFeatureMapV2(
        num_classes=num_classes,
        feature_map_anchors=large_map_anchors,
        grid_size_h=predictions_l.shape[3],
        grid_size_w=predictions_l.shape[4],
        is_ordinal_objectness=is_ordinal_objectness,
    )
    return loss_s, loss_m, loss_l


def _init_v3(
    predictions_s: torch.Tensor,
    predictions_m: torch.Tensor,
    predictions_l: torch.Tensor,
    num_classes: int,
    small_map_anchors: torch.Tensor,
    medium_map_anchors: torch.Tensor,
    large_map_anchors: torch.Tensor,
    loss_v3_class: Type[
        custom_yolo_lib.model.e2e.anchor_based.losses.base_v3.BaseYOLOLossPerFeatureMapV3
    ] = None,
):

    loss_s = loss_v3_class(
        num_classes=num_classes,
        feature_map_anchors=small_map_anchors,
        grid_size_h=predictions_s.shape[3],
        grid_size_w=predictions_s.shape[4],
    )
    loss_m = loss_v3_class(
        num_classes=num_classes,
        feature_map_anchors=medium_map_anchors,
        grid_size_h=predictions_m.shape[3],
        grid_size_w=predictions_m.shape[4],
    )
    loss_l = loss_v3_class(
        num_classes=num_classes,
        feature_map_anchors=large_map_anchors,
        grid_size_h=predictions_l.shape[3],
        grid_size_w=predictions_l.shape[4],
    )
    return loss_s, loss_m, loss_l


def init_loss(
    loss_type: LossType,
    model: custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based.YOLOModel,
    device: torch.device,
    expected_image_size: custom_yolo_lib.image_size.ImageSize,
    num_classes: int,
) -> torch.nn.Module:
    predictions_s, predictions_m, predictions_l = model.train_forward2(
        torch.zeros((1, 3, expected_image_size.height, expected_image_size.width)).to(
            device
        )
    )

    small_map_anchors, medium_map_anchors, large_map_anchors = (
        custom_yolo_lib.model.e2e.anchor_based.training_utils.get_anchors_as_bbox_tensors(
            device
        )
    )
    if loss_type == LossType.THREESCALE_YOLO:
        loss_s, loss_m, loss_l = _init_v2(
            predictions_s,
            predictions_m,
            predictions_l,
            num_classes,
            False,
            small_map_anchors,
            medium_map_anchors,
            large_map_anchors,
        )
    elif loss_type == LossType.THREESCALE_YOLO_ORD:
        loss_s, loss_m, loss_l = _init_v2(
            predictions_s,
            predictions_m,
            predictions_l,
            num_classes,
            True,
            small_map_anchors,
            medium_map_anchors,
            large_map_anchors,
        )
    elif loss_type == LossType.THREESCALE_YOLO_ORD_v3:
        loss_s, loss_m, loss_l = _init_v3(
            predictions_s,
            predictions_m,
            predictions_l,
            num_classes,
            small_map_anchors,
            medium_map_anchors,
            large_map_anchors,
            custom_yolo_lib.model.e2e.anchor_based.losses.loss_v3.YOLOLossPerFeatureMapV3,
        )
    elif loss_type == LossType.THREESCALE_YOLO_ORD_v3_GAUSSIAN_RADIAL_DECAY:
        loss_s, loss_m, loss_l = _init_v3(
            predictions_s,
            predictions_m,
            predictions_l,
            num_classes,
            small_map_anchors,
            medium_map_anchors,
            large_map_anchors,
            custom_yolo_lib.model.e2e.anchor_based.losses.loss_v3.YOLOLossPerFeatureMapV3GaussianRadialDecay,
        )
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    return loss_s, loss_m, loss_l


def calculate_three_scale_loss(
    losses_s: torch.Tensor,
    losses_m: torch.Tensor,
    losses_l: torch.Tensor,
    loss_type: LossType,
    BOX_LOSS_GAIN: float,
    OBJECTNESS_LOSS_GAIN: float,
    OBJECTNESS_LOSS_SMALL_MAP_GAIN: float,
    OBJECTNESS_LOSS_MEDIUM_MAP_GAIN: float,
    OBJECTNESS_LOSS_LARGE_MAP_GAIN: float,
    CLASS_LOSS_GAIN: float,
):
    if (
        loss_type != LossType.THREESCALE_YOLO
        and loss_type != LossType.THREESCALE_YOLO_ORD
        and loss_type != LossType.THREESCALE_YOLO_ORD_v3
        and loss_type != LossType.THREESCALE_YOLO_ORD_v3_GAUSSIAN_RADIAL_DECAY
    ):
        raise ValueError(
            f"calculate_three_scale_loss only supports THREESCALE_YOLO and THREESCALE_YOLO_ORD, but got {loss_type}"
        )
    avg_bbox_loss = (losses_s[0] + losses_m[0] + losses_l[0]) * BOX_LOSS_GAIN
    avg_objectness_loss = (
        losses_s[1] * OBJECTNESS_LOSS_SMALL_MAP_GAIN
        + losses_m[1] * OBJECTNESS_LOSS_MEDIUM_MAP_GAIN
        + losses_l[1] * OBJECTNESS_LOSS_LARGE_MAP_GAIN
    ) * OBJECTNESS_LOSS_GAIN
    avg_class_loss = (losses_s[2] + losses_m[2] + losses_l[2]) * CLASS_LOSS_GAIN
    loss = avg_bbox_loss + avg_objectness_loss + avg_class_loss
    return (avg_bbox_loss, avg_objectness_loss, avg_class_loss), loss
