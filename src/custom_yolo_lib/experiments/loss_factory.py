import enum

import torch

import custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based
import custom_yolo_lib.model.e2e.anchor_based.training_utils
import custom_yolo_lib.image_size
import custom_yolo_lib.model.e2e.anchor_based.losses.loss


class LossType(enum.Enum):
    THREESCALE_YOLO = enum.auto()
    THREESCALE_YOLO_ORD = enum.auto()


def init_loss(
    loss_type: LossType,
    model: custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based.YOLOModel,
    device: torch.device,
    expected_image_size: custom_yolo_lib.image_size.ImageSize,
    num_classes: int,
) -> torch.nn.Module:
    if loss_type == LossType.THREESCALE_YOLO:
        is_ordinal_objectness = False
    elif loss_type == LossType.THREESCALE_YOLO_ORD:
        is_ordinal_objectness = True
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

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
