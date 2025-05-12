import dataclasses
from typing import Tuple
import torch


import custom_yolo_lib.model.building_blocks.heads.anchors.anchors_3_coco
from custom_yolo_lib.model.building_blocks.heads.detections_3_anchors import (
    FeatureMapType,
)


@dataclasses.dataclass
class AnchorsTensor:
    anchors: torch.Tensor
    feature_map_type: FeatureMapType


def get_anchors_as_bbox_tensors(
    device: torch.device = torch.device("cpu"),
) -> Tuple[AnchorsTensor, AnchorsTensor, AnchorsTensor]:
    """
    Get anchors as bounding box tensors for the specified feature map type.

    Args:
        feature_map_type (FeatureMapType): The feature map type.

    Returns:
        torch.Tensor: Anchors as bounding box tensors.
    """
    small_anchors = AnchorsTensor(
        anchors=torch.tensor(
            custom_yolo_lib.model.building_blocks.heads.anchors.anchors_3_coco.SMALL_MAP_FEATS_ANCHORS_LIST,
            device=device,
        ),
        feature_map_type=FeatureMapType.SMALL,
    )
    medium_anchors = AnchorsTensor(
        anchors=torch.tensor(
            custom_yolo_lib.model.building_blocks.heads.anchors.anchors_3_coco.MEDIUM_MAP_FEATS_ANCHORS_LIST,
            device=device,
        ),
        feature_map_type=FeatureMapType.MEDIUM,
    )
    large_anchors = AnchorsTensor(
        anchors=torch.tensor(
            custom_yolo_lib.model.building_blocks.heads.anchors.anchors_3_coco.LARGE_MAP_FEATS_ANCHORS_LIST,
            device=device,
        ),
        feature_map_type=FeatureMapType.LARGE,
    )
    return small_anchors, medium_anchors, large_anchors
