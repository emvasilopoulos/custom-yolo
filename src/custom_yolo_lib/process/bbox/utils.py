import torch
import custom_yolo_lib.process.bbox


def calculate_iou(
    bbox1: custom_yolo_lib.process.bbox.Bbox, bbox2: custom_yolo_lib.process.bbox.Bbox
) -> float:
    if bbox1.is_normalized != bbox2.is_normalized:
        raise ValueError("Both bounding boxes must have the same normalization state.")
    if bbox1.is_top_left != bbox2.is_top_left:
        raise ValueError("Both bounding boxes must have the same coordinate system.")

    inter_area = min(bbox1.w, bbox2.w) * min(bbox1.h, bbox2.h)
    union_area = bbox1.w * bbox1.h + bbox2.w * bbox2.h - inter_area

    return inter_area / (union_area + 1e-6)


def calculate_iou_tensors(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU) for two sets of bounding boxes.

    Args:
        bbox1 (torch.Tensor): Tensor of shape (N, 4) representing the first set of bounding boxes.
        bbox2 (torch.Tensor): Tensor of shape (M, 4) representing the second set of bounding boxes.

    Returns:
        torch.Tensor: Tensor of shape (N, M) containing the IoU values.
    """
    box1_w, box1_h = bbox1[..., 2], bbox1[..., 3]
    box2_w, box2_h = bbox2[..., 2], bbox2[..., 3]

    inter_area = torch.min(box1_w, box2_w) * torch.min(box1_h, box2_h)
    union_area = box1_w * box1_h + box2_w * box2_h - inter_area

    return inter_area / (union_area + 1e-6)
