import math

import torch

import custom_yolo_lib.process.bbox.metrics.utils


def bbox_iou(
    box1: torch.Tensor, box2: torch.Tensor, xywh: bool = True, eps: float = 1e-7
) -> torch.Tensor:
    b1_x1, b1_y1, b1_x2, b1_y2, w1, h1 = (
        custom_yolo_lib.process.bbox.metrics.utils.extract_coords(box1, xywh, eps)
    )
    b2_x1, b2_y1, b2_x2, b2_y2, w2, h2 = (
        custom_yolo_lib.process.bbox.metrics.utils.extract_coords(box2, xywh, eps)
    )
    inter = custom_yolo_lib.process.bbox.metrics.utils.intersection(
        b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2
    )  # Intersection area
    union = custom_yolo_lib.process.bbox.metrics.utils.union(
        w1, h1, w2, h2, inter, eps
    )  # Union Area
    return inter / union


def bbox_giou(
    box1: torch.Tensor, box2: torch.Tensor, xywh: bool = True, eps: float = 1e-7
) -> torch.Tensor:
    """
    Calculate the Intersection over Union (IoU) between bounding boxes.

    This function supports various shapes for `box1` and `box2` as long as the last dimension is 4.
    For instance, you may pass tensors shaped like (4,), (N, 4), (B, N, 4), or (B, N, 1, 4).
    Internally, the code will split the last dimension into (x, y, w, h) if `xywh=True`,
    or (x1, y1, x2, y2) if `xywh=False`.

    Args:
        box1 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        box2 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format.
        eps (float, optional): A small value to avoid division by zero.
    Returns:
        (torch.Tensor): Generalized IoU values.
    """
    b1_x1, b1_y1, b1_x2, b1_y2, w1, h1 = (
        custom_yolo_lib.process.bbox.metrics.utils.extract_coords(box1, xywh, eps)
    )
    b2_x1, b2_y1, b2_x2, b2_y2, w2, h2 = (
        custom_yolo_lib.process.bbox.metrics.utils.extract_coords(box2, xywh, eps)
    )
    inter = custom_yolo_lib.process.bbox.metrics.utils.intersection(
        b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2
    )  # Intersection area
    union = custom_yolo_lib.process.bbox.metrics.utils.union(
        w1, h1, w2, h2, inter, eps
    )  # Union Area

    # IoU
    iou = inter / union

    # convex (smallest enclosing box)
    cw, ch = custom_yolo_lib.process.bbox.metrics.utils.convex_smallest_enclosing_box(
        b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2
    )

    c_area = cw * ch + eps  # convex area
    return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf


def bbox_diou(
    box1: torch.Tensor, box2: torch.Tensor, xywh: bool = True, eps: float = 1e-7
) -> torch.Tensor:
    """
    Calculate the Intersection over Union (IoU) between bounding boxes.

    This function supports various shapes for `box1` and `box2` as long as the last dimension is 4.
    For instance, you may pass tensors shaped like (4,), (N, 4), (B, N, 4), or (B, N, 1, 4).
    Internally, the code will split the last dimension into (x, y, w, h) if `xywh=True`,
    or (x1, y1, x2, y2) if `xywh=False`.

    Args:
        box1 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        box2 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format.
        eps (float, optional): A small value to avoid division by zero.
    Returns:
        (torch.Tensor): Distance IoU values.
    """
    b1_x1, b1_y1, b1_x2, b1_y2, w1, h1 = (
        custom_yolo_lib.process.bbox.metrics.utils.extract_coords(box1, xywh, eps)
    )
    b2_x1, b2_y1, b2_x2, b2_y2, w2, h2 = (
        custom_yolo_lib.process.bbox.metrics.utils.extract_coords(box2, xywh, eps)
    )
    inter = custom_yolo_lib.process.bbox.metrics.utils.intersection(
        b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2
    )  # Intersection area
    union = custom_yolo_lib.process.bbox.metrics.utils.union(
        w1, h1, w2, h2, inter, eps
    )  # Union Area

    # IoU
    iou = inter / union

    cw, ch = custom_yolo_lib.process.bbox.metrics.utils.convex_smallest_enclosing_box(
        b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2
    )  # convex (smallest enclosing box)

    c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
    rho2 = (
        (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
    ) / 4  # center dist**2
    return iou - rho2 / c2


def bbox_ciou(
    box1: torch.Tensor, box2: torch.Tensor, xywh: bool = True, eps: float = 1e-7
) -> torch.Tensor:
    """
    Calculate the Intersection over Union (IoU) between bounding boxes.

    This function supports various shapes for `box1` and `box2` as long as the last dimension is 4.
    For instance, you may pass tensors shaped like (4,), (N, 4), (B, N, 4), or (B, N, 1, 4).
    Internally, the code will split the last dimension into (x, y, w, h) if `xywh=True`,
    or (x1, y1, x2, y2) if `xywh=False`.

    Args:
        box1 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        box2 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format.
        eps (float, optional): A small value to avoid division by zero.
    Returns:
        (torch.Tensor): Complete IoU values.
    """
    b1_x1, b1_y1, b1_x2, b1_y2, w1, h1 = (
        custom_yolo_lib.process.bbox.metrics.utils.extract_coords(box1, xywh, eps)
    )
    b2_x1, b2_y1, b2_x2, b2_y2, w2, h2 = (
        custom_yolo_lib.process.bbox.metrics.utils.extract_coords(box2, xywh, eps)
    )
    inter = custom_yolo_lib.process.bbox.metrics.utils.intersection(
        b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2
    )  # Intersection area
    union = custom_yolo_lib.process.bbox.metrics.utils.union(
        w1, h1, w2, h2, inter, eps
    )  # Union Area

    # IoU
    iou = inter / union

    cw, ch = custom_yolo_lib.process.bbox.metrics.utils.convex_smallest_enclosing_box(
        b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2
    )  # convex (smallest enclosing box)
    c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
    rho2 = (
        (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
    ) / 4  # center dist**2
    # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
    v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)  # CIoU
