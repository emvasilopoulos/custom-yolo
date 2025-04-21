import enum
import torch

import custom_yolo_lib.process.bbox.metrics.iou


class FocalLoss(torch.nn.Module):
    def __init__(
        self, alpha: float = 0.25, gamma: float = 1.5, reduction: str = "mean"
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = predictions.clone()
        ce_loss = torch.nn.functional.binary_cross_entropy(
            predictions, targets, reduction="none"
        )
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # Check reduction option and return loss accordingly
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'self.reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss


class BoxLoss(torch.nn.Module):

    class IoUType(enum.Enum):
        CIoU = enum.auto()
        DIoU = enum.auto()
        IoU = enum.auto()
        GIoU = enum.auto()

    IOU_THRESHOLD = 0.5

    def __init__(self, iou_type: IoUType) -> None:
        super(BoxLoss, self).__init__()
        if iou_type == self.IoUType.CIoU:
            self.iou_fn = custom_yolo_lib.process.bbox.metrics.iou.bbox_ciou
        elif iou_type == self.IoUType.DIoU:
            self.iou_fn = custom_yolo_lib.process.bbox.metrics.iou.bbox_diou
        elif iou_type == self.IoUType.IoU:
            self.iou_fn = custom_yolo_lib.process.bbox.metrics.iou.bbox_iou
        elif iou_type == self.IoUType.GIoU:
            self.iou_fn = custom_yolo_lib.process.bbox.metrics.iou.bbox_giou
        else:
            raise ValueError(
                f"Invalid IoU type: {iou_type}. Supported types: {list(self.IoUType)}"
            )

    def _filter(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Filter out boxes with zero width or height.
        Args:
            boxes (torch.Tensor): Bounding boxes of shape (N, 4).
        Returns:
            torch.Tensor: Filtered bounding boxes.
        """
        boxes = boxes[boxes[:, 2] > 0.0]
        if boxes.shape[0] == 0:
            return boxes
        return boxes[boxes[:, 3] > 0.0]

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions (torch.Tensor): Predicted bounding boxes of shape (N, 4).
            targets (torch.Tensor): Target bounding boxes of shape (M, 4).
        Returns:
            torch.Tensor: Loss value.
        """
        predictions = self._filter(predictions)
        targets = self._filter(targets)
        if predictions.shape[0] == 0 and targets.shape[0] > 0:
            return torch.tensor(1.0, device=predictions.device)
        if targets.shape[0] == 0 and predictions.shape[0] > 0:
            return torch.tensor(1.0, device=predictions.device)
        if predictions.shape[0] == 0 and targets.shape[0] == 0:
            return torch.tensor(0.0, device=predictions.device)
        iou_scores = self.iou_fn(predictions, targets).squeeze(1)
        loss = 1.0 - iou_scores
        return loss.sum()
