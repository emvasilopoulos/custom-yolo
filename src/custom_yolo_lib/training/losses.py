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

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE with logits for better numerical stability
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # Probabilities for the correct class
        p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (
            1 - targets
        )

        # Modulating factor
        focal_term = (1 - p_t) ** self.gamma

        # Alpha balancing
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_term * bce_loss
        else:
            focal_loss = focal_term * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        elif self.reduction == "none":
            return focal_loss
        else:
            raise ValueError(
                f"Invalid reduction: {self.reduction}. Supported: 'none', 'mean', 'sum'."
            )


class BoxLoss(torch.nn.Module):

    class IoUType(enum.Enum):
        CIoU = enum.auto()
        DIoU = enum.auto()
        IoU = enum.auto()
        GIoU = enum.auto()

    IOU_THRESHOLD = 0.5

    def __init__(self, iou_type: IoUType) -> None:
        """
        NO REDUCTION
        """
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
        self.__iou = None

    @property
    def iou(self) -> torch.Tensor:
        """
        Returns the IoU value.
        """
        if self.__iou is None:
            raise ValueError("IoU has not been computed yet.")
        return self.__iou

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions (torch.Tensor): Predicted bounding boxes of shape (N, 4).
            targets (torch.Tensor): Target bounding boxes of shape (N, 4).
        Returns:
            torch.Tensor: Loss value.
        """
        self.__iou = self.iou_fn(predictions, targets).squeeze(1)

        return 1.0 - self.__iou
