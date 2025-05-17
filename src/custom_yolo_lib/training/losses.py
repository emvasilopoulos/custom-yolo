import enum
import torch

import torchvision.ops

import custom_yolo_lib.process.bbox.metrics.iou


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torchvision.ops.sigmoid_focal_loss(
            logits, targets, alpha=self.alpha, gamma=self.gamma, reduction="none"
        )


class BoxLoss(torch.nn.Module):

    class IoUType(enum.Enum):
        CIoU = enum.auto()
        DIoU = enum.auto()
        IoU = enum.auto()
        GIoU = enum.auto()

    IOU_THRESHOLD = 0.5

    def __init__(self, iou_type: IoUType, xywh: bool = True) -> None:
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
        self.__xywh = xywh

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
            torch.Tensor: Loss value of shape (N,).

        NOTE on the assertion:
        Generally, when the number of boxes in predictions and targets are not equal
        a RuntimeError is raised except for the case when one of predictions or targets
        has 1 box and the other has N boxes. In that case, broadcasting is applied and
        an IoU is calculated which is not the one we want.
        """

        assert predictions.shape[0] == targets.shape[0], (
            f"Predictions and targets must have the same number of boxes. "
            f"Got {predictions.shape[0]} and {targets.shape[0]}"
        )
        self.__iou = self.iou_fn(predictions, targets, xywh=self.__xywh).squeeze(1)

        return 1.0 - self.__iou
