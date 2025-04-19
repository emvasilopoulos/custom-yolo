from typing import List, Tuple
import torch

from custom_yolo_lib.model.building_blocks.heads.detections_3_anchors import (
    DetectionHeadOutput,
)
import custom_yolo_lib.model.e2e.anchor_based.training_utils
import custom_yolo_lib.training.losses


class YOLOLossPerFeatureMap(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        feature_map_anchors: custom_yolo_lib.model.e2e.anchor_based.training_utils.AnchorsTensor,
    ) -> None:
        super(YOLOLossPerFeatureMap, self).__init__()
        self.num_classes = num_classes
        self.class_loss = custom_yolo_lib.training.losses.FocalLoss()
        self.box_loss = custom_yolo_lib.training.losses.BoxLoss(
            iou_type=custom_yolo_lib.training.losses.BoxLoss.IoUType.CIoU
        )
        self.objectness_loss = torch.nn.BCELoss()
        self.feature_map_anchors = feature_map_anchors

    def _get_targets_in_grid(
        self,
        targets_batch: List[torch.Tensor],
        grid_size_h: int,
        grid_size_w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        targets_in_grid = []
        targets_masks = []
        for trgt in targets_batch:
            target_in_grid, target_mask = (
                custom_yolo_lib.model.e2e.anchor_based.training_utils.build_feature_map_targets(
                    trgt,
                    anchor_tensor=self.feature_map_anchors,
                    grid_size_h=grid_size_h,  # NOTE: All anchor outputs have the same shape
                    grid_size_w=grid_size_w,
                    num_classes=self.num_classes,
                    check_values=True,
                )
            )
            targets_in_grid.append(target_in_grid)
            targets_masks.append(target_mask)

        targets_in_grid = torch.stack(targets_in_grid, dim=0)
        targets_masks = torch.stack(targets_masks, dim=0)
        return targets_in_grid, targets_masks

    def _batch_bbox_loss(
        self,
        predicted_bboxes_raw: torch.Tensor,
        target_bboxes_raw: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        # The predictions & targets are both in the form of:
        """
        x: [0, 1] in grid_x
        y: [0, 1] in grid_y
        w: log(bw / anchors[best_anchor, 2])
        h: log(bh / anchors[best_anchor, 3])
        """
        batch_bbox_loss = 0
        for sample_i in range(predicted_bboxes_raw.shape[0]):
            sample_i_target_mask = target_mask[sample_i]
            if sample_i_target_mask.sum() == 0:
                continue
            sample_i_predicted_bboxes = predicted_bboxes_raw[sample_i][
                sample_i_target_mask, :
            ]
            sample_i_target_bboxes = target_bboxes_raw[sample_i][
                sample_i_target_mask, :
            ]
            batch_bbox_loss += self.box_loss(
                sample_i_predicted_bboxes,
                sample_i_target_bboxes,
            )
        return batch_bbox_loss / predicted_bboxes_raw.shape[0]

    def _batch_class_loss(
        self,
        target_mask: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        batch_class_loss = 0
        for sample_i in range(predictions.shape[0]):
            sample_i_target_mask = target_mask[sample_i]
            # Calculate class loss only where there are targets
            if sample_i_target_mask.sum() == 0:
                continue
            sample_i_predictions = predictions[sample_i]
            sample_i_targets = targets[sample_i]
            batch_class_loss += self.class_loss(
                sample_i_predictions,
                sample_i_targets,
            )
        return batch_class_loss / predictions.shape[0]

    def forward(
        self,
        predictions: DetectionHeadOutput,
        targets_batch: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            predictions (DetectionHeadOutput): The predictions from the model.
            targets_batch (List[torch.Tensor]): The targets for the batch. Each target is a tensor of shape (num_objects, 5).
                The 5 values are (x, y, w, h, class_id).
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The bbox loss, objectness loss, and class loss.
        """
        targets_in_grid, targets_masks = self._get_targets_in_grid(
            targets_batch,
            grid_size_h=predictions.anchor1_output.shape[2],
            grid_size_w=predictions.anchor1_output.shape[3],
        )

        groups = []
        # (batch_size, num_classes + 5, grid_size_h, grid_size_w)
        predictions_anchor1 = predictions.anchor1_output
        targets_anchor1 = targets_in_grid[:, 0]
        targets_masks_anchor1 = targets_masks[:, 0]
        predictions_anchor2 = predictions.anchor2_output
        targets_anchor2 = targets_in_grid[:, 1]
        targets_masks_anchor2 = targets_masks[:, 1]
        predictions_anchor3 = predictions.anchor3_output
        targets_anchor3 = targets_in_grid[:, 2]
        targets_masks_anchor3 = targets_masks[:, 2]

        groups = [
            (predictions_anchor1, targets_anchor1, targets_masks_anchor1),
            (predictions_anchor2, targets_anchor2, targets_masks_anchor2),
            (predictions_anchor3, targets_anchor3, targets_masks_anchor3),
        ]

        total_bbox_loss = 0
        class_loss = 0
        objectness_loss = 0
        batch_size = len(targets_batch)
        for anchor_i, (
            predictions_anchors,
            targets_anchor,
            targets_mask_anchor,
        ) in enumerate(groups):
            # reshape to (batch_size, grid_size_h * grid_size_w)
            target_mask = targets_mask_anchor.view(batch_size, -1)
            # reshape to (batch_size, grid_size_h * grid_size_w, 4)
            target_bboxes_raw = targets_anchor[:, :4].view(batch_size, -1, 4)
            predicted_bboxes_raw = predictions_anchors[:, :4].view(batch_size, -1, 4)

            total_bbox_loss += self._batch_bbox_loss(
                predicted_bboxes_raw,
                target_bboxes_raw,
                target_mask,
            )
            objectness_loss += self.objectness_loss(
                predictions_anchors[:, 4],
                targets_anchor[:, 4],
            )

            class_loss += self._batch_class_loss(
                target_mask,
                predictions_anchors[:, 5:],
                targets_anchor[:, 5:],
            )
        total_loss = total_bbox_loss + objectness_loss + class_loss
        return (
            total_bbox_loss / len(groups),
            objectness_loss / len(groups),
            class_loss / len(groups),
            total_loss / len(groups),
        )
