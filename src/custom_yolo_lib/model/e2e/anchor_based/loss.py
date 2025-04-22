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
        self.class_loss = custom_yolo_lib.training.losses.FocalLoss(reduction="none")
        self.box_loss = custom_yolo_lib.training.losses.BoxLoss(
            iou_type=custom_yolo_lib.training.losses.BoxLoss.IoUType.CIoU
        )
        self.objectness_loss = torch.nn.BCELoss(reduction="none")
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
                    check_values=False,
                    positive_sample_iou_thershold=0.5,
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
        batch_loss_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predicted_bboxes_raw (torch.Tensor): A batch of bboxes with shape (batch_size, grid_h * grid_w, 4).
            target_bboxes_raw (torch.Tensor): A batch of bboxes with shape (batch_size, grid_h * grid_w, 4).
            predicted_bboxes_objectnesses (torch.Tensor): A batch of objectness scores with shape (batch_size, grid_h * grid_w).
            target_mask (torch.Tensor): A batch of target masks with shape (batch_size, grid_h * grid_w).
        """
        # The predictions & targets are both in the form of:
        """
        x: [0, 1] in grid_x
        y: [0, 1] in grid_y
        w: log(bw / anchors[best_anchor, 2])
        h: log(bh / anchors[best_anchor, 3])
        "REDUCTION" is sum
        """

        batch_bbox_loss = 0
        for sample_i in range(predicted_bboxes_raw.shape[0]):
            loss_weights = batch_loss_weights[sample_i]
            sample_i_target_mask = target_mask[sample_i]
            if sample_i_target_mask.sum() == 0:
                batch_bbox_loss += loss_weights[loss_weights > 0.5].sum()
                continue

            sample_i_predicted_bboxes = predicted_bboxes_raw[sample_i]
            sample_i_target_bboxes = target_bboxes_raw[sample_i]

            """
            loss is provided with tensors:
            predictions shape (num_objects, 4)
            targets shape (num_objects, 4)
            """
            sample_bbox_loss = self.box_loss(
                sample_i_predicted_bboxes,
                sample_i_target_bboxes,
            )
            batch_bbox_loss += (sample_bbox_loss * loss_weights).sum()
        return batch_bbox_loss

    def _batch_class_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
        batch_loss_weights: torch.Tensor,
    ) -> torch.Tensor:
        batch_class_loss = 0
        for sample_i in range(predictions.shape[0]):
            sample_i_target_mask = target_mask[sample_i]
            loss_weights = batch_loss_weights[sample_i]
            # Calculate class loss only where there are targets
            if sample_i_target_mask.sum() == 0:
                batch_class_loss += loss_weights[loss_weights > 0.5].sum()
                continue

            sample_i_predictions = predictions[
                sample_i
            ]  # shape (grid_h * grid_w, num_classes)
            sample_i_targets = targets[sample_i]
            cls_loss = self.class_loss(
                sample_i_predictions,
                sample_i_targets,
            )
            batch_class_loss += cls_loss.sum()
        return batch_class_loss

    def forward(
        self,
        predictions: DetectionHeadOutput,
        targets_batch: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            predictions (DetectionHeadOutput): The predictions from the model.
            targets_batch (List[torch.Tensor]): The targets for the batch. Each target is a tensor of shape (num_objects, 5).
                The 5 values are (x, y, w, h, class_id).
                x --> center x of the bbox in the image normalized
                y --> center y of the bbox in the image normalized
                w --> width of the bbox in the image normalized
                h --> height of the bbox in the image normalized
                class_id --> class id of the object (0-indexed)
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

        loss = torch.zeros(4, device=predictions.anchor1_output.device)
        batch_size = len(targets_batch)
        if targets_masks.sum() == 0:
            for anchor_i, (
                predictions_anchors,
                targets_anchor,
                targets_mask_anchor,
            ) in enumerate(groups):
                loss[0] += torch.nn.functional.binary_cross_entropy(
                    predictions_anchors[:, :4], targets_anchor[:, :4], reduction="sum"
                )
                loss[1] += torch.nn.functional.binary_cross_entropy(
                    predictions_anchors[:, 4], targets_anchor[:, 4], reduction="sum"
                )
                loss[2] += torch.nn.functional.binary_cross_entropy(
                    predictions_anchors[:, 5:], targets_anchor[:, 5:], reduction="sum"
                )
            # maybe should not penalize the predicted bboxes and class scores if there are no targets
            # loss[3] = loss[1] could be better
            loss[3] = loss[:3].sum()
        else:
            # [bbox_loss, objectness_loss, class_loss, total_loss]
            for anchor_i, (
                predictions_anchors,
                targets_anchor,
                targets_mask_anchor,
            ) in enumerate(groups):
                # reshape to (batch_size, grid_size_h * grid_size_w)
                target_mask = targets_mask_anchor.view(batch_size, -1)
                prediction_objectness = predictions_anchors[:, 4].view(batch_size, -1)
                total_target_objects = target_mask.sum()
                if total_target_objects == 0:
                    continue  # Skip this anchor if there are no targets (Not sure)

                # reshape to (batch_size, grid_size_h * grid_size_w, 4)
                target_bboxes_raw = targets_anchor[:, :4].view(batch_size, -1, 4)
                predicted_bboxes_raw = predictions_anchors[:, :4].view(
                    batch_size, -1, 4
                )

                batch_loss_weights = torch.abs(
                    input=prediction_objectness - target_mask.float()
                )
                loss[0] += self._batch_bbox_loss(
                    predicted_bboxes_raw,
                    target_bboxes_raw,
                    target_mask,
                    batch_loss_weights,
                )

                _objectness_loss = self.objectness_loss(
                    predictions_anchors[:, 4],
                    targets_anchor[:, 4],
                )  # (batch_size, grid_size_h, grid_size_w)
                loss[1] += _objectness_loss.sum() / total_target_objects

                prediction_class_scores = predictions_anchors[:, 5:].view(
                    batch_size, -1, self.num_classes
                )
                target_class_scores = targets_anchor[:, 5:].view(
                    batch_size, -1, self.num_classes
                )
                loss[2] += self._batch_class_loss(
                    prediction_class_scores,
                    target_class_scores,
                    target_mask,
                    batch_loss_weights,
                )
            loss[0] *= 3.0
            loss[1] *= 0.8
            loss[2] *= 1.0
            loss[3] = loss[:3].sum()

        return loss / (batch_size * len(groups))
