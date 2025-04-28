from typing import List, Tuple
import torch

from custom_yolo_lib.model.building_blocks.heads.detections_3_anchors import (
    DetectionHeadOutput,
)
import custom_yolo_lib.model.e2e.anchor_based.training_utils
import custom_yolo_lib.training.losses


class YOLOLossPerFeatureMapV2(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        feature_map_anchors: custom_yolo_lib.model.e2e.anchor_based.training_utils.AnchorsTensor,
        grid_size_h: int,
        grid_size_w: int,
    ) -> None:
        super(YOLOLossPerFeatureMapV2, self).__init__()
        self.num_classes = num_classes
        self.class_loss = custom_yolo_lib.training.losses.FocalLoss(reduction="none")
        self.box_loss = custom_yolo_lib.training.losses.BoxLoss(
            iou_type=custom_yolo_lib.training.losses.BoxLoss.IoUType.CIoU
        )
        self.objectness_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.feature_map_anchors = feature_map_anchors
        self.grid_size_h = grid_size_h
        self.grid_size_w = grid_size_w

    def forward(
        self,
        predictions: torch.Tensor,
        targets_batch: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            predictions (torch.Tensor): The predictions from the model corresponding to the specified Feature Map. Shape (batch_size, num_anchors, feats_per_anchor, grid_size_h, grid_size_w).
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
        num_anchors = predictions.shape[1]
        batch_size = predictions.shape[0]
        num_feats_per_anchor = predictions.shape[2]

        targets_in_grid, targets_masks = _get_targets_in_grid(
            targets_batch,
            grid_size_h=self.grid_size_h,
            grid_size_w=self.grid_size_w,
            feature_map_anchors=self.feature_map_anchors,
            num_classes=self.num_classes,
        )

        if targets_masks.sum() == 0:
            raise NotImplementedError("No targets in the batch. handle it somehow.")

        batch_obj_loss = self.objectness_loss(
            predictions[:, :, 4], targets_in_grid[:, :, 4]
        ).view(
            batch_size, num_anchors, -1
        )  # batch_size, num_anchors, grid_h * grid_w

        batch_bbox_loss = []
        batch_cls_loss = []
        for anchor_i in range(num_anchors):
            # Extract the predictions and targets for the current anchor
            anchor_predictions = predictions[:, anchor_i].view(
                batch_size, num_feats_per_anchor, -1
            )
            anchor_targets = targets_in_grid[:, anchor_i].view(
                batch_size, num_feats_per_anchor, -1
            )
            anchor_targets_mask = targets_masks[:, anchor_i].view(batch_size, -1)

            # Objectness loss
            # anchor_pred_objectnesses = anchor_predictions[:, 4, :]
            anchor_target_objectnesses = anchor_targets[:, 4, :]

            batch_loss_weights = anchor_target_objectnesses

            anchor_num_objects = anchor_targets_mask.sum()  # batch
            if anchor_num_objects == 0:
                continue

            # BBox loss
            anchor_pred_bboxes = anchor_predictions[:, :4, :].permute(0, 2, 1)
            anchor_target_bboxes = anchor_targets[:, :4, :].permute(0, 2, 1)
            batch_bbox_loss.append(
                _batch_bbox_loss(
                    self.box_loss,
                    anchor_pred_bboxes,
                    anchor_target_bboxes,
                    anchor_targets_mask,
                    batch_loss_weights,
                )
            )

            # Class loss
            anchor_pred_class_scores = anchor_predictions[:, 5:, :]
            anchor_target_class_scores = anchor_targets[:, 5:, :]
            batch_cls_loss.append(
                self.class_loss(anchor_pred_class_scores, anchor_target_class_scores)
            )
        # mean across anchors
        batch_box_loss = torch.stack(batch_bbox_loss, dim=1).mean(dim=1)
        batch_cls_loss = torch.stack(batch_cls_loss, dim=1).mean(dim=1)
        batch_objectness_loss = batch_obj_loss.mean(dim=1)

        # gain
        batch_box_loss.mul_(5.0)
        batch_objectness_loss.mul_(0.4)
        batch_cls_loss.mul_(1.0)

        total_loss = batch_box_loss + batch_objectness_loss + batch_cls_loss
        return total_loss, (batch_box_loss, batch_objectness_loss, batch_cls_loss)


def _batch_bbox_loss(
    box_loss: torch.nn.Module,
    predicted_bboxes_raw: torch.Tensor,
    target_bboxes_raw: torch.Tensor,
    target_mask: torch.Tensor,
    batch_loss_weights: torch.Tensor,
    objectness_threshold: float = 0.5,
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
    """

    batch_bbox_loss = []
    for sample_i in range(predicted_bboxes_raw.shape[0]):
        loss_weights = batch_loss_weights[sample_i]
        sample_i_target_mask = target_mask[sample_i]

        # Calculate bbox loss only where there are targets
        if sample_i_target_mask.sum() == 0:
            batch_bbox_loss.append(torch.zeros_like(loss_weights))
            # batch_bbox_loss += loss_weights[loss_weights > objectness_threshold].sum()
            continue

        sample_i_predicted_bboxes = predicted_bboxes_raw[sample_i]
        sample_i_target_bboxes = target_bboxes_raw[sample_i]
        """
        loss is provided with tensors:
        predictions shape (num_objects, 4)
        targets shape (num_objects, 4)
        """
        sample_bbox_loss = box_loss(
            sample_i_predicted_bboxes,
            sample_i_target_bboxes,
        )
        batch_bbox_loss.append(sample_bbox_loss * loss_weights)
    return torch.stack(batch_bbox_loss, dim=0)


def _batch_class_loss(
    class_loss: torch.nn.Module,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    batch_loss_weights: torch.Tensor,
) -> torch.Tensor:
    batch_class_loss = []
    for sample_i in range(predictions.shape[0]):
        sample_i_target_mask = target_mask[sample_i]
        total_target_objects = sample_i_target_mask.sum()
        loss_weights = batch_loss_weights[sample_i]

        sample_i_targets = targets[sample_i]
        # Calculate class loss only where there are targets
        if total_target_objects == 0:
            batch_class_loss.append(torch.zeros_like(sample_i_targets))
            # batch_class_loss += loss_weights[loss_weights > 0.5].mean()
            continue

        sample_i_predictions = predictions[
            sample_i
        ]  # shape (grid_h * grid_w, num_classes)
        cls_loss = class_loss(
            sample_i_predictions,
            sample_i_targets,
        )
        batch_class_loss.append(cls_loss / total_target_objects)
    return torch.stack(batch_class_loss, dim=0)


def _get_targets_in_grid(
    targets_batch: List[torch.Tensor],
    grid_size_h: int,
    grid_size_w: int,
    feature_map_anchors: custom_yolo_lib.model.e2e.anchor_based.training_utils.AnchorsTensor,
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    targets_in_grid = []
    targets_masks = []
    for trgt in targets_batch:
        target_in_grid, target_mask = (
            custom_yolo_lib.model.e2e.anchor_based.training_utils.build_feature_map_targets(
                trgt,
                anchor_tensor=feature_map_anchors,
                grid_size_h=grid_size_h,  # NOTE: All anchor outputs have the same shape
                grid_size_w=grid_size_w,
                num_classes=num_classes,
                check_values=False,
                positive_sample_iou_thershold=0.3,
            )
        )
        targets_in_grid.append(target_in_grid)
        targets_masks.append(target_mask)

    targets_in_grid = torch.stack(targets_in_grid, dim=0)
    targets_masks = torch.stack(targets_masks, dim=0)
    return targets_in_grid, targets_masks
