from typing import List, Tuple
import torch
import torchvision

from custom_yolo_lib.model.e2e.anchor_based.constants import ANCHOR_GAIN
import custom_yolo_lib.model.e2e.anchor_based.training_utils
import custom_yolo_lib.training.losses

BOX_LOSS_GAIN = 1.0
OBJECTNESS_LOSS_GAIN = 1.0
CLASS_LOSS_GAIN = 1.0


class YOLOLossPerFeatureMapV2(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        feature_map_anchors: custom_yolo_lib.model.e2e.anchor_based.training_utils.AnchorsTensor,
        grid_size_h: int,
        grid_size_w: int,
        is_ordinal_objectness: bool = False,
    ) -> None:
        super(YOLOLossPerFeatureMapV2, self).__init__()
        self.num_classes = num_classes
        self.box_loss = custom_yolo_lib.training.losses.BoxLoss(
            iou_type=custom_yolo_lib.training.losses.BoxLoss.IoUType.CIoU
        )
        self.objectness_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        # self.class_loss = torchvision.ops.sigmoid_focal_loss
        self.class_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.feature_map_anchors = feature_map_anchors
        self.grid_size_h = grid_size_h
        self.grid_size_w = grid_size_w
        self.losses_calculated = 0
        self.is_ordinal_objectness = is_ordinal_objectness

    def set_box_loss(
        self, box_loss: custom_yolo_lib.training.losses.BoxLoss.IoUType
    ) -> None:
        self.box_loss = custom_yolo_lib.training.losses.BoxLoss(iou_type=box_loss)

    def _maybe_end_training(self, anchor_pred_bboxes: torch.Tensor) -> None:
        n_boxes = anchor_pred_bboxes.shape[0]
        if n_boxes < 6:
            return
        assert (
            anchor_pred_bboxes[:, 0].sum() < n_boxes * 0.987
        ), "'x' for predicted bboxes converging to 1.0 for all values. Early stopping"
        assert (
            anchor_pred_bboxes[:, 1].sum() < n_boxes * 0.987
        ), "'y' for predicted bboxes converging to 1.0 for all values. Early stopping"
        assert (
            anchor_pred_bboxes[:, 2].sum() < n_boxes * 0.987
        ), "'w' for predicted bboxes converging to 1.0 for all values. Early stopping"
        assert (
            anchor_pred_bboxes[:, 3].sum() < n_boxes * 0.987
        ), "'h' for predicted bboxes converging to 1.0 for all values. Early stopping"

    def _calculate_loss(
        self,
        predictions: torch.Tensor,
        targets_in_grid: torch.Tensor,
        targets_masks: torch.Tensor,
    ) -> torch.Tensor:
        num_anchors = predictions.shape[1]
        batch_size = predictions.shape[0]
        num_feats_per_anchor = predictions.shape[2]
        if targets_masks.sum() == 0:
            batch_obj_loss = self.objectness_loss(
                predictions[:, :, 4], targets_in_grid[:, :, 4]
            ).view(
                batch_size, num_anchors, -1
            )  # batch_size, num_anchors, grid_h * grid_w
            loss = batch_obj_loss.mean()
            return loss, (
                torch.zeros(1, device=predictions.device),
                loss,
                torch.zeros(1, device=predictions.device),
            )

        batch_bbox_loss = 0
        batch_cls_loss = 0

        # to (batch_size, num_anchors, grid_h, grid_w, feats_per_anchor)
        predictions = predictions.permute(0, 1, 3, 4, 2)
        targets_in_grid = targets_in_grid.permute(0, 1, 3, 4, 2)
        predicted_bboxes = predictions[:, :, :, :, :4].sigmoid()
        for anchor_i in range(num_anchors):
            # Extract the predictions and targets for the current anchor
            anchor_predictions = predictions[:, anchor_i]
            anchor_targets = targets_in_grid[:, anchor_i]
            anchor_targets_mask = targets_masks[
                :, anchor_i
            ]  # (batch_size ,grid_h ,grid_w)

            if anchor_targets_mask.sum() == 0:
                continue

            """ BBox loss """
            # part 1 of 2 -  where there are targets
            # NOTE: not translating pred nor target bboxes to a grid cell because they already have the same origin
            ## preds
            anchor_pred_bboxes = predicted_bboxes[:, anchor_i][anchor_targets_mask]
            x = anchor_pred_bboxes[:, 0] * ANCHOR_GAIN - (ANCHOR_GAIN - 1.0) / 2.0
            y = anchor_pred_bboxes[:, 1] * ANCHOR_GAIN - (ANCHOR_GAIN - 1.0) / 2.0
            """ 
            FOR SOME REASON w & h converge at very high values
            specifically (anchor_pred_bboxes[:, 2] * ANCHOR_GAIN) ** 2 converges to 4 ==>
            ==> anchor_pred_bboxes[:, 2] converges to 1.0 with ANCHOR_GAIN == 2.0
            Why? fix this and your experiment will work
            FIXed by multiplying by grid size. Prior anchor values are in the
            range of 0.0 to 1.0 which means the model tries to compensate
            by increasing the values of its output way to much.
            grid_size_X is either 20, 40, 80 etc. denormalizing the prior anchor values.
            """
            if self.losses_calculated % 100 == 0:
                self._maybe_end_training(anchor_pred_bboxes)

            w = (
                (anchor_pred_bboxes[:, 2] * ANCHOR_GAIN) ** 2
                * self.feature_map_anchors.anchors[anchor_i, 2]
                * self.grid_size_w
            )
            h = (
                (anchor_pred_bboxes[:, 3] * ANCHOR_GAIN) ** 2
                * self.feature_map_anchors.anchors[anchor_i, 3]
                * self.grid_size_h
            )
            anchor_pred_bboxes_decoded = torch.cat(
                [
                    x.unsqueeze(1),
                    y.unsqueeze(1),
                    w.unsqueeze(1),
                    h.unsqueeze(1),
                ],
                dim=1,
            )

            ## targets
            anchor_target_bboxes = anchor_targets[:, :, :, :4][anchor_targets_mask]
            # TODO: should I denormalize xy as well?
            # I remembered. I compare boxes in the same grid cell so either way the result will be the same
            # skipping denormalization
            x = anchor_target_bboxes[:, 0]
            y = anchor_target_bboxes[:, 1]
            w = anchor_target_bboxes[:, 2] * self.grid_size_w
            h = anchor_target_bboxes[:, 3] * self.grid_size_h
            anchor_target_bboxes_decoded = torch.cat(
                [
                    x.unsqueeze(1),
                    y.unsqueeze(1),
                    w.unsqueeze(1),
                    h.unsqueeze(1),
                ],
                dim=1,
            )
            batch_bbox_loss += self.box_loss(
                anchor_pred_bboxes_decoded, anchor_target_bboxes_decoded
            ).mean()
            # keep for later because after the second box_loss calculation it will be overwritten

            """ will this work as improvement? """
            # # part 2 of 2 - where there are no targets
            # not_anchor_targets_mask = ~anchor_targets_mask
            # anchor_pred_no_bboxes = predicted_bboxes[:, anchor_i][
            #     not_anchor_targets_mask
            # ]
            # anchor_target_no_bboxes = torch.zeros_like(anchor_pred_no_bboxes) + 1e-7
            # batch_bbox_loss += self.box_loss(
            #     anchor_pred_no_bboxes, anchor_target_no_bboxes
            # ).mean()
            # Update objectness target scores
            if not self.is_ordinal_objectness:
                iou = self.box_loss.iou.detach().clamp(0.0, 1.0)
                anchor_targets[:, :, :, 4][anchor_targets_mask] = iou

            # Class loss
            anchor_pred_class_scores = anchor_predictions[:, :, :, 5:][
                anchor_targets_mask
            ]
            anchor_target_class_scores = anchor_targets[:, :, :, 5:][
                anchor_targets_mask
            ]
            batch_cls_loss += self.class_loss(
                anchor_pred_class_scores, anchor_target_class_scores
            ).mean()

        # objectness loss
        batch_obj_loss = self.objectness_loss(
            predictions[:, :, :, :, 4], targets_in_grid[:, :, :, :, 4]
        ).mean()
        # less sparse tensors ==> bigger loss
        # extra_obj_loss = self.objectness_loss(
        #     predictions[:, :, :, :, 4][targets_masks],
        #     targets_in_grid[:, :, :, :, 4][targets_masks],
        # ).mean()

        # mean across anchors
        final_bbox_loss = batch_bbox_loss
        final_class_loss = batch_cls_loss
        final_objectness_loss = batch_obj_loss

        total_loss = final_bbox_loss + final_objectness_loss + final_class_loss
        self.losses_calculated += 1
        return total_loss, (final_bbox_loss, final_objectness_loss, final_class_loss)

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

        targets_in_grid, targets_masks = _get_targets_in_grid(
            targets_batch,
            grid_size_h=self.grid_size_h,
            grid_size_w=self.grid_size_w,
            feature_map_anchors=self.feature_map_anchors,
            num_classes=self.num_classes,
            is_ordinal_objectness=self.is_ordinal_objectness,
        )

        return self._calculate_loss(
            predictions,
            targets_in_grid,
            targets_masks,
        )

    def debug_forward(
        self,
        predictions: torch.Tensor,
        targets_batch: List[torch.Tensor],
    ) -> torch.Tensor:

        targets_in_grid, targets_masks = _get_targets_in_grid(
            targets_batch,
            grid_size_h=self.grid_size_h,
            grid_size_w=self.grid_size_w,
            feature_map_anchors=self.feature_map_anchors,
            num_classes=self.num_classes,
            is_ordinal_objectness=self.is_ordinal_objectness,
        )

        return (
            self._calculate_loss(
                predictions,
                targets_in_grid,
                targets_masks,
            ),
            targets_in_grid,
            targets_masks,
        )


def _get_targets_in_grid(
    targets_batch: List[torch.Tensor],
    grid_size_h: int,
    grid_size_w: int,
    feature_map_anchors: custom_yolo_lib.model.e2e.anchor_based.training_utils.AnchorsTensor,
    num_classes: int,
    is_ordinal_objectness: bool = False,
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
                is_ordinal_objectness=is_ordinal_objectness,
            )
        )
        targets_in_grid.append(target_in_grid)
        targets_masks.append(target_mask)

    targets_in_grid = torch.stack(targets_in_grid, dim=0)
    targets_masks = torch.stack(targets_masks, dim=0)
    return targets_in_grid, targets_masks
