from typing import List, Tuple
import torch

from custom_yolo_lib.model.e2e.anchor_based.constants import ANCHOR_GAIN
import custom_yolo_lib.model.e2e.anchor_based.training_utils
import custom_yolo_lib.training.losses
import custom_yolo_lib.model.e2e.anchor_based.training_utils

BOX_LOSS_GAIN = 1.0
OBJECTNESS_LOSS_GAIN = 1.0
CLASS_LOSS_GAIN = 1.0

OBJECTNESS_LOSS_GAIN = 1.0
CLASS_LOSS_GAIN = 1.0


class YOLOLossPerFeatureMapV3(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        feature_map_anchors: custom_yolo_lib.model.e2e.anchor_based.training_utils.AnchorsTensor,
        grid_size_h: int,
        grid_size_w: int,
    ) -> None:
        super(YOLOLossPerFeatureMapV3, self).__init__()
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

    def set_box_loss(
        self, box_loss: custom_yolo_lib.training.losses.BoxLoss.IoUType
    ) -> None:
        self.box_loss = custom_yolo_lib.training.losses.BoxLoss(iou_type=box_loss)

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
                _maybe_end_training(anchor_pred_bboxes)

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
        )

        return self._calculate_loss(predictions, targets_in_grid, targets_masks)

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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    targets_in_grid = torch.zeros(
        (
            len(targets_batch),
            feature_map_anchors.anchors.shape[0],
            5 + num_classes,
            grid_size_h,
            grid_size_w,
        ),
        device=targets_batch[0].device,
    )
    targets_masks = torch.zeros(
        (
            len(targets_batch),
            feature_map_anchors.anchors.shape[0],
            grid_size_h,
            grid_size_w,
        ),
        device=targets_batch[0].device,
        dtype=torch.bool,
    )
    for i in range(len(targets_batch)):
        trgt = targets_batch[i]
        target_in_grid, target_mask = _build_feature_map_targets2(
            trgt,
            anchor_tensor=feature_map_anchors,
            grid_size_h=grid_size_h,  # NOTE: All anchor outputs have the same shape
            grid_size_w=grid_size_w,
            num_classes=num_classes,
            check_values=False,
        )
        targets_in_grid[i] = target_in_grid
        targets_masks[i] = target_mask
    return targets_in_grid, targets_masks


def _build_feature_map_targets(
    annotations: torch.Tensor,
    anchor_tensor: custom_yolo_lib.model.e2e.anchor_based.training_utils.AnchorsTensor,
    grid_size_h: int,
    grid_size_w: int,
    num_classes: int,
    check_values: bool = False,
    epsilon: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    anchors = anchor_tensor.anchors
    num_anchors = anchors.shape[0]
    assert num_anchors == 3, "Only 3 anchors are supported for each feature map type"

    targets = torch.zeros(
        (num_anchors, 5 + num_classes, grid_size_h, grid_size_w),
        device=annotations.device,
    )
    targets_mask = torch.zeros(
        (num_anchors, grid_size_h, grid_size_w),
        dtype=torch.bool,
        device=annotations.device,
    )

    if len(annotations) == 0:
        return targets, targets_mask

    # Extract annotation data
    bxc, byc, bw, bh, class_ids = annotations.unbind(1)
    class_ids = class_ids.long()

    # Reshape for broadcasting
    bw_expanded = bw.unsqueeze(1)  # [N, 1]
    bh_expanded = bh.unsqueeze(1)  # [N, 1]
    anchor_w = anchors[:, 2].unsqueeze(0)  # [1, num_anchors]
    anchor_h = anchors[:, 3].unsqueeze(0)  # [1, num_anchors]

    # Check which anchors are valid for each annotation
    valid_mask = (
        (anchor_w * (ANCHOR_GAIN**2) - epsilon >= bw_expanded)
        & (anchor_w / (ANCHOR_GAIN**2) + epsilon <= bw_expanded)
        & (anchor_h * (ANCHOR_GAIN**2) - epsilon >= bh_expanded)
        & (anchor_h / (ANCHOR_GAIN**2) + epsilon <= bh_expanded)
    )

    # Get indices of valid pairs
    ann_idx, anchor_idx = valid_mask.nonzero(as_tuple=True)

    if len(ann_idx) == 0:
        return targets, targets_mask

    # Get corresponding data for valid pairs
    sel_bxc = bxc[ann_idx]
    sel_byc = byc[ann_idx]
    sel_bw = bw[ann_idx]
    sel_bh = bh[ann_idx]
    sel_class_ids = class_ids[ann_idx]

    # Calculate grid positions and offsets
    grid_x = sel_bxc * grid_size_w
    grid_x_int = grid_x.int()
    x_offset = grid_x - grid_x_int
    grid_y = sel_byc * grid_size_h
    grid_y_int = grid_y.int()
    y_offset = grid_y - grid_y_int
    grid_y_int_top = grid_y_int - (sel_bh * grid_size_h).int() // 2
    grid_y_int_bottom = grid_y_int + (sel_bh * grid_size_h).int() // 2
    grid_x_int_left = grid_x_int - (sel_bw * grid_size_w).int() // 2
    grid_x_int_right = grid_x_int + (sel_bw * grid_size_w).int() // 2

    # center x and y in the grid cell
    objectness_score = 1.0
    for i in range(len(ann_idx)):
        a_idx = anchor_idx[i]
        g_y = grid_y_int[i]
        g_x = grid_x_int[i]
        gy_top = grid_y_int_top[i]
        gy_bottom = grid_y_int_bottom[i]
        gx_left = grid_x_int_left[i]
        gx_right = grid_x_int_right[i]

        if targets[a_idx, 4, g_y, g_x] < objectness_score:
            # Update bbox
            targets[a_idx, 0, g_y, g_x] = x_offset[i]
            targets[a_idx, 1, g_y, g_x] = y_offset[i]
            targets[a_idx, 2, g_y, g_x] = sel_bw[i]
            targets[a_idx, 3, g_y, g_x] = sel_bh[i]

            if check_values:
                assert 0 <= x_offset[i] <= 1
                assert 0 <= y_offset[i] <= 1
                assert 0 <= sel_bw[i] <= 1
                assert 0 <= sel_bh[i] <= 1

            # Update objectness
            targets[a_idx, 4, g_y, g_x] = objectness_score
            targets[a_idx, 4, gy_top:gy_bottom, gx_left:gx_right] = (
                objectness_score * 0.5
            )  # this is correct
            targets_mask[a_idx, g_y, g_x] = True
            # targets_mask[a_idx, gy_top:gy_bottom, gx_left:gx_right] = True # for debugging

            # Update class
            targets[a_idx, 5 + sel_class_ids[i], g_y, g_x] = 1.0

    return targets, targets_mask


def _build_feature_map_targets2(
    annotations: torch.Tensor,
    anchor_tensor: custom_yolo_lib.model.e2e.anchor_based.training_utils.AnchorsTensor,
    grid_size_h: int,
    grid_size_w: int,
    num_classes: int,
    check_values: bool = False,
    epsilon: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    anchors = anchor_tensor.anchors
    num_anchors = anchors.shape[0]
    assert num_anchors == 3, "Only 3 anchors are supported for each feature map type"

    targets = torch.zeros(
        (num_anchors, 5 + num_classes, grid_size_h, grid_size_w),
        device=annotations.device,
    )
    targets_mask = torch.zeros(
        (num_anchors, grid_size_h, grid_size_w),
        dtype=torch.bool,
        device=annotations.device,
    )

    if len(annotations) == 0:
        return targets, targets_mask

    # Extract annotation data
    bxc, byc, bw, bh, class_ids = annotations.unbind(1)
    class_ids = class_ids.long()

    # Reshape for broadcasting
    bw_expanded = bw.unsqueeze(1)  # [N, 1]
    bh_expanded = bh.unsqueeze(1)  # [N, 1]
    anchor_w = anchors[:, 2].unsqueeze(0)  # [1, num_anchors]
    anchor_h = anchors[:, 3].unsqueeze(0)  # [1, num_anchors]

    # Check which anchors are valid for each annotation
    valid_mask = (
        (anchor_w * (ANCHOR_GAIN**2) - epsilon >= bw_expanded)
        & (anchor_w / (ANCHOR_GAIN**2) + epsilon <= bw_expanded)
        & (anchor_h * (ANCHOR_GAIN**2) - epsilon >= bh_expanded)
        & (anchor_h / (ANCHOR_GAIN**2) + epsilon <= bh_expanded)
    )

    # Get indices of valid pairs
    ann_idx, anchor_idx = valid_mask.nonzero(as_tuple=True)

    if len(ann_idx) == 0:
        return targets, targets_mask

    # Get corresponding data for valid pairs
    sel_bxc = bxc[ann_idx]
    sel_byc = byc[ann_idx]
    sel_bw = bw[ann_idx]
    sel_bh = bh[ann_idx]
    sel_class_ids = class_ids[ann_idx]

    # Calculate grid positions and offsets
    grid_x = sel_bxc * grid_size_w
    g_x = grid_x.int()
    x_offset = grid_x - g_x
    grid_y = sel_byc * grid_size_h
    g_y = grid_y.int()
    y_offset = grid_y - g_y

    gy_top = g_y - (sel_bh * grid_size_h).int() // 2
    gy_bottom = g_y + (sel_bh * grid_size_h).int() // 2
    gx_left = g_x - (sel_bw * grid_size_w).int() // 2
    gx_right = g_x + (sel_bw * grid_size_w).int() // 2

    # Update bbox, objectness, and class in a vectorized manner
    objectness_score = 1.0
    a_idx = anchor_idx

    # Update bbox
    targets[a_idx, 0, g_y, g_x] = x_offset
    targets[a_idx, 1, g_y, g_x] = y_offset
    targets[a_idx, 2, g_y, g_x] = sel_bw
    targets[a_idx, 3, g_y, g_x] = sel_bh

    if check_values:
        assert torch.all((0 <= x_offset) & (x_offset <= 1))
        assert torch.all((0 <= y_offset) & (y_offset <= 1))
        assert torch.all((0 <= sel_bw) & (sel_bw <= 1))
        assert torch.all((0 <= sel_bh) & (sel_bh <= 1))

    # Update objectness
    targets_mask[a_idx, g_y, g_x] = True
    for i in range(len(a_idx)):
        H = gy_bottom[i] - gy_top[i]
        W = gx_right[i] - gx_left[i]
        if H <= 0 or W <= 0:
            continue
        radial_decay_map = radial_decay(
            H,
            W,
            mode="linear",
            sigma=None,
            dtype=targets.dtype,
            device=targets.device,
        )
        targets[a_idx[i], 4, gy_top[i] : gy_bottom[i], gx_left[i] : gx_right[i]] = (
            radial_decay_map
        )
    targets[a_idx, 4, g_y, g_x] = objectness_score

    # Update class
    targets[a_idx, 5 + sel_class_ids, g_y, g_x] = 1.0

    return targets, targets_mask


def _maybe_end_training(anchor_pred_bboxes: torch.Tensor) -> None:
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


def radial_decay(
    H: int,
    W: int,
    mode: str = "linear",  # "linear" or "gaussian"
    sigma: float | None = None,  # only for Gaussian
    dtype=torch.float32,
    device="cpu",
) -> torch.Tensor:
    """
    Create an HxW tensor whose centre is 1 and whose values
    decrease toward 0 with radial distance.
    """
    # 1. pixel-coordinate grid, centred at (0,0)
    yy = torch.arange(H, dtype=dtype, device=device) - (H - 1) / 2
    xx = torch.arange(W, dtype=dtype, device=device) - (W - 1) / 2
    yy, xx = torch.meshgrid(yy, xx, indexing="ij")  # shape (H, W)

    # 2. radial distance from the centre for every pixel
    r = torch.sqrt(xx**2 + yy**2)

    if mode == "linear":
        # scale so that the farthest pixel is distance 1
        r_norm = r / r.max()
        tensor = (1.0 - r_norm).clamp(min=0.0)

    elif mode == "gaussian":
        # default σ: quarter of the image diagonal
        if sigma is None:
            sigma = 0.25 * r.max()
        tensor = torch.exp(-(r**2) / (2 * sigma**2))

    else:
        raise ValueError("mode must be 'linear' or 'gaussian'")

    return tensor
