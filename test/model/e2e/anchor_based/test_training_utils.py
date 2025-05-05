import unittest

import torch

from custom_yolo_lib.model.e2e.anchor_based.training_utils import (
    build_feature_map_targets,
    build_feature_map_targets_backup,
    get_anchors_as_bbox_tensors,
)


class TestTrainingUtils(unittest.TestCase):
    def test_build_feature_map_targets(self):
        small_anchors, medium_anchors, large_anchors = get_anchors_as_bbox_tensors()
        annotations = torch.tensor(
            [
                [0.02, 0.02, 0.1, 0.1, 0],
                [0.22, 0.22, 0.2, 0.2, 1],
            ]
        )
        grid_size_h = 80
        grid_size_w = 80
        num_classes = 80

        targets, targets_mask = build_feature_map_targets(
            annotations,
            anchor_tensor=small_anchors,
            grid_size_h=grid_size_h,
            grid_size_w=grid_size_w,
            num_classes=num_classes,
        )
        targets_true, targets_mask_true = build_feature_map_targets_backup(
            annotations,
            anchor_tensor=small_anchors,
            grid_size_h=grid_size_h,
            grid_size_w=grid_size_w,
            num_classes=num_classes,
        )

        assert targets.shape == targets_true.shape
        assert targets_mask.shape == targets_mask_true.shape
        assert torch.allclose(targets, targets_true)
        assert torch.allclose(targets_mask, targets_mask_true)


if __name__ == "__main__":
    unittest.main()
