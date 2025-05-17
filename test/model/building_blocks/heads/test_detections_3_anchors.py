import math
import unittest

import torch

from custom_yolo_lib.model.building_blocks.heads.detections_3_anchors import (
    DetectionHead,
    FeatureMapType,
)


class TestDetectionHead(unittest.TestCase):

    def test_forward(self):
        model = DetectionHead(
            in_channels=1024, num_classes=80, feature_map_type=FeatureMapType.SMALL
        )
        input_tensor = torch.randn(
            (1, 1024, 13, 13)
        )  # Batch size of 1, 1024 channels, 13x13 feature map
        output_tensor = model(input_tensor, training=False)
        self.assertEqual(output_tensor.shape, (1, 3, 85, 13, 13))

    def test_forward2(self):
        model = DetectionHead(
            in_channels=1024, num_classes=80, feature_map_type=FeatureMapType.MEDIUM
        )
        input_tensor = torch.randn((1, 1024, 20, 20))
        # Batch size of 1, 1024 channels, 20x20 feature map
        output_tensor = model(input_tensor, training=False)
        self.assertEqual(output_tensor.shape, (1, 3, 85, 20, 20))

    def test_forward_all_anchors_shapes(self):
        model = DetectionHead(
            in_channels=512, num_classes=80, feature_map_type=FeatureMapType.LARGE
        )
        input_tensor = torch.randn((1, 512, 26, 26))

        # Batch size of 1, 512 channels, 26x26 feature map
        output = model(input_tensor, training=False)

        self.assertEqual(output.shape, (1, 3, 85, 26, 26))


if __name__ == "__main__":
    unittest.main()
