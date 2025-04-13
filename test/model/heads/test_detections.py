import math
import unittest

import torch

from custom_yolo_lib.model.heads.detections_3_anchors import (
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
        self.assertEqual(output_tensor.anchor1_output.shape, (1, 85, 13, 13))

    def test_forward2(self):
        model = DetectionHead(
            in_channels=1024, num_classes=80, feature_map_type=FeatureMapType.MEDIUM
        )
        input_tensor = torch.randn((1, 1024, 20, 20))
        # Batch size of 1, 1024 channels, 20x20 feature map
        output_tensor = model(input_tensor, training=False)
        self.assertEqual(output_tensor.anchor1_output.shape, (1, 85, 20, 20))

    def test_forward_all_anchors_shapes(self):
        model = DetectionHead(
            in_channels=512, num_classes=80, feature_map_type=FeatureMapType.LARGE
        )
        input_tensor = torch.randn((1, 512, 26, 26))

        # Batch size of 1, 512 channels, 26x26 feature map
        anchor_outs = model(input_tensor, training=False)

        out1, out2, out3 = anchor_outs
        self.assertEqual(out1.shape, (1, 85, 26, 26))
        self.assertEqual(out2.shape, (1, 85, 26, 26))
        self.assertEqual(out3.shape, (1, 85, 26, 26))

    def test_forward_values(self):

        model = DetectionHead(
            in_channels=1, num_classes=1, feature_map_type=FeatureMapType.MEDIUM
        )
        a = model._multiplier
        b = (a - 1) / 2
        input_tensor = torch.zeros((1, 1, 13, 13))
        input_tensor[0, 0, :6, :] = 1.0

        out = model(input_tensor, training=False)
        out1, out2, out3 = out.anchor1_output, out.anchor2_output, out.anchor3_output
        for i in range(13):
            # x
            assert i - b <= out1[0, 0, 0, i] < i + a
            assert i - b <= out2[0, 0, 0, i] < i + a
            assert i - b <= out3[0, 0, 0, i] < i + a

            # y
            assert i - b <= out1[0, 1, i, 0] < i + a
            assert i - b <= out2[0, 1, i, 0] < i + a
            assert i - b <= out3[0, 1, i, 0] < i + a

    def test_forward_values2(self):
        model = DetectionHead(
            in_channels=1, num_classes=1, feature_map_type=FeatureMapType.MEDIUM
        )
        a = model._multiplier
        b = (a - 1) / 2

        input_tensor = torch.zeros((1, 1, 13, 13))
        input_tensor[0, 0, :6, :] = 1.0
        out1, out2, out3 = model(input_tensor, training=False)

        # h
        self.assertTrue((-b <= out1[0, 2]).all())
        self.assertTrue((out1[0, 2] <= math.exp(a)).all())
        self.assertTrue((-b <= out2[0, 2]).all())
        self.assertTrue((out2[0, 2] <= math.exp(a)).all())
        self.assertTrue((-b <= out3[0, 2]).all())
        self.assertTrue((out3[0, 2] <= math.exp(a)).all())

        # w
        self.assertTrue((0 <= out1[0, 3]).all())
        self.assertTrue((out1[0, 3] <= math.exp(a)).all())
        self.assertTrue((0 <= out2[0, 3]).all())
        self.assertTrue((out2[0, 3] <= math.exp(a)).all())
        self.assertTrue((0 <= out3[0, 3]).all())
        self.assertTrue((out3[0, 3] <= math.exp(a)).all())


if __name__ == "__main__":
    unittest.main()
