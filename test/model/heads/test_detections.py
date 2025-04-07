import unittest

import torch

from custom_yolo_lib.model.heads.detections import DetectionHead


class TestDetectionHead(unittest.TestCase):

    def test_forward(self):
        model = DetectionHead(in_channels=1024, num_classes=80, num_anchors=3)
        input_tensor = torch.randn(
            (1, 1024, 13, 13)
        )  # Batch size of 1, 1024 channels, 13x13 feature map
        output_tensor = model(input_tensor)
        self.assertEqual(output_tensor.shape, (1, 255, 13, 13))

    def test_forward2(self):
        model = DetectionHead(in_channels=1024, num_classes=80, num_anchors=3)
        input_tensor = torch.randn((1, 1024, 20, 20))
        # Batch size of 1, 1024 channels, 20x20 feature map
        output_tensor = model(input_tensor)
        # 3 anchors, 80 classes, 4 bounding box coordinates, 1 objectness score
        # (80 + 5) * 3 = 255
        self.assertEqual(output_tensor.shape, (1, 255, 20, 20))

    def test_forward3(self):
        # 5 anchors
        model = DetectionHead(in_channels=512, num_classes=80, num_anchors=5)
        input_tensor = torch.randn((1, 512, 26, 26))

        # Batch size of 1, 512 channels, 26x26 feature map
        output_tensor = model(input_tensor)

        # 5 anchors, 80 classes, 4 bounding box coordinates, 1 objectness score
        # (80 + 5) * 5 = 425
        self.assertEqual(output_tensor.shape, (1, 425, 26, 26))


if __name__ == "__main__":
    unittest.main()
