import math
import unittest

import torch

import custom_yolo_lib.training.losses as losses


class TestBoxLoss(unittest.TestCase):

    def _test_box_loss_zero_iou_type(self, iou_type):
        bbox_loss = losses.BoxLoss(iou_type)
        pred_boxes = torch.Tensor([[0.5, 0.5, 1.0, 1.0]])
        true_boxes = torch.Tensor([[0.5, 0.5, 1.0, 1.0]])
        loss = bbox_loss(pred_boxes, true_boxes).sum().item()
        self.assertAlmostEqual(loss, 0.0, places=6)

    def test_box_loss_zero_iou(self):
        self._test_box_loss_zero_iou_type(losses.BoxLoss.IoUType.IoU)

    def test_box_loss_zero_giou(self):
        self._test_box_loss_zero_iou_type(losses.BoxLoss.IoUType.GIoU)

    def test_box_loss_zero_diou(self):
        self._test_box_loss_zero_iou_type(losses.BoxLoss.IoUType.DIoU)

    def test_box_loss_zero_ciou(self):
        self._test_box_loss_zero_iou_type(losses.BoxLoss.IoUType.CIoU)

    ################################################################################
    BOX1 = torch.Tensor([50, 100, 200, 300])  # (x1, y1, x2, y2)
    BOX2 = torch.Tensor([80, 120, 220, 310])  # (x1, y1, x2, y2)

    def _get_box_loss(self, iou_type: losses.BoxLoss.IoUType, xywh: bool = False):
        bbox_loss = losses.BoxLoss(iou_type, xywh=xywh)
        pred_boxes = self.BOX1.unsqueeze(0)
        true_boxes = self.BOX2.unsqueeze(0)
        loss = bbox_loss(pred_boxes, true_boxes).sum().item()
        return loss

    def test_box_loss_iou(self):
        loss = self._get_box_loss(losses.BoxLoss.IoUType.IoU, xywh=False)

        self.assertAlmostEqual(loss, 0.38, places=2)

        iou = 1.0 - loss
        intersection = iou * 35000
        self.assertAlmostEqual(intersection, 21600, places=2)
        union = 21600 / iou
        self.assertAlmostEqual(union, 35000, places=2)

    def test_box_loss_giou(self):
        loss = self._get_box_loss(losses.BoxLoss.IoUType.GIoU, xywh=False)
        expected_iou = 0.62
        extra_term = 0.0196  # (smallest enclosing box - union) / union)
        expected_giou_loss = 1.0 - (expected_iou - extra_term)
        self.assertAlmostEqual(loss, expected_giou_loss, places=2)

    def test_box_loss_diou(self):
        loss = self._get_box_loss(losses.BoxLoss.IoUType.DIoU, xywh=False)
        """
        center_box1 --> (125, 200)
        center_box2 --> (150, 215)
        top_left_smallest_enclosing --> (50, 100)
        bottom_right_smallest_enclosing --> (220, 310)
        """
        r_squared = 850
        c_squared = 73000
        expected_iou = 0.62
        expected_diou_term = r_squared / c_squared
        expected_diou_loss = 1.0 - expected_iou + expected_diou_term
        self.assertAlmostEqual(loss, expected_diou_loss, places=2)

    def test_box_loss_ciou(self):
        loss = self._get_box_loss(losses.BoxLoss.IoUType.CIoU, xywh=False)

        r_squared = 850
        c_squared = 73000
        expected_iou = 0.62
        expected_diou_term = r_squared / c_squared
        box1_w = torch.tensor(150)
        box1_h = torch.tensor(200)
        box2_w = torch.tensor(140)
        box2_h = torch.tensor(190)
        v = (4 / math.pi**2) * torch.pow(
            torch.atan(box2_w / box2_h) - torch.atan(box1_w / box1_h), 2
        )
        eps = 1e-7
        alpha = v / (v - expected_iou + (1 + eps))
        expected_ciou_loss = 1.0 - expected_iou + expected_diou_term + alpha * v
        self.assertAlmostEqual(loss, expected_ciou_loss.item(), places=2)

    ################################################################################

    def test_box_loss_error_raise(self):
        bbox_loss = losses.BoxLoss(losses.BoxLoss.IoUType.IoU)
        pred_boxes = torch.Tensor([[0.5, 0.5, 1.0, 1.0], [0.5, 0.5, 1.0, 1.0]])
        true_boxes = torch.Tensor(
            [[0.4, 0.4, 1.0, 1.0], [0.1, 0.6, 1.0, 1.0], [0.5, 0.5, 1.0, 1.0]]
        )
        with self.assertRaises(AssertionError):
            bbox_loss(pred_boxes, true_boxes)

    def test_box_loss_error_raise2(self):
        bbox_loss = losses.BoxLoss(losses.BoxLoss.IoUType.IoU)
        pred_boxes = torch.Tensor([[0.5, 0.5, 1.0, 1.0]])
        true_boxes = torch.Tensor(
            [[0.4, 0.4, 1.0, 1.0], [0.1, 0.6, 1.0, 1.0], [0.5, 0.5, 1.0, 1.0]]
        )
        with self.assertRaises(AssertionError):
            bbox_loss(pred_boxes, true_boxes)


if __name__ == "__main__":
    unittest.main()
