import unittest
import torch
import math

import custom_yolo_lib.process.bbox.metrics.iou


def ultralytics_bbox_iou(
    box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7
):
    """
    Calculate the Intersection over Union (IoU) between bounding boxes.

    This function supports various shapes for `box1` and `box2` as long as the last dimension is 4.
    For instance, you may pass tensors shaped like (4,), (N, 4), (B, N, 4), or (B, N, 1, 4).
    Internally, the code will split the last dimension into (x, y, w, h) if `xywh=True`,
    or (x1, y1, x2, y2) if `xywh=False`.

    Args:
        box1 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        box2 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format.
        GIoU (bool, optional): If True, calculate Generalized IoU.
        DIoU (bool, optional): If True, calculate Distance IoU.
        CIoU (bool, optional): If True, calculate Complete IoU.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(
            b2_x1
        )  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2)
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # center dist**2
            if (
                CIoU
            ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return (
            iou - (c_area - union) / c_area
        )  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


class TestBboxIoU(unittest.TestCase):

    def test_iou_xywh(self):
        box1 = torch.tensor([0, 0, 4, 4])
        box2 = torch.tensor([0, 0, 4, 4])
        expected_iou = ultralytics_bbox_iou(box1, box2, xywh=True)
        computed_iou = custom_yolo_lib.process.bbox.metrics.iou.bbox_iou(
            box1, box2, xywh=True
        )
        self.assertEqual(computed_iou.item(), expected_iou)

    def test_iou_xyxy(self):
        box1 = torch.tensor([0, 0, 4, 4])
        box2 = torch.tensor([2, 2, 6, 6])
        expected_iou = ultralytics_bbox_iou(box1, box2, xywh=False)
        computed_iou = custom_yolo_lib.process.bbox.metrics.iou.bbox_iou(
            box1, box2, xywh=False
        )
        self.assertEqual(computed_iou.item(), expected_iou)

    def test_giou(self):
        box1 = torch.tensor([0, 0, 4, 4])
        box2 = torch.tensor([2, 2, 6, 6])
        expected_iou = ultralytics_bbox_iou(box1, box2, xywh=False, GIoU=True)
        computed_giou = custom_yolo_lib.process.bbox.metrics.iou.bbox_giou(
            box1, box2, xywh=False
        )
        self.assertEqual(computed_giou.item(), expected_iou)

    def test_diou(self):
        box1 = torch.tensor([0, 0, 4, 4])
        box2 = torch.tensor([1, 1, 5, 5])
        expected_iou = ultralytics_bbox_iou(box1, box2, xywh=False, DIoU=True)
        computed_diou = custom_yolo_lib.process.bbox.metrics.iou.bbox_diou(
            box1, box2, xywh=False
        )
        self.assertEqual(computed_diou.item(), expected_iou)

    def test_ciou(self):
        box1 = torch.tensor([0, 0, 4, 4])
        box2 = torch.tensor([1, 1, 3, 3])
        expected_iou = ultralytics_bbox_iou(box1, box2, xywh=False, CIoU=True)
        computed_ciou = custom_yolo_lib.process.bbox.metrics.iou.bbox_ciou(
            box1, box2, xywh=False
        )
        self.assertEqual(computed_ciou.item(), expected_iou)

    def test_zero_intersection(self):
        box1 = torch.tensor([0, 0, 1, 1])
        box2 = torch.tensor([2, 2, 3, 3])
        expected_iou = ultralytics_bbox_iou(box1, box2, xywh=False)
        computed_iou = custom_yolo_lib.process.bbox.metrics.iou.bbox_iou(
            box1, box2, xywh=False
        )
        self.assertEqual(computed_iou.item(), expected_iou)

    def test_batch_input(self):
        box1 = torch.tensor([[0, 0, 4, 4], [2, 2, 6, 6]])
        box2 = torch.tensor([[2, 2, 6, 6], [0, 0, 4, 4]])
        computed_iou = custom_yolo_lib.process.bbox.metrics.iou.bbox_iou(
            box1, box2, xywh=False
        )
        expected_iou = ultralytics_bbox_iou(box1, box2, xywh=False)
        self.assertTrue(
            torch.allclose(computed_iou, expected_iou), "Batch input IoU mismatch"
        )

    def test_batch_input2(self):
        bboxes1 = torch.tensor(
            [[[0, 0, 4, 4], [2, 2, 6, 6]], [[0, 0, 4, 4], [2, 2, 6, 6]]]
        )
        bboxes2 = torch.tensor(
            [[[2, 2, 6, 6], [0, 0, 4, 4]], [[2, 2, 6, 6], [0, 0, 4, 4]]]
        )
        expected_iou = ultralytics_bbox_iou(bboxes1, bboxes2, xywh=False)
        computed_iou = custom_yolo_lib.process.bbox.metrics.iou.bbox_iou(
            bboxes1, bboxes2, xywh=False
        )

        self.assertTrue(
            torch.allclose(computed_iou, expected_iou), "Batch input IoU mismatch"
        )


if __name__ == "__main__":
    unittest.main()
