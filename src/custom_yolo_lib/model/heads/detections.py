import torch


class DetectionHead(torch.nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors=3):
        super(DetectionHead, self).__init__()
        self.pred = torch.nn.Conv2d(
            in_channels, num_anchors * (5 + num_classes), 1, 1, 0
        )

    def forward(self, x):
        return self.pred(x)
