import dataclasses

import torch

import custom_yolo_lib.process.bbox


@dataclasses.dataclass
class Object:
    bbox: custom_yolo_lib.process.bbox.Bbox
    class_id: int

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [self.bbox.x, self.bbox.y, self.bbox.w, self.bbox.h, self.class_id]
        )
