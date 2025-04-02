import dataclasses

import torch

import custom_yolo_lib.process.bbox


@dataclasses.dataclass
class Object:
    bbox: custom_yolo_lib.process.bbox.Bbox
    class_id: int
