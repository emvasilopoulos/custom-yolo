import enum
from typing import Dict

import torch


COCODatasetSample = Dict[str, torch.Tensor]


class COCODatasetSampleKeys(enum.Enum):
    IMAGE_TENSOR = "image_tensor"
    OBJECTS_TENSOR = "objects_tensor"
    OBJECTS_COUNT = "objects_count"
