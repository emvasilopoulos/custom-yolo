import pathlib
from typing import Any, Dict, List, Tuple

import torch

import custom_yolo_lib.io.read
import custom_yolo_lib.dataset.coco.tasks.base
import custom_yolo_lib.process.bbox

ANNOTATIONS_GROUPED_KEY = "annotations_grouped_by_image_id"


class COCOInstances2017(
    custom_yolo_lib.dataset.coco.tasks.base.COCODatasetInstances2017
):
    """
    Supports annotations after running tools/coco/group_annotations_by_image_id.py
    which groups annotations by image_id.
    """

    def _get_coco_item(
        self, idx: int
    ) -> Tuple[torch.Tensor, List[custom_yolo_lib.process.bbox.Bbox]]:
        self.annotations = self.annotations[self.image_ids[idx]]
        raise NotImplementedError()

    def _define_length(self):
        return len(self.image_ids)

    def get_pair(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError()

    def _read_annotations(self, annotations_path: pathlib.Path):
        all_annotations = custom_yolo_lib.io.read.read_csv(annotations_path).groupby(
            "image_id"
        )
        self.annotations = all_annotations
        self.original_groups_indexed = list(self.annotations.groups.keys())
