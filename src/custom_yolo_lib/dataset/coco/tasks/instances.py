import pathlib
from typing import List, Tuple

import pandas as pd

import custom_yolo_lib.image_size
import custom_yolo_lib.io.read
import custom_yolo_lib.dataset.coco.tasks.base
import custom_yolo_lib.dataset.object
import custom_yolo_lib.process.bbox

ANNOTATIONS_GROUPED_KEY = "annotations_grouped_by_image_id"


class COCOInstances2017(
    custom_yolo_lib.dataset.coco.tasks.base.COCODatasetInstances2017
):
    """
    Supports annotations after running tools/coco/group_annotations_by_image_id.py
    which groups annotations by image_id.
    """

    def _extract_image_path(self, sample: pd.DataFrame) -> pathlib.Path:
        image_filename = self._image_file_name_from_id(sample["image_id"].values[0])
        return self.images_dir / image_filename

    def _is_in_classes(self, class_id: int) -> bool:
        # TODO: replace with dictionary search
        return 0 < class_id < len(self.desired_classes)

    def _norm(self, coord: float, dim_size: int) -> float:
        return max(0, min(1, coord / dim_size))

    def _extract_objects(
        self, sample: pd.DataFrame, image_size: custom_yolo_lib.image_size.ImageSize
    ):
        x1s = sample["x1"].values
        y1s = sample["y1"].values
        ws = sample["w"].values
        hs = sample["h"].values
        is_crowds = sample["iscrowd"].values
        class_ids = sample["category_id"].values
        objects_ = []
        for x1, y1, w, h, class_id, is_crowd in zip(
            x1s, y1s, ws, hs, class_ids, is_crowds
        ):
            if not self._is_in_classes(class_id):
                continue

            if is_crowd:
                continue

            bbox = custom_yolo_lib.process.bbox.Bbox(
                x=self._norm(x1, image_size.width),
                y=self._norm(y1, image_size.height),
                w=self._norm(w, image_size.width),
                h=self._norm(h, image_size.height),
                is_normalized=True,
            )
            objects_.append(
                custom_yolo_lib.dataset.object.Object(
                    bbox=bbox,
                    class_id=class_id - 1,  # 0-based index
                )
            )
        return objects_

    def _get_coco_item(
        self, idx: int
    ) -> Tuple[pathlib.Path, List[custom_yolo_lib.dataset.object.Object]]:
        groups = self.original_groups_indexed[idx]
        sample = self.annotations.get_group(groups)

        image_path = self._extract_image_path(sample)
        image_size = custom_yolo_lib.io.read.read_image_dimensions(image_path)
        objects_ = self._extract_objects(sample, image_size)

        return image_path, objects_

    def _define_length(self):
        return len(self.original_groups_indexed)

    def _read_annotations(self, annotations_path: pathlib.Path):
        self.annotations = custom_yolo_lib.io.read.read_csv(annotations_path).groupby(
            "image_id"
        )
        self.original_groups_indexed = list(self.annotations.groups.keys())
