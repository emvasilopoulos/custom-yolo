from typing import Any, Dict, List
import json
import pathlib
import tqdm


class RawCOCOAnnotationsParser:

    def __init__(self, annotations_path: pathlib.Path):
        self.annotations_path = annotations_path
        self.objects_by_image_id: Dict[str, List[Any]] = {}
        self.data_to_store = {}
        self.images = None
        self.annotations = None

    def parse_data(self):
        self._load_json_data()
        self.group_objects_by_image_id()
        self.data_to_store["annotations_grouped_by_image_id"] = self.objects_by_image_id
        self.data_to_store["images"] = self.images

    def write_data(self, output_path: pathlib.Path):
        with open(output_path, "w") as f:
            json.dump(self.data_to_store, f)

    def _load_json_data(self):
        with open(self.annotations_path, "r") as f:
            data = json.load(f)
        self.images = data["images"]
        self.annotations = data["annotations"]

    def group_objects_by_image_id(self):
        if self.images is None or self.annotations is None:
            self._load_json_data()

        iterator = tqdm.tqdm(
            self.annotations,
            total=len(self.annotations),
            desc="Grouping objects by image id",
        )
        for annotation in iterator:
            image_id = annotation["image_id"]
            if image_id not in self.objects_by_image_id:
                self.objects_by_image_id[image_id] = []
            self.objects_by_image_id[image_id].append(annotation)
        return self.objects_by_image_id


def sama_coco_to_original_coco(
    annotations_dir_path: pathlib.Path,
) -> None:
    """
    Args:
        annotations_dir_path (pathlib.Path): Path to the directory containing the annotations files, which are separated into parts. For example,
        sama_coco_coco_format_val_0.json, sama_coco_coco_format_val_1.json, etc.
    """
    annotations_parts_path = list(annotations_dir_path.glob("*.json"))
    if len(annotations_parts_path) < 1:
        raise ValueError(
            f"Annotations path {annotations_dir_path} does not contain any json files."
        )
    combined_annotations = {
        "info": {},
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": [],
    }
    for annotations_part_path in annotations_parts_path:
        with open(annotations_part_path, "r") as f:
            annotations_part = json.load(f)
        if combined_annotations["info"] == {}:
            combined_annotations["info"] = annotations_part["info"]
        if combined_annotations["licenses"] == []:
            combined_annotations["licenses"] = annotations_part["licenses"]
        combined_annotations["licenses"].extend(annotations_part["licenses"])
        combined_annotations["categories"].extend(annotations_part["categories"])
        combined_annotations["images"].extend(annotations_part["images"])
        combined_annotations["annotations"].extend(annotations_part["annotations"])
    output_path = annotations_dir_path / f"combined.json"
    with open(output_path, "w") as f:
        json.dump(combined_annotations, f)
