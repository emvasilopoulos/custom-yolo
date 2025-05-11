import pathlib

import pandas as pd

import custom_yolo_lib.io.read
import custom_yolo_lib.dataset.coco.tasks.instances


def convert_grouped_instances_json_to_csv(json_path: pathlib.Path):
    annotations = custom_yolo_lib.io.read.read_json(json_path)[
        custom_yolo_lib.dataset.coco.tasks.instances.ANNOTATIONS_GROUPED_KEY
    ]
    annotations_for_df = {
        "image_id": [],
        "category_id": [],
        "x1": [],
        "y1": [],
        "w": [],
        "h": [],
        "area": [],
        "iscrowd": [],
        "id": [],
        "segmentation": [],
    }

    for item in annotations.keys():
        for annotation in annotations[item]:
            annotations_for_df["image_id"].append(item)
            annotations_for_df["category_id"].append(annotation["category_id"])
            annotations_for_df["x1"].append(annotation["bbox"][0])
            annotations_for_df["y1"].append(annotation["bbox"][1])
            annotations_for_df["w"].append(annotation["bbox"][2])
            annotations_for_df["h"].append(annotation["bbox"][3])
            annotations_for_df["area"].append(annotation["area"])
            annotations_for_df["iscrowd"].append(annotation["iscrowd"])
            annotations_for_df["id"].append(annotation["id"])
            annotations_for_df["segmentation"].append(annotation.get("segmentation"))

    df = pd.DataFrame(annotations_for_df)
    df.to_csv(json_path.with_suffix(".csv"), index=False)
