import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Custom YOLO model")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/manos/custom-yolo/coco_data",
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--experiment_path",
        type=str,
        default=__file__.replace(".py", ""),
        help="Path to the experiment directory",
    )

    return parser.parse_args()
