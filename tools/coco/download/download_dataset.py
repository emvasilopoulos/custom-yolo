import argparse
import pathlib
from typing import Tuple

from custom_yolo_lib.dataset.coco.downloader import *


def parse_args() -> Tuple[bool, bool, bool, pathlib.Path]:
    parser = argparse.ArgumentParser(description="Download COCO dataset")
    parser.add_argument(
        "--save_dir",
        type=pathlib.Path,
        default=pathlib.Path("/home/manos/custom_yolo_lib/coco_data"),
        help="Directory to save the dataset",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Download the training images",
    )
    parser.add_argument(
        "--val",
        action="store_true",
        help="Download the validation images",
    )
    parser.add_argument(
        "--annotations",
        action="store_true",
        help="Download the annotations (both training & validation)",
    )
    args = parser.parse_args()
    return args.train, args.val, args.annotations, pathlib.Path(args.save_dir)


def main(train: bool, val: bool, annotations: bool, save_dir: pathlib.Path):
    if not save_dir.exists():
        print(f"Creating directory {save_dir}")
        save_dir.mkdir(parents=True, exist_ok=True)
    if annotations:
        download_train_val_annotations_2017(save_dir=save_dir)
    if train:
        download_train_images_2017(save_dir=save_dir)
    if val:
        download_val_images_2017(save_dir=save_dir)
    if not (train or val or annotations):
        print("No options selected. Use --help for more information.")


if __name__ == "__main__":
    train, val, annotations, save_dir = parse_args()
    main(train, val, annotations, save_dir)
