import argparse

import custom_yolo_lib.dataset.coco.raw_annotations_parser as raw_annotations_parser
import pathlib


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert SAMA COCO format annotations to original COCO format."
    )
    parser.add_argument(
        "--annotations-dir",
        type=pathlib.Path,
        help="Path to the directory containing the annotations files, which are separated into parts. For example, sama_coco_coco_format_val_0.json, sama_coco_coco_format_val_1.json, etc.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    annotations_dir_path = args.annotations_dir
    raw_annotations_parser.sama_coco_to_original_coco(annotations_dir_path)

if __name__ == "__main__":
    main()
