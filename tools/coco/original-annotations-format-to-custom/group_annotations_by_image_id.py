import argparse
import pathlib

import custom_yolo_lib.dataset.coco.raw_annotations_parser as coco_raw_parser


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_path", type=pathlib.Path, required=True)
    return parser.parse_args()


def main(annotations_path: pathlib.Path):
    parser = coco_raw_parser.RawCOCOAnnotationsParser(annotations_path)
    parser.parse_data()
    grouped_annotations_path = annotations_path.parent / annotations_path.name.replace(
        ".json", "_grouped_by_image_id.json"
    )
    parser.write_data(grouped_annotations_path)


if __name__ == "__main__":
    args = parse_args()
    main(args.annotations_path)
