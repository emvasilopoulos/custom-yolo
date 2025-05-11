import pathlib
import argparse

import cv2
import torch

torch.manual_seed(42)

import numpy as np

np.printoptions(threshold=np.inf)


import custom_yolo_lib.dataset.coco.tasks.instances
import custom_yolo_lib.dataset.coco.tasks.loader
import custom_yolo_lib.experiments_utils
import custom_yolo_lib.image_size
import custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based
import custom_yolo_lib.model.e2e.anchor_based.loss
import custom_yolo_lib.model.e2e.anchor_based.training_utils
import custom_yolo_lib.training.utils


BASE_LR = 0.01 / 64
EXPERIMENT_NAME = "debug"
WARMUP_EPOCHS = 3
EPOCHS = 12
NUM_CLASSES = 80
BATCH_SIZE = 4
LR = BASE_LR * BATCH_SIZE
MOMENTUM = 0.9
DECAY = 0.001
IMAGE_SIZE = custom_yolo_lib.image_size.ImageSize(640, 640)
CLASS_LOSS_GAIN = 1.0
OBJECTNESS_LOSS_GAIN = 1.0
BOX_LOSS_GAIN = 1.0
OBJECTNESS_LOSS_SMALL_MAP_GAIN = 4.0  # bigger grid 80x80 results in smaller loss if BCE
OBJECTNESS_LOSS_MEDIUM_MAP_GAIN = 1.0
OBJECTNESS_LOSS_LARGE_MAP_GAIN = 0.4
# torch.set_anomaly_enabled(True)


def init_model(
    device: torch.device,
) -> custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based.YOLOModel:
    model = custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based.YOLOModel(
        num_classes=NUM_CLASSES, training=True
    )
    model.to(device)
    return model


def init_optimizer(
    model: custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based.YOLOModel,
) -> torch.optim.Optimizer:
    parameters_grouped = custom_yolo_lib.training.utils.get_params_grouped(model)
    optimizer = torch.optim.AdamW(
        parameters_grouped.with_weight_decay,
        lr=LR,
        betas=(MOMENTUM, 0.999),
        weight_decay=DECAY,
    )
    optimizer.add_param_group(
        {"params": parameters_grouped.bias, "weight_decay": DECAY}
    )
    optimizer.add_param_group(
        {"params": parameters_grouped.no_weight_decay, "weight_decay": 0.0}
    )
    return optimizer


def init_losses(
    model: custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based.YOLOModel,
    device: torch.device,
):
    predictions_s, predictions_m, predictions_l = model.train_forward2(
        torch.zeros((1, 3, IMAGE_SIZE.height, IMAGE_SIZE.width)).to(device)
    )

    small_map_anchors, medium_map_anchors, large_map_anchors = (
        custom_yolo_lib.model.e2e.anchor_based.training_utils.get_anchors_as_bbox_tensors(
            device
        )
    )
    loss_s = custom_yolo_lib.model.e2e.anchor_based.loss.YOLOLossPerFeatureMapV2(
        num_classes=NUM_CLASSES,
        feature_map_anchors=small_map_anchors,
        grid_size_h=predictions_s.shape[3],
        grid_size_w=predictions_s.shape[4],
    )
    loss_m = custom_yolo_lib.model.e2e.anchor_based.loss.YOLOLossPerFeatureMapV2(
        num_classes=NUM_CLASSES,
        feature_map_anchors=medium_map_anchors,
        grid_size_h=predictions_m.shape[3],
        grid_size_w=predictions_m.shape[4],
    )
    loss_l = custom_yolo_lib.model.e2e.anchor_based.loss.YOLOLossPerFeatureMapV2(
        num_classes=NUM_CLASSES,
        feature_map_anchors=large_map_anchors,
        grid_size_h=predictions_l.shape[3],
        grid_size_w=predictions_l.shape[4],
    )
    return loss_s, loss_m, loss_l


def init_dataloaders(dataset_path: pathlib.Path):
    classes = [i for i in range(NUM_CLASSES)]
    train_dataset = custom_yolo_lib.dataset.coco.tasks.instances.COCOInstances2017(
        dataset_path,
        "train",
        expected_image_size=IMAGE_SIZE,
        classes=classes,
        is_sama=True,
    )
    training_loader = custom_yolo_lib.dataset.coco.tasks.loader.COCODataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    return training_loader


def draw_targets_from_dataloader(
    image: np.ndarray,
    target: torch.Tensor,
    sample_dir: pathlib.Path,
    sample_in_batch: int,
):
    # draw targets on image
    for obj in target:
        xc, yc, w, h, class_id = obj
        xc = int(xc.item() * IMAGE_SIZE.width)
        yc = int(yc.item() * IMAGE_SIZE.height)
        w = int(w.item() * IMAGE_SIZE.width)
        h = int(h.item() * IMAGE_SIZE.height)
        x1 = int((xc - w / 2))
        y1 = int((yc - h / 2))
        x2 = int((xc + w / 2))
        y2 = int((yc + h / 2))
        class_id = int(class_id.item())
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            str(class_id),
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    cv2.imwrite(str(sample_dir / f"sample_{sample_in_batch}_debug.jpg"), image)


def _write_to_txt_targets_in_grid(
    sample_dir: pathlib.Path, sample_in_batch: int, target_in_grid, feats_size: str
):
    anchors, feats, grid_h, grid_w = target_in_grid.shape
    with open(
        str(sample_dir / f"sample_{sample_in_batch}_in_grid_{feats_size}.txt"), "w"
    ) as f:
        f.write(f"target_in_grid_{feats_size} shape: {target_in_grid.shape}\n")
        for anchor_i in range(anchors):
            f.write(f"============== ANCHOR {anchor_i} ==============\n")
            for y in range(grid_h):
                for x in range(grid_w):
                    f.write(f"grid_y: {y}, grid_x: {x}\n")
                    f.write(f"{target_in_grid[anchor_i, :, y, x]}\n")
                    f.write("--------------------------------------------\n")


def write_to_txt_targets_in_grid(
    sample_dir: pathlib.Path,
    sample_in_batch: int,
    target_in_grid_s,
    target_in_grid_m,
    target_in_grid_l,
):
    # write targets_in_grid_s to...
    # target_in_grid_s is a tensor of shape (num_anchors, num_classes + 5, grid_h, grid_w)

    _write_to_txt_targets_in_grid(sample_dir, sample_in_batch, target_in_grid_s, "s")
    _write_to_txt_targets_in_grid(sample_dir, sample_in_batch, target_in_grid_m, "m")
    _write_to_txt_targets_in_grid(sample_dir, sample_in_batch, target_in_grid_l, "l")


def draw_targets_from_targets_in_grid(
    image: np.ndarray,
    target_in_grid,
    target_mask,
    sample_dir: pathlib.Path,
    sample_in_batch: int,
    feat_size: str,
):
    num_anchors = target_in_grid.shape[0]
    anchor_colors = [(0, 200, 0), (200, 0, 0), (0, 0, 200)]
    for anchor_i in range(num_anchors):
        anchor_targets = target_in_grid[anchor_i]
        anchor_targets_mask = target_mask[anchor_i]
        grid_x = torch.arange(anchor_targets.shape[1], device=anchor_targets.device)
        grid_y = torch.arange(anchor_targets.shape[0], device=anchor_targets.device)
        for i in range(anchor_targets.shape[0]):
            anchor_targets[i, :, 0].add_(grid_x)
        for i in range(anchor_targets.shape[1]):
            anchor_targets[:, i, 1].add_(grid_y)
        anchor_targets[:, :, 0].div_(anchor_targets.shape[1])
        anchor_targets[:, :, 1].div_(anchor_targets.shape[0])

        anchor_target_bboxes = anchor_targets[:, :, :4][anchor_targets_mask]
        x = anchor_target_bboxes[:, 0] * IMAGE_SIZE.width
        y = anchor_target_bboxes[:, 1] * IMAGE_SIZE.height
        w = anchor_target_bboxes[:, 2] * IMAGE_SIZE.width
        h = anchor_target_bboxes[:, 3] * IMAGE_SIZE.height
        x1s = x - w / 2
        y1s = y - h / 2
        x2s = x + w / 2
        y2s = y + h / 2
        class_ids = torch.argmax(anchor_targets[:, :, 5:], dim=2)[anchor_targets_mask]
        for x1, y1, x2, y2, class_id in zip(x1s, y1s, x2s, y2s, class_ids):
            x1 = int(x1.item())
            y1 = int(y1.item())
            x2 = int(x2.item())
            y2 = int(y2.item())
            cv2.rectangle(
                image,
                (x1 + 5 * anchor_i, y1),
                (x2 + 5 * anchor_i, y2),
                anchor_colors[anchor_i],
                2,
            )
            cv2.putText(
                image,
                f"{class_id}-{anchor_i}",
                (x1 + 5 * anchor_i, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                anchor_colors[anchor_i],
                2,
            )
    cv2.imwrite(
        str(sample_dir / f"sample_{sample_in_batch}_{feat_size}_debug.jpg"), image
    )


def infe_one_batch(
    model: custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based.YOLOModel,
    training_loader: torch.utils.data.DataLoader,
    loss_s: custom_yolo_lib.model.e2e.anchor_based.loss.YOLOLossPerFeatureMapV2,
    loss_m: custom_yolo_lib.model.e2e.anchor_based.loss.YOLOLossPerFeatureMapV2,
    loss_l: custom_yolo_lib.model.e2e.anchor_based.loss.YOLOLossPerFeatureMapV2,
    experiment_path: pathlib.Path,
    device: torch.device = torch.device("cuda:0"),
):
    model.train()
    for i, coco_batch in enumerate(training_loader):

        images = coco_batch.images_batch.to(device)
        targets = [t.to(device) for t in coco_batch.objects_batch]

        predictions_s, predictions_m, predictions_l = model.train_forward2(images)

        _, targets_in_grid_s, targets_mask_s = loss_s.debug_forward(
            predictions_s, targets
        )  #
        _, targets_in_grid_m, targets_mask_m = loss_m.debug_forward(
            predictions_m, targets
        )
        _, targets_in_grid_l, targets_mask_l = loss_l.debug_forward(
            predictions_l, targets
        )
        break

    for i, (
        image,
        target,
        target_in_grid_s,
        target_mask_s,
        target_in_grid_m,
        target_mask_m,
        target_in_grid_l,
        target_mask_l,
    ) in enumerate(
        zip(
            images,
            targets,
            targets_in_grid_s,
            targets_mask_s,
            targets_in_grid_m,
            targets_mask_m,
            targets_in_grid_l,
            targets_mask_l,
        )
    ):
        # create sample_i subdir
        sample_dir = experiment_path / f"sample_{i}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # convert image to opencv format
        image = image.permute(1, 2, 0).cpu().numpy() * 255.0
        image = image.astype("uint8")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(sample_dir / f"sample_{i}.jpg"), image)
        # write targets to txt file
        with open(str(sample_dir / f"sample_{i}.txt"), "w") as f:
            for obj in target:
                for item in obj:
                    f.write(f"{item.item()}, ")
                f.write("\n")

        img1 = image.copy()
        draw_targets_from_dataloader(img1, target, sample_dir, i)
        write_to_txt_targets_in_grid(
            sample_dir,
            i,
            target_in_grid_s,
            target_in_grid_m,
            target_in_grid_l,
        )

        target_in_grid_s = target_in_grid_s.permute(0, 2, 3, 1)
        target_in_grid_m = target_in_grid_m.permute(0, 2, 3, 1)
        target_in_grid_l = target_in_grid_l.permute(0, 2, 3, 1)

        # draw targets_in_grid_s on image
        img2 = image.copy()
        draw_targets_from_targets_in_grid(
            img2,
            target_in_grid_s,
            target_mask_s,
            sample_dir,
            i,
            "small",
        )
        # draw targets_in_grid_m on image
        img3 = image.copy()
        draw_targets_from_targets_in_grid(
            img3,
            target_in_grid_m,
            target_mask_m,
            sample_dir,
            i,
            "medium",
        )
        # draw targets_in_grid_l on image
        img4 = image.copy()
        draw_targets_from_targets_in_grid(
            img4,
            target_in_grid_l,
            target_mask_l,
            sample_dir,
            i,
            "large",
        )
        break
        print("- Done with sample", i)


def session_loop(
    model: custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based.YOLOModel,
    training_loader: torch.utils.data.DataLoader,
    loss_s: custom_yolo_lib.model.e2e.anchor_based.loss.YOLOLossPerFeatureMapV2,
    loss_m: custom_yolo_lib.model.e2e.anchor_based.loss.YOLOLossPerFeatureMapV2,
    loss_l: custom_yolo_lib.model.e2e.anchor_based.loss.YOLOLossPerFeatureMapV2,
    experiment_path: pathlib.Path,
    device: torch.device = torch.device("cuda:0"),
):
    infe_one_batch(
        model,
        training_loader,
        loss_s,
        loss_m,
        loss_l,
        experiment_path,
        device,
    )


def main(dataset_path: pathlib.Path, experiment_path: pathlib.Path):
    experiment_path /= EXPERIMENT_NAME
    experiment_path.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Why bother bro?.")
    device = torch.device("cuda:0")

    model = init_model(device)
    training_loader = init_dataloaders(dataset_path)

    # loss_s, loss_m, loss_l = init_losses(device)
    loss_s, loss_m, loss_l = init_losses(model, device)

    session_loop(
        model,
        training_loader,
        loss_s,
        loss_m,
        loss_l,
        experiment_path,
        device,
    )


def parse_args():

    parser = argparse.ArgumentParser(description="Train YOLOv5 model")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/manos/custom-yolo/coco_data",
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--experiment_path",
        type=str,
        # required=True,
        default="/home/manos/custom-yolo/experiments",
        help="Path to the experiment directory",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_path = pathlib.Path(args.dataset_path)
    experiment_path = pathlib.Path(args.experiment_path)
    main(dataset_path, experiment_path)
