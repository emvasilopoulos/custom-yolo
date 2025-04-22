# Dataset
import pathlib

import torch
import tqdm
import pandas as pd

import custom_yolo_lib.training.utils
import custom_yolo_lib.training.lr_scheduler
import custom_yolo_lib.image_size
import custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based
import custom_yolo_lib.dataset.coco.tasks.instances
import custom_yolo_lib.dataset.coco.tasks.loader

from custom_yolo_lib.dataset.coco.constants import MEDIUM_AREA_RANGE, SMALL_AREA_RANGE
import custom_yolo_lib.model.e2e.anchor_based.loss
import custom_yolo_lib.model.e2e.anchor_based.training_utils
import custom_yolo_lib.dataset.coco.tasks.loader

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    dataset_path = pathlib.Path("/home/manos/custom-yolo/coco_data")
    images_path = dataset_path / "images"
    train_images_path = images_path / "train2017"
    val_images_path = images_path / "val2017"
    annotations_path = dataset_path / "annotations"

    epochs = 300
    num_classes = 80
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    image_size = custom_yolo_lib.image_size.ImageSize(640, 640)

    model = custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based.YOLOModel(
        num_classes=num_classes, training=True
    )
    model.to(device)
    parameters_grouped = custom_yolo_lib.training.utils.get_params_grouped(model)
    lr = 0.0015
    momentum = 0.937
    decay = 0.001
    batch_size = 8
    optimizer = torch.optim.AdamW(
        parameters_grouped.bias, lr=lr, betas=(momentum, 0.999), weight_decay=0.0
    )
    optimizer.add_param_group(
        {"params": parameters_grouped.weight_decay, "weight_decay": decay}
    )
    optimizer.add_param_group(
        {"params": parameters_grouped.no_weight_decay, "weight_decay": 0.0}
    )
    scheduler = custom_yolo_lib.training.lr_scheduler.StepLRScheduler(
        optimizer, update_step_size=10000
    )

    classes = [i for i in range(num_classes)]

    val_dataset = custom_yolo_lib.dataset.coco.tasks.instances.COCOInstances2017(
        dataset_path, "val", expected_image_size=image_size, classes=classes
    )
    train_dataset = custom_yolo_lib.dataset.coco.tasks.instances.COCOInstances2017(
        dataset_path, "train", expected_image_size=image_size, classes=classes
    )
    training_loader = (
        custom_yolo_lib.dataset.coco.tasks.loader.COCODataLoaderThreeFeatureMaps(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
    )
    validation_loader = (
        custom_yolo_lib.dataset.coco.tasks.loader.COCODataLoaderThreeFeatureMaps(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
    )

    torch.manual_seed(42)

    SMALL_AREA_UPPER_BOUND = SMALL_AREA_RANGE[1]
    MEDIUM_AREA_LOWER_BOUND = MEDIUM_AREA_RANGE[0]
    MEDIUM_AREA_UPPER_BOUND = MEDIUM_AREA_RANGE[1]
    LARGE_AREA_LOWER_BOUND = MEDIUM_AREA_UPPER_BOUND

    small_map_anchors, medium_map_anchors, large_map_anchors = (
        custom_yolo_lib.model.e2e.anchor_based.training_utils.get_anchors_as_bbox_tensors(
            device
        )
    )
    loss_s = custom_yolo_lib.model.e2e.anchor_based.loss.YOLOLossPerFeatureMap(
        num_classes=num_classes,
        feature_map_anchors=small_map_anchors,
    )
    loss_m = custom_yolo_lib.model.e2e.anchor_based.loss.YOLOLossPerFeatureMap(
        num_classes=num_classes,
        feature_map_anchors=medium_map_anchors,
    )
    loss_l = custom_yolo_lib.model.e2e.anchor_based.loss.YOLOLossPerFeatureMap(
        num_classes=num_classes,
        feature_map_anchors=large_map_anchors,
    )

    coco_batch: (
        custom_yolo_lib.dataset.coco.tasks.loader.COCODataLoaderThreeFeatureMapBatch
    )
    # Training loop
    training_step = 0
    validation_step = 0
    for epoch in range(epochs):

        # TRAINING
        training_session_data = {
            "bbox_loss_avg_featmap": [],
            "objectness_loss_avg_featmap": [],
            "class_loss_avg_featmap": [],
            "total_loss_avg_featmap": [],
            "epoch": [],
            "step": [],
        }
        tqdm_obj = tqdm.tqdm(training_loader)
        model.train()
        print(f"Training epoch {epoch + 1}/{epochs} | Losses:")
        for i, coco_batch in enumerate(tqdm_obj):

            images = coco_batch.images_batch.to(device)
            targets_s = [t.to(device) for t in coco_batch.small_objects_batch]
            targets_m = [t.to(device) for t in coco_batch.medium_objects_batch]
            targets_l = [t.to(device) for t in coco_batch.large_objects_batch]

            optimizer.zero_grad()
            predictions_s, predictions_m, predictions_l = model(images)

            loss_s_ = loss_s(predictions_s, targets_s)
            loss_m_ = loss_m(predictions_m, targets_m)
            loss_l_ = loss_l(predictions_l, targets_l)
            loss = (loss_s_[3] + loss_m_[3] + loss_l_[3]) / 3
            loss.backward()

            scheduler.update_loss(loss)
            optimizer.step()

            avg_bbox_loss = (loss_s_[0] + loss_m_[0] + loss_l_[0]) / 3
            avg_objectness_loss = (loss_s_[1] + loss_m_[1] + loss_l_[1]) / 3
            avg_class_loss = (loss_s_[2] + loss_m_[2] + loss_l_[2]) / 3
            tqdm_obj.set_description(
                f"Total: {loss.item():.4f} | BBox: {avg_bbox_loss.item():.4f} | Obj: {avg_objectness_loss.item():.4f} | Class: {avg_class_loss.item():.4f}"
            )
            training_session_data["bbox_loss_avg_featmap"].append(avg_bbox_loss.item())
            training_session_data["objectness_loss_avg_featmap"].append(
                avg_objectness_loss.item()
            )
            training_session_data["class_loss_avg_featmap"].append(
                avg_class_loss.item()
            )
            training_session_data["total_loss_avg_featmap"].append(loss.item())
            training_session_data["epoch"].append(epoch)
            training_session_data["step"].append(training_step)
            for i, lr in enumerate(scheduler.get_lr()):
                if f"lr-{i}" not in training_session_data:
                    training_session_data[f"lr-{i}"] = []
                training_session_data[f"lr-{i}"].append(lr)
            training_step += 1

        # Store training data
        model_state = model.state_dict()
        torch.save(model_state, f"model_epoch_{epoch}.pth")
        pd.DataFrame(training_session_data).to_csv(
            f"training_session_data_epoch_{epoch}.csv"
        )

        # VALIDATION
        validation_session_data = {
            "bbox_loss_avg_featmap": [],
            "objectness_loss_avg_featmap": [],
            "class_loss_avg_featmap": [],
            "total_loss_avg_featmap": [],
            "epoch": [],
            "step": [],
        }
        tqdm_obj = tqdm.tqdm(validation_loader)
        model.eval()
        model.training = True
        print(f"Validation epoch {epoch + 1}/{epochs} | Losses:")
        with torch.no_grad():
            for i, coco_batch in enumerate(tqdm_obj):
                """
                targets.shape = (batch_size, MAX_ALLOWED_OBJECTS, 5 + num_classes) | 5-->(x, y, w, h, objectness)
                targets_n_objects.shape = (batch_size, 1)
                """

                images = coco_batch.images_batch.to(device)
                targets_s = [t.to(device) for t in coco_batch.small_objects_batch]
                targets_m = [t.to(device) for t in coco_batch.medium_objects_batch]
                targets_l = [t.to(device) for t in coco_batch.large_objects_batch]
                predictions_s, predictions_m, predictions_l = model(images)

                loss_s_ = loss_s(predictions_s, targets_s)
                loss_m_ = loss_m(predictions_m, targets_m)
                loss_l_ = loss_l(predictions_l, targets_l)

                loss = (loss_s_[3] + loss_m_[3] + loss_l_[3]) / 3
                avg_bbox_loss = (loss_s_[0] + loss_m_[0] + loss_l_[0]) / 3
                avg_objectness_loss = (loss_s_[1] + loss_m_[1] + loss_l_[1]) / 3
                avg_class_loss = (loss_s_[2] + loss_m_[2] + loss_l_[2]) / 3
                tqdm_obj.set_description(
                    f"Total: {loss.item():.4f} | BBox: {avg_bbox_loss.item():.4f} | Obj: {avg_objectness_loss.item():.4f} | Class: {avg_class_loss.item():.4f}"
                )
                validation_session_data["bbox_loss_avg_featmap"].append(
                    avg_bbox_loss.item()
                )
                validation_session_data["objectness_loss_avg_featmap"].append(
                    avg_objectness_loss.item()
                )
                validation_session_data["class_loss_avg_featmap"].append(
                    avg_class_loss.item()
                )
                validation_session_data["total_loss_avg_featmap"].append(loss.item())
                validation_session_data["epoch"].append(epoch)
                validation_session_data["step"].append(validation_step)
                validation_step += 1
        # Store validation data
        pd.DataFrame(validation_session_data).to_csv(
            f"validation_session_data_epoch_{epoch}.csv"
        )
