import pathlib
import cv2
import torch
import custom_yolo_lib.model.e2e.anchor_based.bundled_anchor_based as anchor_models
import custom_yolo_lib.process.image.pipeline
import custom_yolo_lib.process.tensor
import custom_yolo_lib.process.normalize
import custom_yolo_lib.process.image.resize.fixed_ratio
import custom_yolo_lib.image_size
import custom_yolo_lib.io.read

import torchvision

#
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

model = anchor_models.YOLOModel(num_classes=80, training=False)
# To CPU
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
loaded_model = torch.load("model_best.pth", map_location=device)
model.load_state_dict(loaded_model)
model.eval()

input_pipeline = custom_yolo_lib.process.image.pipeline.ImagePipeline(
    dtype_converter=custom_yolo_lib.process.tensor.TensorDtypeConverter(torch.float32),
    normalize=custom_yolo_lib.process.normalize.SimpleImageNormalizer(),
)
padding_percent = 0.5
expected_image_size = custom_yolo_lib.image_size.ImageSize(
    width=640,
    height=640,
)
camera_image_size = custom_yolo_lib.image_size.ImageSize(
    width=frame_width,
    height=frame_height,
)
resize_fixed_ratio_components = (
    custom_yolo_lib.process.image.resize.fixed_ratio.ResizeFixedRatioComponents_v2(
        current_image_size=camera_image_size,
        expected_image_size=expected_image_size,
    )
)

image_path = pathlib.Path("frame.jpg")
with torch.no_grad():
    i = 0
    while True:
        i += 1
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        cv2.imwrite(image_path.as_posix(), frame)
        img_tensor = custom_yolo_lib.io.read.read_image_torchvision(image_path)

        input_tensor = input_pipeline(img_tensor)
        input_tensor = custom_yolo_lib.process.image.resize.fixed_ratio.resize_image_with_ready_components(
            input_tensor,
            fixed_ratio_components=resize_fixed_ratio_components,
            padding_percent=padding_percent,
            pad_value=114,
        )

        input_tensor = input_tensor.unsqueeze(0)
        # Run inference
        detections_batch = model(input_tensor)

        for detections in detections_batch:
            detections = detections.permute(1, 0, 2)
            detections = detections.reshape(detections.shape[0], -1)
            detections = detections.permute(1, 0)
            detections = detections[detections[:, 4] > 0.1]
            print("----------------")
            print(f"Detections with confidence > 0.1:\n{detections[:, :5]}")
            indices = torchvision.ops.nms(detections[:, :4], detections[:, 4], 0.1)
            detections = detections[indices]
            for i in range(detections.shape[0]):
                class_id = torch.argmax(detections[i, 5:])
                x1, y1, x2, y2, conf = detections[i, :5]
                xc = int(x1.item() * frame.shape[1])
                yc = int(y1.item() * frame.shape[0])
                w = int(x2.item() * frame.shape[1])
                h = int(y2.item() * frame.shape[0])
                x1 = xc - w // 2
                y1 = yc - h // 2
                x2 = xc + w // 2
                y2 = yc + h // 2
                conf = conf.item()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{class_id} - {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
                # print(f"Detected: {cls} at ({x1}, {y1}), ({x2}, {y2}) with confidence {conf}")

        # Process and display detections
        # (Add your code here to visualize the detections on the frame)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
