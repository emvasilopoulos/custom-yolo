# Custom YOLO
Building a YOLO detector from scratch. Current implementation can be trained on COCO basic data augmentation techniques, not including Cutmix and Mixup. I don't intend to implement colour-based augmentation techniques.
The one and only implemented model is anchor-based. Inspired by YOLOv3, borrowing techniques from YOLOv4 [Grid Sensitive](https://paperswithcode.com/method/grid-sensitive) and reverse engineering YOLOv5.

## Note
In the code "v3" does not imply YOLOv3, but rather an internal versioning "system".

# Research
Experimenting with radial decay initially in objectness scores and later will see how it fits in other places. I believe it will improve performance, because [0,0,1,0,0] seems to make less sense than [0.25, 0.5, 1, 0.5, 0.25].
(Don't worry I will write more on this.)

# Acknowledgements
Joseph Redmon, Ali Farhadi (YOLOv2 & YOLOv3)
Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao (YOLOv4)
Glenn Jocher & Ultralytics team (YOLOv5) On one hand the code is hard to understand in parts with PyTorch operations (feels magic). On the other, the repos are an advanced course to deep learning with PyTorch. 

# TODOs
1. when converting from original COCO annotations to this repo's format create a standardized naming
  - Issue in annotations name and the expected name from class dataset.coco.tasks.base.BaseCOCODatasetGrouped
2. Create new dataset COCO Units, i.e. crop each individual object into a separate image with the corresponding associated label
