import custom_yolo_lib.model.e2e.anchor_based.losses.base_v3


class YOLOLossPerFeatureMapV3(
    custom_yolo_lib.model.e2e.anchor_based.losses.base_v3.BaseYOLOLossPerFeatureMapV3
):
    def define_radial_decay_mode(
        self,
    ) -> custom_yolo_lib.model.e2e.anchor_based.losses.base_v3.RadialDecayMode:
        return (
            custom_yolo_lib.model.e2e.anchor_based.losses.base_v3.RadialDecayMode.LINEAR
        )


class YOLOLossPerFeatureMapV3GaussianRadialDecay(
    custom_yolo_lib.model.e2e.anchor_based.losses.base_v3.BaseYOLOLossPerFeatureMapV3
):
    def define_radial_decay_mode(
        self,
    ) -> custom_yolo_lib.model.e2e.anchor_based.losses.base_v3.RadialDecayMode:
        return (
            custom_yolo_lib.model.e2e.anchor_based.losses.base_v3.RadialDecayMode.GAUSSIAN
        )
