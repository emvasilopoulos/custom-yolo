import enum


class AugmentationType(enum.Enum):
    """
    Enum for different augmentation types.
    """

    FLIP_X = enum.auto()
    FLIP_Y = enum.auto()
    ROTATE = enum.auto()
    SLIGHT_COLOR_JITTER = enum.auto()
    SLIGHT_RESIZE = enum.auto()
    MASK_BACKGROUND = enum.auto()
    MASK_FOREGROUND = enum.auto()
    CUTOUT = enum.auto()
    CUTMIX = enum.auto()
    # MIXUP = enum.auto() # I don't like it
