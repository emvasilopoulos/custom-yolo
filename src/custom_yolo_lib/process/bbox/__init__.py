import dataclasses

import torch


class InvalidBoundingBoxError(Exception):
    pass


@dataclasses.dataclass
class Bbox:
    """
    x: x coordinate of the top left corner of the bounding box
    y: y coordinate of the top left corner of the bounding box
    w: width of the bounding box
    h: height of the bounding box
    is_normalized: whether the bounding box is normalized or not
    """

    x: float
    y: float
    w: float
    h: float
    is_normalized: bool
    is_top_left: bool = True

    def _assert_positive(self) -> None:
        if any(k < 0 for k in (self.x, self.y, self.w, self.h)):
            raise InvalidBoundingBoxError(
                f"Bounding box values should be positive for bbox: {self}"
            )

    def _assert_is_valid_normalized(self) -> None:
        if any(k > 1 for k in (self.x, self.y, self.w, self.h)):
            raise InvalidBoundingBoxError(
                f"Normalized bounding box values should be between 0 and 1 for bbox: {self}"
            )

    def _assert_is_valid_non_normalized(self) -> None:
        if any(k > 1 for k in (self.w, self.h)):
            raise InvalidBoundingBoxError(
                f"Non-normalized bounding box values should be less than 1 for bbox: {self}"
            )

    def __post_init__(self) -> None:

        self._assert_positive()
        if self.is_normalized:
            self._assert_is_valid_normalized()
        else:
            self._assert_is_valid_non_normalized()

    def to_center(self) -> "Bbox":
        """
        Convert the bounding box from top-left format to center format.
        """
        if self.is_top_left:
            center_x = self.x + self.w / 2
            center_y = self.y + self.h / 2
            self.is_top_left = False
            self.x = center_x
            self.y = center_y
        return self

    def to_top_left(self) -> "Bbox":
        """
        Convert the bounding box from center format to top-left format.
        """
        if not self.is_top_left:
            top_left_x = self.x - self.w / 2
            top_left_y = self.y - self.h / 2
            self.is_top_left = True
            self.x = top_left_x
            self.y = top_left_y
        return self


def bboxes_to_tensor(bboxes: list[Bbox]) -> torch.Tensor:
    """
    Convert a list of Bbox objects to a tensor.
    """
    return torch.tensor(
        [[bbox.x, bbox.y, bbox.w, bbox.h] for bbox in bboxes], dtype=torch.float32
    )
