import dataclasses


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

    def _assert_positive(self):
        if any(k < 0 for k in (self.x, self.y, self.w, self.h)):
            raise InvalidBoundingBoxError(
                f"Bounding box values should be positive for bbox: {self}"
            )

    def _assert_is_valid_normalized(self):
        if any(k > 1 for k in (self.x, self.y, self.w, self.h)):
            raise InvalidBoundingBoxError(
                f"Normalized bounding box values should be between 0 and 1 for bbox: {self}"
            )

    def _assert_is_valid_non_normalized(self):
        if any(k > 1 for k in (self.w, self.h)):
            raise InvalidBoundingBoxError(
                f"Non-normalized bounding box values should be less than 1 for bbox: {self}"
            )

    def __post_init__(self):

        self._assert_positive()
        if self.is_normalized:
            self._assert_is_valid_normalized()
        else:
            self._assert_is_valid_non_normalized()
