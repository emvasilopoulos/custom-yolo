import torch


class TensorDtypeConverter(torch.nn.Module):
    """
    Don't use the ready torchvision.transforms.ConvertImageDtype
    because it normalizes the image to [0, 1] with a unique way
    and we want to know the exact way it is normalized before
    feeding it to the model.
    Otherwise changes in the normalization will affect the model's
    performance and it will be hard to understand why.
    """

    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.dtype)
