import pathlib
from typing import Any, Dict
import yaml


def read_yaml(file_path: pathlib.Path) -> Dict[str, Any]:
    with open(file_path, "r") as file:
        base_configuration = yaml.safe_load(file, Loader=yaml.FullLoader)
    return base_configuration
