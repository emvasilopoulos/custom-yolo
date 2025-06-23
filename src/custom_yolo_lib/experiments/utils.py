import pathlib


def make_experiment_dir(
    experiment_name: str,
    base_dir: pathlib.Path,
) -> pathlib.Path:
    """
    Create a directory for the experiment with the given name and base directory.
    The directory will be created if it does not exist.

    Args:
        experiment_name (str): Name of the experiment.
        base_dir (pathlib.Path): Base directory where the experiment directory will be created.

    Returns:
        pathlib.Path: Path to the experiment directory.
    """
    experiment_dir = base_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    # Create a new directory with a counter if the directory already exists
    counter = 1
    while (experiment_dir / f"{experiment_name}_{counter}").exists():
        counter += 1
    experiment_dir = experiment_dir / f"{experiment_name}_{counter}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir
