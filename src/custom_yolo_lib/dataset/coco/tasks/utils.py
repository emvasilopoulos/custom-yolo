def get_task_file(task_name: str, split: str, year: str) -> str:
    """
    Get the task file path based on the task name, split, and year.

    Args:
        task_name (str): The name of the task.
        split (str): The data split (e.g., 'train', 'val').
        year (str): The year of the dataset.

    Returns:
        str: The path to the task file.
    """
    return f"{task_name}_{split}{year}.json"
