import argparse
import pathlib

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Concatenate CSV files.")
    parser.add_argument(
        "--csv_dir", type=str, help="List of input CSV files to concatenate."
    )
    parser.add_argument(
        "--csv_file_template",
        type=str,
        required=True,
        help="Input CSV file name.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="concatenated_output.csv",
        help="Output CSV file name.",
    )
    return parser.parse_args()


def concatenate_csv(
    csv_dir: pathlib.Path, csv_file_template: str, output_file: pathlib.Path
):
    """
    Concatenate multiple CSV files into a single CSV file.

    Args:
        input_files (list): List of input CSV file paths.
        output_file (str): Output CSV file path.
    """
    input_files = []
    epoch = 0
    while 1:
        csv_path = csv_dir / f"{csv_file_template}_{epoch}.csv"
        if not csv_path.exists():
            break
        input_files.append(csv_path)
        epoch += 1

    print(input_files)
    # Read and concatenate the CSV files
    dataframes = [pd.read_csv(file.as_posix()) for file in input_files]
    concatenated_df = pd.concat(dataframes, ignore_index=True)

    # Save the concatenated DataFrame to a new CSV file
    concatenated_df.to_csv(output_file, index=False)
    print(f"Concatenated {len(input_files)} files into {output_file}.")


if __name__ == "__main__":
    args = parse_args()
    concatenate_csv(
        pathlib.Path(args.csv_dir),
        args.csv_file_template,
        pathlib.Path(args.output_file),
    )
