import argparse

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Concatenate CSV files.")
    parser.add_argument(
        "input_files", nargs="+", help="List of input CSV files to concatenate."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="concatenated_output.csv",
        help="Output CSV file name.",
    )
    return parser.parse_args()


def concatenate_csv(input_files, output_file):
    """
    Concatenate multiple CSV files into a single CSV file.

    Args:
        input_files (list): List of input CSV file paths.
        output_file (str): Output CSV file path.
    """
    # Read and concatenate the CSV files
    dataframes = [pd.read_csv(file) for file in input_files]
    concatenated_df = pd.concat(dataframes, ignore_index=True)

    # Save the concatenated DataFrame to a new CSV file
    concatenated_df.to_csv(output_file, index=False)
    print(f"Concatenated {len(input_files)} files into {output_file}.")


if __name__ == "__main__":
    args = parse_args()
    concatenate_csv(args.input_files, args.output_file)
