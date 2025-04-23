import argparse

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot CSV file.")
    parser.add_argument("--input_file", type=str, help="CSV file to plot.")
    parser.add_argument(
        "--output_file",
        type=str,
        default="plot_output.png",
        help="Output plot file name.",
    )
    return parser.parse_args()


def plot_csv(input_file, output_file):
    """
    Plot data from a CSV file.

    Args:
        input_files (list): List of input CSV file paths.
        output_file (str): Output plot file path.
    """
    # Read the CSV files
    df = pd.read_csv(input_file)
    headers_to_plot = [
        "bbox_loss_avg_featmap",
        "objectness_loss_avg_featmap",
        "class_loss_avg_featmap",
        "total_loss_avg_featmap",
    ]
    x_axis_name = "step"

    # plot moving average
    for header in headers_to_plot:
        df[f"{header}_ma"] = df[header].rolling(window=100).mean()
    df.plot(
        x=x_axis_name,
        y=[f"{x}_ma" for x in headers_to_plot],
        title="Losses",
        xlabel=x_axis_name,
        ylabel="Loss",
        ylim=(0, 5),
        grid=True,
        figsize=(25, 10),
    )
    plt.legend(loc="upper right")
    plt.savefig(output_file)


if __name__ == "__main__":
    args = parse_args()
    plot_csv(args.input_file, args.output_file)
