import argparse
import pathlib

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Concatenate CSV files.")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        help="List of input CSV files to concatenate.",
        required=True,
    )
    parser.add_argument("--train", action="store_true", help="Train mode.")
    parser.add_argument(
        "--csv_file_template",
        type=str,
        default="session_data_epoch",
        help="Input CSV file name.",
    )
    parser.add_argument(
        "--ma_steps",
        type=int,
        default=25,
        help="Moving average steps.",
    )
    parser.add_argument(
        "--y_upper_limit",
        type=int,
        default=20,
        help="Y axis upper limit.",
    )
    return parser.parse_args()


def concatenate_csv(
    csv_dir: pathlib.Path,
    csv_file_template: str,
    store: bool = False,
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

    # Read and concatenate the CSV files
    dataframes = [pd.read_csv(file.as_posix()) for file in input_files]
    concatenated_df = pd.concat(dataframes, ignore_index=True)

    # Save the concatenated DataFrame to a new CSV file
    if store:
        output_file = csv_dir / f"{csv_file_template}_concatenated_{epoch+1}_epochs.csv"
        concatenated_df.to_csv(output_file, index=False)
    print(f"Concatenated {len(input_files)} files.")
    return concatenated_df, epoch + 1


def plot_csv(
    df: pd.DataFrame,
    output_path: pathlib.Path,
    ma_steps: int = 25,
    y_upper_limit: int = 20,
):
    """
    Plot data from a CSV file.

    Args:
        input_files (list): List of input CSV file paths.
        output_file (str): Output plot file path.
    """
    # Read the CSV files
    headers_to_plot = [
        "bbox_loss_avg_featmap",
        "objectness_loss_avg_featmap",
        "class_loss_avg_featmap",
        "total_loss_avg_featmap",
    ]
    x_axis_name = "step"

    # plot moving average
    for header in headers_to_plot:
        df[f"{header}_ma"] = df[header].rolling(window=ma_steps).mean()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(25, 10))
    
    # Plot each header's moving average
    for header in headers_to_plot:
        ax.plot(df[x_axis_name], df[f"{header}_ma"], label=header)
    
    # Set plot properties
    ax.set_title("Losses")
    ax.set_xlabel(x_axis_name)
    ax.set_ylabel("Loss")
    ax.set_ylim(0, y_upper_limit)
    ax.grid(True)
    ax.legend(loc="upper right")
    
    for epoch_group in df.groupby("epoch"):
        # vertical line
        # place labels in x axis
        ax.axvline(
            x=epoch_group[1][x_axis_name].iloc[0],
            color="black",
            linestyle="--",
        )
        ax.text(
            epoch_group[1][x_axis_name].iloc[0] + 10,
            y_upper_limit - (0.05 * y_upper_limit),
            f"Epoch {epoch_group[0]} start",
        )
        
    # Save the figure
    plt.savefig(output_path.as_posix())
    plt.close(fig)

if __name__ == "__main__":
    args = parse_args()
    if args.train:
        csv_template = f"training_{args.csv_file_template}"
    else:
        csv_template = f"validation_{args.csv_file_template}"
    df, epoch = concatenate_csv(
        pathlib.Path(args.experiment_dir),
        csv_template,
        store=False,
    )
    plot_csv(
        df,
        pathlib.Path(args.experiment_dir) / f"{csv_template}_{epoch}_epochs_plot.png",
        ma_steps=args.ma_steps,
        y_upper_limit=args.y_upper_limit,
    )
