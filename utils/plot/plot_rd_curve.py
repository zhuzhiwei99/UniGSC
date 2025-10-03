import os
import fnmatch
import argparse
import json
from typing import List
import matplotlib.pyplot as plt

DEFAULT_TARGRTS = [
    "Cricket_player",
    "LEGO_Bugatti",
    "LEGO_Ferrari",
    "Plant",
    "Solo_Tango_Female",
    "Solo_Tango_Male",
    "Tango_duo",
    "Tennis_player",
    'bartender',
    'cinema',
    'breakfast'
    ]

def search_files(root_dir: str, patterns: List[str]) -> List[str]:
    """Search for files matching given patterns in a directory and its subdirectories."""
    matches = []
    for dirpath, _, filenames in os.walk(root_dir):
        for pattern in patterns:
            for filename in fnmatch.filter(filenames, pattern):
                matches.append(os.path.join(dirpath, filename))
    return matches

def Mbps_to_MB(mbps: float) -> float:
    """Convert Mbps to MB/s."""
    return mbps / 8.0 /30 
   
def plot_rd_curve(rd_json_list: List[str], labels: List[str], output_dir: str, target: str, to_MB: bool = True):
    """Plot RD curves from a list of JSON files.
    
    Args:
        rd_json_list (List[str]): List of JSON paths containing RD metrics.
        labels (List[str]): Labels for each RD curve (usually filenames).
        output_dir (str): Directory to save the figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics = ["RGB_PSNR", "YUV_PSNR", "YUV_SSIM", "YUV_IVSSIM", "LPIPS"]

    for metric in metrics:
        plt.figure(figsize=(8, 4))
        for json_path, label in zip(rd_json_list, labels):
            with open(json_path, "r") as f:
                data = json.load(f)
            # 提取 bitrate 和 metric
            bitrates = []
            values = []
            for key, val in sorted(data.items()):
                if to_MB:
                    bitrates.append(Mbps_to_MB(val["bitrate"]))
                else:
                    bitrates.append(val["bitrate"])
                values.append(val[metric])
            plt.plot(bitrates, values, marker="o", label=label)

        if to_MB:
            plt.xlabel("MB")
        else:
            plt.xlabel("Mbps")
        plt.ylabel(metric)
        plt.title(target)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        save_path = os.path.join(output_dir, f"{metric}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Search JSON files and plot RD curves."
    )
    parser.add_argument("root_dir", help="root directory to search for JSON files")
    parser.add_argument(
        "target_files",
        nargs="+",
        help="target file patterns to search for (e.g., 'bartender' for files containing 'bartender')",
        default=DEFAULT_TARGRTS,
    )
    parser.add_argument("--output_root_dir", help="output root directory for plots", default=None)
    parser.add_argument("--output_dir_name", help="name of the output directory", default="rd_curves")
    parser.add_argument("--to_MB", action="store_true", help="Convert bitrate Mbps to MB")
    
    args = parser.parse_args()

    for target in args.target_files:
        if args.output_root_dir:
            args.output_root_dir = os.path.join(args.output_root_dir, args.output_dir_name, target)
        else:
            args.output_root_dir = os.path.join('results/rd_curve', args.output_dir_name, target)

        matched_files = search_files(args.root_dir, [f"*{target}*"])

        if not matched_files:
            print("Cannot find any matching files.")
            continue

        print(f"Found {len(matched_files)} files matching {target} in {args.root_dir}")
        print("\n".join(matched_files))
        labels = [os.path.basename(os.path.dirname(p)) for p in matched_files]

        plot_rd_curve(matched_files, labels, args.output_root_dir, target)


if __name__ == "__main__":
    main()
