import json
import os
from collections import defaultdict
import tyro
import pandas as pd
from typing import Dict


def compare_render_stats(stats1: Dict, stats2: Dict, name1: str = "Original", name2: str = "Modified") -> None:
    """
    Compare rendering statistics between two sets of results.
    
    Args:
        stats1 (Dict): First set of rendering statistics
        stats2 (Dict): Second set of rendering statistics
        name1 (str): Name/label for the first set of statistics (default: "Original")
        name2 (str): Name/label for the second set of statistics (default: "Modified")
    
    The function prints a comparison of PSNR, SSIM, and LPIPS metrics for each frame
    and the average across all frames, showing the difference between the two sets.
    """
    print(f"\n=== Rendering Comparison: {name1} vs {name2} ===")
    # frame_ids are common in both stats1 and stats2
    frame_ids1 = set(stats1.keys())
    frame_ids2 = set(stats2.keys())
    common_frame_ids = frame_ids1.intersection(frame_ids2)
    if len(common_frame_ids) < len(frame_ids1) or len(common_frame_ids) < len(frame_ids2):
        print("Warning: The two stats have different frame IDs. Only common frames will be compared.")
        common_frame_ids.remove("average")  # Because frame_ids are not matched, so 'average' cannot be compared.
        common_frame_ids = sorted(list(common_frame_ids))
    
    for frame_id in common_frame_ids:
        if frame_id == "average":
            print("\n=== Average Metrics Comparison ===")
        else:
            print(f"\n=== Frame {frame_id} Comparison ===")
        
        metrics1 = stats1[frame_id]
        metrics2 = stats2[frame_id]
        
        for metric in ["psnr", "ssim", "lpips"]:
            value1 = metrics1[metric]
            value2 = metrics2[metric]
            diff = value2 - value1
            diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
            
            print(f"{metric.upper():<8}: {value1:.4f} -> {value2:.4f} ({diff_str})")


def summary(result_dir: str,  frame_num: int):
    import pandas as pd
    from gsc.config import ATTRIBUTE_MAP
    from gsc.utils.plot import plot_pie
    def format_size(size_bytes):
        """Convert byte size to readable format (KB, MB, GB, etc.)"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.2f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/(1024**2):.2f} MB"
        else:
            return f"{size_bytes/(1024**3):.2f} GB"
        
    ### rate summary
    compress_dir = os.path.join(result_dir, "compression")
    
    # Check if directory exists
    if not os.path.exists(compress_dir):
        print(f"Error: Directory '{compress_dir}' does not exist")
        return
    
    # Store file and size information
    file_sizes = {}
    total_size = 0
    
    for item in os.listdir(compress_dir):
        item_path = os.path.join(compress_dir, item)
        if os.path.isfile(item_path):
            size = os.path.getsize(item_path)
            file_sizes[item] = size
            total_size += size

    file_sizes_grouped = {}
    for key, value in ATTRIBUTE_MAP.items():
        size = sum(size for fname, size in file_sizes.items() if fname.startswith(value))
        file_sizes_grouped[key] = size

    # save to json
    with open(os.path.join(result_dir, "stats", "storage.json"), "w") as fp:
        json.dump(file_sizes_grouped, fp, indent=4)
            # plot pie chart
    plot_pie(file_sizes_grouped, f"{result_dir}/stats/storage_pie_chart.png", title="Storage")

        
    # Get bitrate
    Byte_to_Kbps = lambda filesize, n_frame: filesize / 1024 / n_frame * 8 * 30
    bitrate_Kbps = Byte_to_Kbps(total_size, frame_num)
    
    Byte_to_Mbps = lambda filesize, n_frame: filesize / (1024 ** 2) / n_frame * 8 * 30
    bitrate_Mbps = Byte_to_Mbps(total_size, frame_num)


    # Calculate percentage
    percentages = {name: (size / total_size) * 100 for name, size in file_sizes.items()}

    # Storage breakdown table
    table_data = []
    for name, size in sorted(file_sizes.items(), key=lambda x: x[1], reverse=True):
        size_formatted = format_size(size)
        percentage = percentages[name]
        table_data.append([name, size_formatted, f"{percentage:.2f}%"])
    
    # Create pandas DataFrame for table
    df = pd.DataFrame(table_data, columns=["Filename", "Size", "Percentage"])
    csv_path = os.path.join(result_dir, "stats", "storage_detail.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV file saved to: {csv_path}")

    ### distortion summary
    # compressed vs GT
    with open(os.path.join(result_dir, "stats", "compress.json"), "r") as fp:
        quality_metrics = json.load(fp)
        avg_quality_metrics = quality_metrics["average"]
    
    # compressed vs val (before compression vs after compression)
    with open(os.path.join(result_dir, "stats", "gsc_metrics.json"), "r") as fp:
        gsc_metrics = json.load(fp)
        avg_gsc_metrics = gsc_metrics["average"]
    
    # save summary into a json file
    rd_summary_rendered = {key: value for key, value in avg_gsc_metrics.items()}
    rd_summary_GT = {key: value for key, value in avg_quality_metrics.items() if key != "ellipse_time"}
    rd_summary_GT["bytes"] = total_size
    rd_summary_GT["bitrate"] = bitrate_Mbps
    rd_summary_GT["bitrate_Kbps"] = bitrate_Kbps
    
    rd_summary_rendered["bytes"] = total_size
    rd_summary_rendered["bitrate"] = bitrate_Mbps
    rd_summary_rendered["bitrate_Kbps"] = bitrate_Kbps
    
    
    with open(os.path.join(result_dir, "rd_summary_GT.json"), "w") as fp:
        json.dump(rd_summary_GT, fp, indent=4)
    with open(os.path.join(result_dir, "rd_summary_rendered.json"), "w") as fp:
        json.dump(rd_summary_rendered, fp, indent=4)


def main(results_dir: str, frame_num: int = 1):
    rps = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    rps.sort()  # Sort the result paths to ensure consistent order

    summary_GT = defaultdict(dict)
    summary_rendered = defaultdict(dict)
    summary_info = defaultdict(dict)
    summary_storage = defaultdict(dict)
    for rp in rps:
        rp_dir = os.path.join(results_dir, rp)
        summary(rp_dir, frame_num)  
 
        try:
            with open(os.path.join(rp_dir, f"rd_summary_GT.json"), "r") as f:
                stats = json.load(f)
                for k, v in stats.items():
                    summary_GT[rp][k] = v
        except:
            print(f"Could not find rd_summary_GT.json in {rp_dir}, skipping.")
            continue
        try:
            with open(os.path.join(rp_dir, f"rd_summary_rendered.json"), "r") as f:
                stats = json.load(f)
                for k, v in stats.items():
                    summary_rendered[rp][k] = v
        except:
            print(f"Could not find rd_summary_rendered.json in {rp_dir}, skipping.")
            continue
        
        try:
            with open(os.path.join(rp_dir, f"stats/info.json"), "r") as f:
                stats = json.load(f)
                for k, v in stats.items():
                    summary_info[rp][k] = v
        except:
            print(f"Could not find info.json in {rp_dir}/stats, skipping.")
            continue
        
        try:
            with open(os.path.join(rp_dir, f"stats/storage.json"), "r") as f:
                stats = json.load(f)
                for k, v in stats.items():
                    summary_storage[rp][k] = v
        except:
            print(f"Could not find storage.json in {rp_dir}/stats, skipping.")
            continue
       
    with open(f"{results_dir}/rp_summary.json", "w") as fp:
        json.dump(summary_GT, fp, indent=2)
    with open(f"{results_dir}/rp_summary_rendered.json", "w") as fp:
        json.dump(summary_rendered, fp, indent=2)


    print(json.dumps(summary_GT, indent=2, ensure_ascii=False))
    print(json.dumps(summary_rendered, indent=2, ensure_ascii=False))
    
    print(f"[Summary VS GT] results are saved to: {results_dir}/rp_summary.json")
    print(f"[Summary VS Rendered] results are saved to: {results_dir}/rp_summary_rendered.json")
    # json to csv
    df_GT = pd.DataFrame.from_dict(summary_GT, orient='index')
    df_GT.to_csv(f"{results_dir}/rp_summary.csv")
    print(f"[Summary VS GT] results are saved to: {results_dir}/rp_summary.csv")
    df_rendered = pd.DataFrame.from_dict(summary_rendered, orient='index')
    df_rendered.to_csv(f"{results_dir}/rp_summary_rendered.csv")  
    print(f"[Summary VS Rendered] results are saved to: {results_dir}/rp_summary_rendered.csv")
    df_info = pd.DataFrame.from_dict(summary_info, orient='index')
    df_info.to_csv(f"{results_dir}/rp_summary_info.csv")
    print(f"[Summary Info] results are saved to: {results_dir}/rp_summary_info.csv")
    df_storage = pd.DataFrame.from_dict(summary_storage, orient='index')
    df_storage.to_csv(f"{results_dir}/rp_summary_storage.csv")
    print(f"[Summary Storage] results are saved to: {results_dir}/rp_summary_storage.csv")

if __name__ == "__main__":
    tyro.cli(main)
