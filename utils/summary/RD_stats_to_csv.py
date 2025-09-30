'''
Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
Date: 2025-07-04 00:41:56
LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
LastEditTime: 2025-08-13 17:46:04
FilePath: /VGSC/utils/summary/RD_stats_to_csv.py
Description: Summarize results to csv file

Copyright (c) 2025 by Zhiwei Zhu (zhuzhiwei21@zju.edu.cn), All Rights Reserved. 
'''
import json
import csv
import os
import tyro

def main(results_dir: str):
    if not os.path.exists(results_dir):
        print(f"Experiment directory {results_dir} does not exist.")
        return
    all_rps = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    all_rps.sort()  # Sort the result paths to ensure consistent order
    if not all_rps:
        print(f"No directories found in {results_dir}")
        return
    metrics_paths = [
        os.path.join(results_dir, f"{rp}/stats/gsc_metrics.json") for rp in all_rps
    ]
    summary_paths = [
        os.path.join(results_dir, f"{rp}/rd_summary_rendered.json") for rp in all_rps
    ]

    # Load all data for each rate point
    all_data = []
    for path in metrics_paths:
        with open(path, "r") as f:
            all_data.append(json.load(f))

    # Get all test_view keys (assuming all files have the same order)
    all_v_ids = [k for k in all_data[0].keys() if k.startswith("testv")]

    # Extract bitrate for each rp (in bps)
    bitrates = []
    for path in summary_paths:
        with open(path, "r") as f:
            data = json.load(f)
            bitrate_kbps = data["bitrate_Kbps"]
            bitrates.append(bitrate_kbps)

    # Write combined CSV
    header = ["view", "rate_point", "RGB_PSNR", "YUV_PSNR", "YUV_SSIM", "YUV_IVSSIM", "LPIPS", "bitrate(kbps)"]
    csv_path = os.path.join(results_dir, "gsc_metrics_all.csv")
    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for v_id in all_v_ids:
            for rp_idx, rp in enumerate(all_rps):
                metrics = all_data[rp_idx][v_id]
                row = [
                    v_id, rp,
                    metrics["RGB_PSNR"], metrics["YUV_PSNR"], metrics["YUV_SSIM"], metrics["YUV_IVSSIM"], metrics["LPIPS"],
                    bitrates[rp_idx]
                ]
                writer.writerow(row)
            # Add empty rows for the missing rate points
            if rp_idx < 4:
                for _ in range(4 - rp_idx):
                    writer.writerow([])

    print(f"gsc_metrics_all.csv has been generated at {csv_path} and can be opened with Excel.")

if __name__ == "__main__":
    tyro.cli(main) 