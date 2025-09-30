'''
Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
Date: 2025-07-04 11:03:19
LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
LastEditTime: 2025-09-29 23:31:06
FilePath: /VGSC/utils/summary/summarize_stats.py
Description: 

Copyright (c) 2025 by Zhiwei Zhu (zhuzhiwei21@zju.edu.cn), All Rights Reserved. 
'''
import json
import os
from collections import defaultdict
import tyro
import pandas as pd

def main(results_dir: str):
    rps = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    rps.sort()  # Sort the result paths to ensure consistent order

    summary_GT = defaultdict(dict)
    summary_rendered = defaultdict(dict)
    summary_info = defaultdict(dict)
    for rp in rps:
        rp_dir = os.path.join(results_dir, rp)
 
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
       
    with open(f"{results_dir}/rp_summary.json", "w") as fp:
        json.dump(summary_GT, fp, indent=2)
    with open(f"{results_dir}/rp_summary_rendered.json", "w") as fp:
        json.dump(summary_rendered, fp, indent=2)
    with open(f"{results_dir}/rp_summary_info.json", "w") as fp:
        json.dump(summary_info, fp, indent=2)


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

if __name__ == "__main__":
    tyro.cli(main)
