'''
Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
Date: 2025-10-02 16:57:13
LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
LastEditTime: 2025-10-03 01:02:12
FilePath: /UniGSC/gsc/utils/plot.py
Description: 

Copyright (c) 2025 by Zhiwei Zhu, All Rights Reserved. 
'''
import matplotlib.pyplot as plt
from collections import defaultdict

def group_data_auto_prefix(data):
    grouped = defaultdict(float)

    for name, value in data:
        prefix = name.split("_")[0] if "_" in name else name.split(".")[0]
        grouped[prefix] += value

    return sorted(grouped.items(), key=lambda x: -x[1])  

def plot_pie(data: dict, output_path: str, title: str, group_by_prefix=False):
    if group_by_prefix:
        data = group_data_auto_prefix(data)

    labels, sizes = zip(*data.items())

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(title)
    plt.savefig(output_path)
    plt.close()
   