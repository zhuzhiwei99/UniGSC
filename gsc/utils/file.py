'''
Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
Date: 2025-07-10 10:56:21
LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
LastEditTime: 2025-09-29 17:34:59
FilePath: /VGSC/vgsc/utils/file.py
Description: 

Copyright (c) 2025 by Zhiwei Zhu (zhuzhiwei21@zju.edu.cn), All Rights Reserved. 
'''
import os
import json
import yaml
import shutil
import numpy as np
import torch
from typing import Dict, Any, Union, List
import fnmatch
from pathlib import Path

def force_make_dirs(path):
    """
    Forcefully create directories, removing existing ones if they exist.
    """
    if os.path.exists(path):
        print(f"Directory {path} already exists. It will be removed and recreated.")
        shutil.rmtree(path)
    os.makedirs(path)
    print(f"Directory {path} created successfully.")
    
def safe_make_dirs(path):
    """
    Safely create directories, ensuring they do not exist before creating.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory {path} not exists. Created successfully.")
    else:
        print(f"Directory {path} already exists. No action taken.")
      
def smart_save_meta(meta: Dict[str, Any], meta_path: str | Path):
    """Save metadata to a file.

    Args:
        meta (Dict[str, Any]): Metadata dictionary to save.
        meta_path (str | Path): Path to the metadata file.
    """
    meta_path = Path(meta_path)

    if meta_path.exists():
        print(f"Warning: Metadata file {meta_path} already exists. It will be overwritten.")

    suffix = meta_path.suffix.lower()  

    if suffix == ".json":
        with meta_path.open('w') as f:
            json.dump(meta, f, indent=4)
    elif suffix in {".yaml", ".yml"}:
        with meta_path.open('w') as f:
            yaml.dump(meta, f, default_flow_style=False)
    elif suffix == ".npz":
        np.savez(meta_path, **meta)
    elif suffix in {".pth", ".pt"}:
        torch.save(meta, meta_path)
    else:
        raise ValueError(
            f"Unsupported metadata file format: {meta_path}. "
            "Supported formats are .json, .yaml/.yml, .npz, .pth, and .pt."
        )
    

def smart_load_meta(meta_path: Union[str, Path]) -> Dict[str, Any]:
    """Load metadata from a file.

    Args:
        meta_path (str | Path): Path to the metadata file.

    Returns:
        Dict[str, Any]: Loaded metadata dictionary.
    """
    meta_path = Path(meta_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file {meta_path} does not exist.")

    suffix = meta_path.suffix.lower()  

    if suffix == ".json":
        with meta_path.open('r') as f:
            return json.load(f)
    elif suffix in {".yaml", ".yml"}:
        with meta_path.open('r') as f:
            return yaml.safe_load(f)
    elif suffix == ".npz":
        data = np.load(meta_path)
        return {k: v for k, v in data.items()}
    elif suffix in {".pth", ".pt"}:
        return torch.load(meta_path)
    else:
        raise ValueError(
            f"Unsupported metadata file format: {meta_path}. "
            "Supported formats are .json, .yaml/.yml, .npz, .pth, and .pt."
        )
        
def search_files(root_dir: str, patterns: List[str]) -> List[str]:
    """Search for files matching given patterns in a directory and its subdirectories.
    Args:
        root_dir (str): Root directory to start the search.
        patterns (List[str]): List of filename patterns to match (e.g., ['*.txt', '*.jpg']).
    Returns:
        List[str]: List of paths to files that match the given patterns.
    """
    matches = []
    for dirpath, _, filenames in os.walk(root_dir):
        for pattern in patterns:
            for filename in fnmatch.filter(filenames, pattern):
                matches.append(os.path.join(dirpath, filename))
    return matches