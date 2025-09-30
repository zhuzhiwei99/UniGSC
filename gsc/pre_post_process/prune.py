'''
Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
Date: 2025-07-06 18:28:46
LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
LastEditTime: 2025-09-25 18:34:57
FilePath: /VGSC/vgsc/pre_post_process/prune.py
Description: Common utilities and configurations for the VGSC library.
((
Copyright (c) 2025 by Zhiwei Zhu (zhuzhiwei21@zju.edu.cn), All Rights Reserved. 
'''

import os
import math
import json
import shutil
import torch
from torch import Tensor
from typing import List, Dict, Tuple, Literal, Optional
from gsc.utils.gs_io import save_ply
from gsc import get_logger
logger = get_logger("Pruner")


def prune_splats(
        splats: Dict[str, Tensor],
        keep_indices: Tensor):
    """Prune splats based on keep indices."""
    return {k: v[keep_indices] for k, v in splats.items()}

def prune_splats_threshold(
        splats: Dict[str, Tensor],
        prune_thres_opacities: float = -7.0,
        prune_thres_scales: Optional[float] = None) -> Dict[str, Tensor]:
    """Prune splats based on threshold."""
    opacities = splats["opacities"]
    scales = splats["scales"]
    num_points = opacities.shape[0]

    if prune_thres_opacities is None:
        prune_thres_opacities = -7.0
        logger.warning(f"prune_thres_opacities is None, set to default {prune_thres_opacities}")
        
    idx_low_opacity = torch.where(opacities < prune_thres_opacities)[0]
    logger.info(f"{len(idx_low_opacity)} / {num_points} splats with low opacity < {prune_thres_opacities}")
    
    if prune_thres_scales is not None:
        # scales: (N, 3), when all dimensions are below threshold, we prune it
        idx_low_scale = torch.where((scales < prune_thres_scales).all(dim=1))[0]
        logger.info(f"{len(idx_low_scale)} / {num_points} splats with low scale < {prune_thres_scales} in all dimensions")
        
        idx_prune = torch.unique(torch.cat([idx_low_opacity, idx_low_scale]))
    else:
        idx_prune = idx_low_opacity
    logger.info(f"{len(idx_prune)} / {num_points} splats will be pruned based on thresholds")

    keep_mask = torch.ones(num_points, dtype=torch.bool, device=scales.device)
    keep_mask[idx_prune] = False
    pruned_splats = prune_splats(splats, keep_mask)
    return pruned_splats
      
    
def prune_splats_list(
        splats_list: List[Dict[str, Tensor]],
        prune_type: str = "threshold",
        prune_thres_opacities: float = -7.0,
        prune_thres_scales:  Optional[float] = None) -> List[Dict[str, Tensor]]:
    """Prune splats list based on ratio or threshold."""
    pruned_splats_list = []
    for i, splats in enumerate(splats_list):
        logger.info(f"Pruning frame {i}...")
        if prune_type == "threshold":
            pruned_splats = prune_splats_threshold(
                splats, 
                prune_thres_opacities=prune_thres_opacities, 
                prune_thres_scales=prune_thres_scales)
        else:
            raise ValueError(f"Unknown prune type: {prune_type}")
        pruned_splats_list.append(pruned_splats)
    return pruned_splats_list
    
            
