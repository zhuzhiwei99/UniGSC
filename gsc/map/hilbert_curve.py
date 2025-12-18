'''
Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
Date: 2025-10-16 19:29:34
LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
LastEditTime: 2025-10-17 13:59:59
FilePath: /VGSC/gsc/map/hilbert_curve.py
Description: 

Copyright (c) 2025 by Zhiwei Zhu, All Rights Reserved. 
'''
import torch
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve

def hilbert_codes(pos: torch.Tensor, pos_bits: int=21, dim=3) -> torch.Tensor:
    hilbert_curve = HilbertCurve(p=pos_bits, n=dim, n_procs=-1)
    pos_np = pos.cpu().long().numpy()
    codes = hilbert_curve.distances_from_points(pos_np)
    return torch.tensor(codes, dtype=torch.long, device=pos.device)


def hilbert_sort_2d(H, W, device=None):
    p = max(H, W).bit_length()  
    hilbert_curve = HilbertCurve(p, 2, n_procs=-1)
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    points = np.stack([xs.ravel(), ys.ravel()], axis=1)  # shape=(H*W, 2)
    codes = hilbert_curve.distances_from_points(points)
    return torch.argsort(torch.tensor(codes, device=device))


    