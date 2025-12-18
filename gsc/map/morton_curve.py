'''
Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
Date: 2025-10-16 19:29:08
LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
LastEditTime: 2025-10-16 19:36:45
FilePath: /VGSC/gsc/map/morton_curve.py
Description: 

Copyright (c) 2025 by Zhiwei Zhu, All Rights Reserved. 
'''
import torch

def morton_codes_3d(pos: torch.Tensor, pos_bits: int=21) -> torch.Tensor:
    def splitBy3(a, max_val):
        x = a.clamp(0, max_val)
        x = (x | x << 32) & 0x1F00000000FFFF
        x = (x | x << 16) & 0x1F0000FF0000FF
        x = (x | x << 8) & 0x100F00F00F00F00F
        x = (x | x << 4) & 0x10C30C30C30C30C3
        x = (x | x << 2) & 0x1249249249249249
        return x
    if pos.shape[-1] != 3:
        raise ValueError("Input position tensor must have shape (..., 3)")
    pos = pos.to(torch.long) 
    x, y, z = pos.unbind(-1)
    max_coord_val = (1 << pos_bits) - 1
    codes = torch.zeros(len(pos), dtype=torch.long, device=pos.device)
    codes |= splitBy3(x, max_coord_val) | \
              splitBy3(y, max_coord_val) << 1 | \
              splitBy3(z, max_coord_val) << 2
    return codes

def morton_codes_2d(pos: torch.Tensor, pos_bits: int=21) -> torch.Tensor:
    def splitBy1(a, max_val):
        x = a.clamp(0, max_val)
        x = (x | x << 16) & 0x0000FFFF0000FFFF
        x = (x | x << 8) & 0x00FF00FF00FF00FF
        x = (x | x << 4) & 0x0F0F0F0F0F0F0F0F
        x = (x | x << 2) & 0x3333333333333333
        x = (x | x << 1) & 0x5555555555555555
        return x
    if pos.shape[-1] != 2:
        raise ValueError("Input position tensor must have shape (..., 2)")
    pos = pos.to(torch.long) 
    x, y = pos.unbind(-1)
    max_coord_val = (1 << pos_bits) - 1
    codes = torch.zeros(len(pos), dtype=torch.long, device=pos.device)
    codes |= splitBy1(x, max_coord_val) | \
              splitBy1(y, max_coord_val) << 1
    return codes

def morton_sort_2d(H, W, device=None):
    ys, xs = torch.meshgrid(torch.arange(H, device=device),
                            torch.arange(W, device=device),
                            indexing='ij')

    def part1by1(n):
        n &= 0x0000ffff
        n = (n | (n << 8)) & 0x00FF00FF
        n = (n | (n << 4)) & 0x0F0F0F0F
        n = (n | (n << 2)) & 0x33333333
        n = (n | (n << 1)) & 0x55555555
        return n

    def interleave(x, y):
        return (part1by1(y) << 1) | part1by1(x)

    z = interleave(xs.flatten(), ys.flatten())
    return torch.argsort(z)