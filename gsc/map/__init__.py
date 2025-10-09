'''
Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
Date: 2025-08-06 16:24:52
LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
LastEditTime: 2025-10-01 00:11:59
FilePath: /VGSC/vgsc/map/__init__.py
Description: 

Copyright (c) 2025 by Zhiwei Zhu, All Rights Reserved. 
'''

import torch
import numpy as np



def morton_code(pos: torch.Tensor, pos_bits: int=21) -> torch.Tensor:
    def splitBy3(a, max_val):
        x = a.clamp(0, max_val)
        x = (x | x << 32) & 0x1F00000000FFFF
        x = (x | x << 16) & 0x1F0000FF0000FF
        x = (x | x << 8) & 0x100F00F00F00F00F
        x = (x | x << 4) & 0x10C30C30C30C30C3
        x = (x | x << 2) & 0x1249249249249249
        return x

    pos = pos.to(torch.long) 
    x, y, z = pos.unbind(-1)
    max_coord_val = (1 << pos_bits) - 1
    codes = torch.zeros(len(pos), dtype=torch.long, device=pos.device)
    codes |= splitBy3(x, max_coord_val) | \
              splitBy3(y, max_coord_val) << 1 | \
              splitBy3(z, max_coord_val) << 2
    return codes

