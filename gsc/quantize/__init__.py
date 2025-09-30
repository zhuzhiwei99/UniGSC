'''
Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
Date: 2025-07-06 18:28:46
LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
LastEditTime: 2025-10-01 00:04:31
FilePath: /VGSC/vgsc/quantize/__init__.py
Description: Quantization and de-quantization utilities for splats data.

Copyright (c) 2025 by Zhiwei Zhu (zhuzhiwei21@zju.edu.cn), All Rights Reserved. 
'''

from typing import Dict, List, Any, Union,  Tuple
import torch
from torch import Tensor
from gsc import get_dtype_for_bitdepth, TORCH_DTYPE_MAP
import numpy as np


def _to_tensor(val: Union[float, Tensor], ref: Tensor) -> Tensor:
    return val if isinstance(val, Tensor) else torch.tensor(val, dtype=torch.float32, device=ref.device)

def quantize_torch(data: Tensor, bit_depth: int, min_val: Union[float, Tensor], range_val: Union[float, Tensor]) -> Tensor:
    min_val, range_val = _to_tensor(min_val, data), _to_tensor(range_val, data)
    q_max = (1 << bit_depth) - 1
    scaled = (data - min_val) / range_val * q_max
    dtype = get_dtype_for_bitdepth(bit_depth)["torch"]
    return torch.clamp(torch.round(scaled), 0, q_max).to(dtype)

def dequantize_torch(q_data: Tensor, bit_depth: int, min_val: Union[float, Tensor], range_val: Union[float, Tensor]) -> Tensor:
    min_val, range_val = _to_tensor(min_val, q_data), _to_tensor(range_val, q_data)
    q_max = (1 << bit_depth) - 1
    return q_data.to(torch.float32) / q_max * range_val + min_val
