'''
Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
Date: 2025-10-15 23:54:13
LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
LastEditTime: 2025-10-17 18:43:34
FilePath: /VGSC/gsc/map/hilbert_curve_fast.py
Description: 

Copyright (c) 2025 by Zhiwei Zhu, All Rights Reserved. 
'''
import torch

def hilbert_codes_2d_fast(pos: torch.Tensor, pos_bits: int) -> torch.Tensor:
    """
    Calculate 2D Hilbert codes for a set of 2D coordinates using PyTorch.
    This implementation is optimized for GPU execution and handles serial dependencies
    correctly by unrolling loops where necessary.
    Args:
        pos (torch.Tensor): Input tensor of shape (N, 2) containing 2D coordinates.
        pos_bits (int): Number of bits to represent each coordinate (default: 16).
    Returns:
        torch.Tensor: Tensor of shape (N,) containing the Hilbert codes.
    """
    if pos.dim() != 2 or pos.shape[1] != 2:
        raise ValueError(f"Input tensor 'pos' must have shape (N, 2), but got {pos.shape}")

    coords = pos.long()
    x, y = coords[:, 0], coords[:, 1]

    m = 1 << (pos_bits - 1)
    
    # Part 1: Inverse undo excess work (Unrolled loop for n=2)
    q = m
    while q > 1:
        p_val = q - 1
        
        # i = 0 (x-coordinate)
        mask_x = (x & q) != 0
        x = torch.where(mask_x, x ^ p_val, x)

        # i = 1 (y-coordinate) using updated x
        mask_y = (y & q) != 0
        t = (x ^ y) & p_val
        x = torch.where(mask_y, x ^ p_val, x ^ t)
        y = torch.where(mask_y, y, y ^ t)
        
        q >>= 1

    # Part 2: Gray encode
    y ^= x

    # Part 3: Undo t
    t = torch.zeros_like(x)
    q = m
    while q > 1:
        # For n=2, point[n-1] is point[1], which is y
        t ^= torch.where((y & q) != 0, q - 1, 0)
        q >>= 1
    
    x ^= t
    y ^= t

    # Part 4: Transpose to Hilbert integer (Bit Interleaving)
    h = torch.zeros_like(x)
    for i in range(pos_bits - 1, -1, -1):
        x_bit = (x >> i) & 1
        y_bit = (y >> i) & 1
        
        # Interleave bits: x, y
        h = (h << 1) | x_bit
        h = (h << 1) | y_bit
        
    return h

def hilbert_codes2xy_fast(p: int, device: torch.device):
    """
    Calculate 2D Hilbert curve coordinates (x, y) from distance d using PyTorch.
    This implementation is optimized for GPU execution.
    Args:
        p (int): The order of the Hilbert curve (number of iterations).
        device (torch.device): The device to perform computations on (e.g., 'cuda' or 'cpu').
    Returns:
        (torch.Tensor, torch.Tensor): Two tensors containing the x and y coordinates.
    """
    n = 1 << p  # side length of the grid (2^p)
    N = n * n   # total number of points
    d = torch.arange(N, device=device, dtype=torch.int64)

    x = torch.zeros_like(d)
    y = torch.zeros_like(d)

    s = 1
    while s < n:
        rx = (d // 2) & 1
        ry = (d ^ rx) & 1

        # 旋转：仅在 ry == 0 时
        mask = ry == 0
        x_, y_ = x.clone(), y.clone()
        # 交换
        x[mask] = torch.where(rx[mask] == 1, s - 1 - y_[mask], y_[mask])
        y[mask] = torch.where(rx[mask] == 1, s - 1 - x_[mask], x_[mask])

        # 平移
        x += s * rx
        y += s * ry

        d = d // 4
        s *= 2

    return x, y


def hilbert_sort_2d_fast(H: int, W: int, device=None):
    """
    生成 (H, W) 区域内的 Hilbert 顺序索引 (torch.LongTensor)
    """
    if device is None:
        device = torch.device("cpu")
    p = (max(H, W) - 1).bit_length()  # ceil(log2(max(H,W)))
    x, y = hilbert_codes2xy_fast(p, device)
    mask = (x < W) & (y < H)
    order = (y[mask] * W + x[mask]).long()
    return order

def hilbert_codes_3d_fast(pos: torch.Tensor, pos_bits: int = 16) -> torch.Tensor:
    """
    Calculate 3D Hilbert codes for a set of 3D coordinates using PyTorch.
    This implementation is optimized for GPU execution and handles serial dependencies
    correctly by unrolling loops where necessary.
    Args:
        pos (torch.Tensor): Input tensor of shape (N, 3) containing 3D coordinates.
        pos_bits (int): Number of bits to represent each coordinate (default: 16).
    Returns:
        torch.Tensor: Tensor of shape (N,) containing the Hilbert codes.
    """
    
    if pos.dim() != 2 or pos.shape[1] != 3:
        raise ValueError(f"Input tensor 'pos' must have shape (N, 3), but got {pos.shape}")
    coords = pos.long()
    
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    # --- Part 1: Inverse undo excess work (Correctly handles serial dependency) ---
    m = 1 << (pos_bits - 1)
    q = m
    while q > 1:
        p_val = q - 1
        
        # Unroll the inner loop 'for i in range(dim)' to respect data dependencies.
        # ---- i = 0 (x-coordinate) ----
        # This is the 'if point[0] & q:' branch. The 'else' is a no-op for i=0.
        mask_x = (x & q) != 0
        x = torch.where(mask_x, x ^ p_val, x)

        # ---- i = 1 (y-coordinate) ----
        # Now use the UPDATED x from the i=0 step.
        mask_y = (y & q) != 0
        t = (x ^ y) & p_val
        x = torch.where(mask_y, x ^ p_val, x ^ t)
        y = torch.where(mask_y, y, y ^ t)

        # ---- i = 2 (z-coordinate) ----
        # Now use the UPDATED x from the i=1 step.
        mask_z = (z & q) != 0
        t = (x ^ z) & p_val
        x = torch.where(mask_z, x ^ p_val, x ^ t)
        z = torch.where(mask_z, z, z ^ t)
        
        q >>= 1

    # --- Part 2: Gray encode (This part also has a serial dependency) ---
    # Correctly handle dependency: y uses original x, z uses new y.
    y ^= x
    z ^= y # z must be XORed with the *new* y

    # --- Part 3: Undo t ---
    t = torch.zeros_like(x)
    q = m
    while q > 1:
        t ^= torch.where((z & q) != 0, q - 1, 0)
        q >>= 1
    
    x ^= t
    y ^= t
    z ^= t

    # --- Part 4: Transpose to Hilbert integer ---
    h = torch.zeros_like(x)
    for i in range(pos_bits - 1, -1, -1):
        x_bit = (x >> i) & 1
        y_bit = (y >> i) & 1
        z_bit = (z >> i) & 1
        
        # Interleave bits: x, y, z
        h = (h << 1) | x_bit
        h = (h << 1) | y_bit
        h = (h << 1) | z_bit
    
    return h

def hilbert_sort_3d_fast(pos: torch.Tensor, pos_bits: int = 16) -> torch.Tensor:
    """
    Generate 3D Hilbert order indices for a set of 3D coordinates.
    Args:
        pos (torch.Tensor): Input tensor of shape (N, 3) containing 3D coordinates.
        pos_bits (int): Number of bits to represent each coordinate (default: 16).
    Returns:
        torch.Tensor: Tensor of shape (N,) containing the Hilbert order indices.
    """
    codes = hilbert_codes_3d_fast(pos, pos_bits)
    return torch.argsort(codes)

