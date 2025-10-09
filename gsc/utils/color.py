'''
Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
Date: 2025-07-06 18:28:46
LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
LastEditTime: 2025-07-16 16:20:00
FilePath: /VGSC/vgsc/utils/color.py
Description: Color space conversion utilities for RGB and YCbCr, including ITU-R BT.709, BT.601, and BT.470 standards.

Copyright (c) 2025 by Zhiwei Zhu (zhuzhiwei21@zju.edu.cn), All Rights Reserved. 
'''

import torch
import numpy as np
from typing import Literal

COLOR_SPACE_STANDARD = Literal["BT709", "BT601", "BT470"]

# ==============================================================================
# ITU-R BT.709 (HDTV) - Precise Matrices
# Use for HD, Full HD, 4K UHD content.
# Luma: Y' = 0.2126*R' + 0.7152*G' + 0.0722*B'
# ==============================================================================

# Matrix to convert from linear sRGB (R'G'B') to Y'CbCr
BT709_RGB_TO_YCBCR = [
    [0.2126,      0.7152,      0.0722    ],  # Y'
    [-0.114572,  -0.385428,    0.5       ],  # Cb
    [0.5,        -0.454153,   -0.045847  ]   # Cr
]

# Matrix to convert from Y'CbCr to linear sRGB (R'G'B')
# This is the precise inverse of the above matrix.
BT709_YCBCR_TO_RGB = [
    [1.0,  0.0,        1.5748    ],  # R'
    [1.0, -0.187324,   -0.468124 ],  # G'
    [1.0,  1.8556,      0.0      ]   # B'
]

# ==============================================================================
# ITU-R BT.601 (SDTV) - Precise Matrices
# Use for standard definition content (e.g., from DVDs, older broadcasts).
# Luma: Y' = 0.299*R' + 0.587*G' + 0.114*B'
# ==============================================================================

# Matrix to convert from linear R'G'B' to Y'CbCr
BT601_RGB_TO_YCBCR = [
    [0.299,      0.587,      0.114     ],  # Y'
    [-0.168736, -0.331264,   0.5       ],  # Cb
    [0.5,       -0.418688,  -0.081312  ]   # Cr
]

# Matrix to convert from Y'CbCr to linear R'G'B'
# This is the precise inverse of the above matrix.
BT601_YCBCR_TO_RGB = [
    [1.0,  0.0,        1.402     ],  # R'
    [1.0, -0.344136,   -0.714136 ],  # G'
    [1.0,  1.772,      0.0       ]   # B'
]

# ==============================================================================
# ITU-R BT.470 (PAL/SECAM) - Precise Matrices
# Use for PAL and SECAM content.
# Luma: Y' = 0.299*R' + 0.587*G' + 0.114*B'
# ==============================================================================
BT470_RGB_TO_YCBCR = [
    [0.299,      0.587,      0.114  ],
    [-0.14713,  -0.28886,    0.436  ],
    [0.615,     -0.51498,   -0.10001]
]

BT470_YCBCR_TO_RGB = [
    [1.0,  0.0,        1.13983   ],
    [1.0, -0.39465,   -0.5806    ],
    [1.0,  2.03211,    0.0       ]
]

def rgb_to_ycbcr(rgb : np.array, standard : COLOR_SPACE_STANDARD = "BT470") -> np.array:
    """
    Convert RGB to YCbCr using the specified standard.
    
    Args:
        rgb (np.array): Input RGB image with shape (H, W, 3).
        standard (Literal["BT709", "BT601", "BT470"]): Color space standard to use for conversion.
        
    Returns:
        np.array: Converted YCbCr image with shape (H, W, 3).
    """
    if standard == "BT709":
        matrix = np.array(BT709_RGB_TO_YCBCR, dtype=np.float32)
    elif standard == "BT601":
        matrix = np.array(BT601_RGB_TO_YCBCR, dtype=np.float32)
    elif standard == "BT470":
        matrix = np.array(BT470_RGB_TO_YCBCR, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported color space standard: {standard}")

    original_shape = rgb.shape
    rgb = rgb.reshape(-1, 3)  # Reshape to (H*W, 3) for matrix multiplication  
    # Perform the matrix multiplication
    ycbcr = rgb @ matrix.T
    ycbcr = ycbcr.reshape(original_shape)  # Reshape back to (H, W, 3)
    return ycbcr

def ycbcr_to_rgb(ycbcr : np.array, standard : COLOR_SPACE_STANDARD = "BT470") -> np.array:
    """
    Convert YCbCr to RGB using the specified standard.
    
    Args:
        ycbcr (np.array): Input YCbCr image with shape (H, W, 3).
        standard (Literal["BT709", "BT601", "BT470"]): Color space standard to use for conversion.
        
    Returns:
        np.array: Converted RGB image with shape (H, W, 3).
    """
    if standard == "BT709":
        matrix = np.array(BT709_YCBCR_TO_RGB, dtype=np.float32)
    elif standard == "BT601":
        matrix = np.array(BT601_YCBCR_TO_RGB, dtype=np.float32)
    elif standard == "BT470":
        matrix = np.array(BT470_YCBCR_TO_RGB, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported color space standard: {standard}")
    original_shape = ycbcr.shape
    ycbcr = ycbcr.reshape(-1, 3)  
    # Perform the matrix multiplication
    rgb = ycbcr @ matrix.T
    rgb = rgb.reshape(original_shape)  
    return rgb


def rgb_to_ycbcr_torch(rgb: torch.Tensor, standard: COLOR_SPACE_STANDARD = "BT470") -> torch.Tensor:
    """
    Convert RGB to YCbCr using the specified standard with PyTorch tensors.
    
    Args:
        rgb (torch.Tensor): Input RGB tensor with shape (H, W, 3).
        standard (Literal["BT709", "BT601", "BT470"]): Color space standard to use for conversion.
        
    Returns:
        torch.Tensor: Converted YCbCr tensor with shape (H, W, 3).
    """
    if standard == "BT709":
        matrix = torch.tensor(BT709_RGB_TO_YCBCR, dtype=torch.float32)
    elif standard == "BT601":
        matrix = torch.tensor(BT601_RGB_TO_YCBCR, dtype=torch.float32)
    elif standard == "BT470":
        matrix = torch.tensor(BT470_RGB_TO_YCBCR, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported color space standard: {standard}")
    matrix = matrix.to(rgb.device)  # Ensure the matrix is on the same device as rgb
    # Perform the matrix multiplication
    ycbcr = torch.matmul(rgb, matrix.T)
    return ycbcr

def ycbcr_to_rgb_torch(ycbcr: torch.Tensor, standard: COLOR_SPACE_STANDARD = "BT470") -> torch.Tensor:
    """
    Convert YCbCr to RGB using the specified standard with PyTorch tensors.
    
    Args:
        ycbcr (torch.Tensor): Input YCbCr tensor with shape (H, W, 3).
        standard (Literal["BT709", "BT601", "BT470"]): Color space standard to use for conversion.
        
    Returns:
        torch.Tensor: Converted RGB tensor with shape (H, W, 3).
    """
    if standard == "BT709":
        matrix = torch.tensor(BT709_YCBCR_TO_RGB, dtype=torch.float32)
    elif standard == "BT601":
        matrix = torch.tensor(BT601_YCBCR_TO_RGB, dtype=torch.float32)
    elif standard == "BT470":
        matrix = torch.tensor(BT470_YCBCR_TO_RGB, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported color space standard: {standard}")
    matrix = matrix.to(ycbcr.device)  # Ensure the matrix is on the same device as ycbcr
    # Perform the matrix multiplication
    rgb = torch.matmul(ycbcr, matrix.T)
    return rgb