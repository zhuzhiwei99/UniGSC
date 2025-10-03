'''
Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
Date: 2025-07-06 18:28:46
LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
LastEditTime: 2025-10-03 00:55:46
FilePath: /UniGSC/gsc/config.py
Description: Macro and configuration definitions for the VGSC library.

Copyright (c) 2025 by Zhiwei Zhu (zhuzhiwei21@zju.edu.cn), All Rights Reserved. 
'''

from typing import Literal
# The suffix for quantization metadata files
QUANT_META_NAME = 'meta_quant.json'
# The suffix for compression metadata files
CODEC_META_NAME = 'meta_codec.json'
# The file name for processing information
INFO_NAME = 'info.json'
# The suffix for reconstructed files
DEC_SUFFIX = "_decoded"

SORT_TYPES = Literal["plas", "morton"]
# The supported pixel formats 
PIXEL_FORMATS = Literal["yuv420p", "yuv444p", "yuv400p"]

# The supported downsampling methods 
SUBSAMPLING_METHODS = Literal["skip", "average_pool", "gaussian_blur", "bicubic"]
UPSAMPLING_METHODS = Literal["bilinear", "bicubic"]

# Quantization configuration for different attributes, 
QUANT_CONFIG = {
# Fisrt defined in 148-m69429-v1-[JEE6] GeS-TM as anchor for 3D gaussian coding
    'N00677': {
        'means':        {'bit_depth': 18, 'min': None, 'range': 256.0},  # (val - min) / 256
        'opacities':    {'bit_depth': 12, 'min': -7.0, 'rangr': 25.0}, # (val + 7) / 25
        'sh0':          {'bit_depth': 12, 'min': -4.0, 'range': 8.0},  # (val / 8) + 0.5
        'shN':          {'bit_depth': 12, 'min': -4.0, 'range': 8.0},  # (val / 8) + 0.5
        'scales':       {'bit_depth': 12, 'min': -26.0,'range': 30.0}, # (val + 26) / 30
        'quats':        {'bit_depth': 12, 'min': -1.0, 'range': 2.0},  # (val + 1) / 2
    },
# Defined in 151-MDS25512_WG07_N01292-CTC
    'pcc_N01292': {
        'means':        {'bit_depth': 18, 'min': None, 'range': 256.0},  
        'opacities':    {'bit_depth': 12, 'min': -7.0, 'rangr': 25.0}, 
        'sh0':          {'bit_depth': 12, 'min': -4.0, 'range': 8.0},  
        'shN':          {'bit_depth': 12, 'min': -4.0, 'range': 8.0},  
        'scales':       {'bit_depth': 12, 'min': -26.0,'range': 30.0}, 
        'quats':        {'bit_depth': 12, 'min': -1.0, 'range': 2.0},  
    },
    'video_N01292': {
        'means':        {'bit_depth': 16, 'min': None, 'range': None},  
        'opacities':    {'bit_depth': 10, 'min': None, 'range': None},
        'sh0':          {'bit_depth': 10, 'min': None, 'range': None}, 
        'shN':          {'bit_depth': 10, 'min': None, 'range': None},  
        'scales':       {'bit_depth': 10, 'min': None, 'range': None}, 
        'quats':        {'bit_depth': 10, 'min': None, 'range': None},  
    },
    'video_clamp': {
        'means':        {'bit_depth': 16, 'min': None, 'range': None},  
        'opacities':    {'bit_depth': 10, 'min': None, 'range': None, 'max': 7.0},
        'sh0':          {'bit_depth': 10, 'min': None, 'range': None}, 
        'shN':          {'bit_depth': 10, 'min': None, 'range': None},  
        'scales':       {'bit_depth': 10, 'min': -8.0, 'range': None}, 
        'quats':        {'bit_depth': 10, 'min': None, 'range': None},  
    },
}
QUANT_CONFIG_KEYS = Literal['N00677', 'pcc_N01292', 'video_N01292', 'video_clamp']

DEFAULT_BIT_DEPTH = {
    "means_l": 8,
    "means_u": 8,
    "opacities": 8,
    "quats": 8,
    "scales": 8,
    "sh0": 8,
    "shN": 8,
}

DEFAULT_QP = {
    "means": -1,
    "opacities": 4,
    "quats": 4,
    "scales": 4,
    "sh0": 4,
    "shN": 4,
}

ATTRIBUTE_MAP = {
    "position": "means",
    "sh0": "sh0",
    "sh1": "shN_sh1",
    "sh2": "shN_sh2",
    "sh3": "shN_sh3",  
    "rotation": "quats",
    "scaling": "scales",
    "opacity": "opacities",
    "metadata": "meta",
}