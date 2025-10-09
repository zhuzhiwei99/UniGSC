'''
Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
Date: 2025-07-22 11:23:48
LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
LastEditTime: 2025-10-01 00:41:39
FilePath: /VGSC/vgsc/quantize/gs_quantize.py
Description: 

Copyright (c) 2025 by Zhiwei Zhu, All Rights Reserved. 
'''

from typing import Dict, List, Any, Tuple, Optional, Literal
import torch
from torch import Tensor
import numpy as np
import os
from typing import Dict
from gsc import get_logger, splats_list_to_dict, splats_dict_to_list
from gsc.quantize import quantize_torch, dequantize_torch
logger = get_logger("Quantizer")

import numpy as np


def get_attr_min_range(attr_data: Tensor, attr_name: str, 
                       min_val=None, range_val=None, max_val=None,
                       quant_per_channel: bool=False,
                       quant_shN_per_channel: bool=False,
                       keep_spatial: bool=False,):
    """Calculate the min and range of the attribute data.
    Args:
        attr_data (Tensor): The attribute data tensor. [Num_frames, Num_splats, C, ...]
    """
    if attr_name == 'means':
        attr_data = attr_data.view(-1, 3)  
        if min_val is None:
            min_val = torch.min(attr_data, dim=0).values
        if range_val is None:
            if max_val is None:
                max_val = torch.max(attr_data, dim=0).values
            range_val = max_val - min_val
            if keep_spatial:
                range_val = torch.max(range_val)
    elif attr_name == 'shN':
        if quant_shN_per_channel:
            attr_data = attr_data.view(-1, *attr_data.shape[2:])    
            if min_val is None:
                min_val = torch.min(attr_data, dim=0).values
            if range_val is None:
                if max_val is None:
                    max_val = torch.max(attr_data, dim=0).values
                range_val = max_val - min_val       
        else:
            if min_val is None:
                min_val = torch.min(attr_data)
            if range_val is None:
                if max_val is None:
                    max_val = torch.max(attr_data)
                range_val = max_val - min_val
    else: 
        if quant_per_channel:
            attr_data = attr_data.view(-1, *attr_data.shape[2:])    
            if min_val is None:
                min_val = torch.min(attr_data, dim=0).values
            if range_val is None:
                if max_val is None:
                    max_val = torch.max(attr_data, dim=0).values
                range_val = max_val - min_val       
        else:
            if min_val is None:
                min_val = torch.min(attr_data)
            if range_val is None:
                if max_val is None:
                    max_val = torch.max(attr_data)
                range_val = max_val - min_val
    return min_val, range_val


def quantize_splats_dict_torch(splats: Dict[str, Tensor], quant_config: Dict,
                               quant_per_channel: bool=False, 
                               quant_shN_per_channel: bool=False,
                               keep_spatial: bool = False) -> Tuple[Dict[str, Tensor], Dict]:
    """Quantizes the splats attributes using the specified quantization configuration.
    Args:
        splats (Dict[str, Tensor]): Dictionary of splats attributes.
        quant_config (Dict): Quantization configuration for each attribute.
        keep_spatial (bool): Whether to keep the spatial information when calculating range.
                             If True, the range is calculated across all spatial dimensions.
                             If False, the range is calculated independently for each spatial dimension.
                             Default is True.
    Returns:
        Tuple[Dict[str, Tensor], Dict]: A tuple containing the quantized splats and the quantization metadata.
    """
    quant_splats = {}
    quant_meta = {}

    for attr_name, attr_data in splats.items():
        assert attr_name in quant_config, f"Attribute {attr_name} not found in quantization config."
        assert 'bit_depth' in quant_config[attr_name], f"Bit depth not specified for {attr_name} in quantization config."
        
        cfg = quant_config[attr_name]
        min_val, range_val = get_attr_min_range(attr_data, attr_name, 
                                                cfg.get('min', None), 
                                                cfg.get('range', None), 
                                                cfg.get('max', None),
                                                quant_per_channel,
                                                quant_shN_per_channel,
                                                keep_spatial)

        quant_splats[attr_name] = quantize_torch(attr_data, cfg['bit_depth'], min_val, range_val)  
        quant_meta[attr_name] = {
            'min': min_val.tolist() if isinstance(min_val, Tensor) else min_val,
            'range': range_val.tolist() if isinstance(range_val, Tensor) else range_val,
            'bit_depth': cfg['bit_depth']
        }
    return quant_splats, quant_meta
    
def dequantize_splats_dict_torch(quant_splats: Dict[str, Tensor], quant_meta:  Dict) -> Dict[str, Tensor]:
    """De-quantizes the quantized splats attributes using the quantization metadata.
    Args:
        quant_splats (Dict[str, Tensor]): Dictionary of quantized splats attributes.
        quant_meta (Dict): Quantization metadata for each attribute.
    Returns:
        Dict[str, Tensor]: A dictionary containing the de-quantized splats attributes.
    """
    
    splats = {}

    for attr_name, attr_data in quant_splats.items():
        bit_depth = quant_meta[attr_name]['bit_depth']
        min_val = quant_meta[attr_name]['min']
        range_val = quant_meta[attr_name]['range']

        splats[attr_name] = dequantize_torch(attr_data, bit_depth, min_val, range_val)

    return splats


def update_quant_bit_depth(
    quant_config: Dict,
    bit_depth_config: Optional[Dict[str, Any]] = None
) -> Dict:
    """
    Updates the quantization configuration with the specified bit depth configuration.

    Args:
        quant_config (Dict): Original quantization configuration.
        bit_depth_config (Optional[Dict[str, Any]]): Bit depth configuration for each attribute.

    Returns:
        Dict: Updated quantization configuration.
    """
    logger.info("Updating quantization configuration with bit depth settings.")
    bit_depth_keys = bit_depth_config.keys() if bit_depth_config is not None else []
    if bit_depth_config is not None:
        logger.info(f'Initial quantization configuration: {quant_config}')
        logger.info(f"Update quantization configuration with bit depth configuration: {bit_depth_config}")
        for attr_name in quant_config.keys():
            if attr_name in bit_depth_config:
                quant_config[attr_name]['bit_depth'] = bit_depth_config[attr_name]
            else:
                attr_list = [key for key in bit_depth_keys if attr_name in key]
                if len(attr_list) > 0:
                    attr_bit_depth = 0
                    for attr in attr_list:
                        attr_bit_depth += bit_depth_config[attr]
                    if attr_bit_depth > 0:
                        quant_config[attr_name]['bit_depth'] = attr_bit_depth
                else:
                    logger.warning(f"Attribute {attr_name} not found in bit depth configuration. Using default quantization configuration.")
    else:
        logger.warning("No bit depth configuration provided. Using default quantization configuration.")
    logger.info(f"Final quantization configuration: {quant_config}")
    return quant_config      
       
def quantize_splats_list_seperately(
    splats_list: List[Dict[str, Tensor]],
    quant_config: Dict,
    bit_depth_config: Optional[Dict[str, Any]] = None,
    quant_per_channel: bool=False,
    quant_shN_per_channel: bool=False,
    keep_spatial: bool = False,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Quantizes a list of splats dictionaries separately and returns the quantized splats and metadata.

    Args:
        splats_list (List[Dict]): List of splats dictionaries.
        quant_config (Dict): Quantization configuration for each attribute.
    Returns:
        List[Dict]: List of quantized splats dictionaries.
    """
    quant_splats_list = []
    quant_meta_list = []
    quant_config = update_quant_bit_depth(quant_config, bit_depth_config)
    
    for i, splats in enumerate(splats_list):
        quant_splats, quant_meta = quantize_splats_dict_torch(splats, quant_config,
                                                              quant_per_channel, 
                                                              quant_shN_per_channel,
                                                              keep_spatial)
        quant_splats_list.append(quant_splats)
        quant_meta_list.append(quant_meta)
        logger.info(f"Successfully quantized splats {i+1}/{len(splats_list)}")
    
    return quant_splats_list, quant_meta_list

def dequantize_splats_list_seperately(
    quant_splats_list: List[Dict[str, Tensor]],
    quant_meta_list: List[Dict],
) -> List[Dict]:
    """ De-quantizes a list of quantized splats dictionaries separately and returns the de-quantized splats.    
    Args:
        quant_splats_list (List[Dict]): List of quantized splats dictionaries.
        meta_dir (str): Directory to load metadata from.
    Returns:
        List[Dict]: List of de-quantized splats dictionaries.
    """
    dequant_splats_list = []
    for i, quant_splats in enumerate(quant_splats_list):
        quant_meta = quant_meta_list[i]
        splats = dequantize_splats_dict_torch(quant_splats, quant_meta)

        dequant_splats_list.append(splats)
        logger.info(f"Successfully de-quantized splats {i+1}/{len(quant_splats_list)}")
    return dequant_splats_list

def quantize_splats_list_jointly(
    splats_list: List[Dict[str, Tensor]],
    quant_config: Dict,
    bit_depth_config: Optional[Dict[str, Any]] = None,
    quant_per_channel: bool=False,
    quant_shN_per_channel: bool=False,
    keep_spatial: bool = False,
) -> Tuple[List[Dict], Dict]:
    """
    Quantizes a list of splats dictionaries jointly and saves them to disk.

    Args:
        splats_list (List[Dict]): List of splats dictionaries.
        quant_config (Dict): Quantization configuration for each attribute.

    Returns:
        List[Dict]: List of quantized splats dictionaries.
    """
    splats_dict = splats_list_to_dict(splats_list)
    quant_config = update_quant_bit_depth(quant_config, bit_depth_config)

    quant_splats, quant_meta = quantize_splats_dict_torch(splats_dict, quant_config, 
                                                          quant_per_channel, 
                                                          quant_shN_per_channel,
                                                          keep_spatial,)

    quant_splats_list = splats_dict_to_list(quant_splats)

    return quant_splats_list, quant_meta

def dequantize_splats_list_jointly(
    quant_splats_list: List[Dict[str, Tensor]],
    quant_meta: Dict,
) -> List[Dict]:
    """
    De-quantizes a list of quantized splats dictionaries jointly and returns the de-quantized splats.

    Args:
        quant_splats_list (List[Dict]): List of quantized splats dictionaries.
        meta_dir (str): Directory to load metadata from.

    Returns:
        List[Dict]: List of de-quantized splats dictionaries.
    """
    splats_dict = splats_list_to_dict(quant_splats_list)

    dequant_splats_dict = dequantize_splats_dict_torch(splats_dict, quant_meta)

    dequant_splats_list = splats_dict_to_list(dequant_splats_dict)
    return dequant_splats_list