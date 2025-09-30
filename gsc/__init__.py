'''
Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
Date: 2025-07-06 18:28:46
LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
LastEditTime: 2025-09-24 22:46:56
FilePath: /VGSC/vgsc/__init__.py
Description: Common utilities and configurations for the VGSC library.

Copyright (c) 2025 by Zhiwei Zhu (zhuzhiwei21@zju.edu.cn), All Rights Reserved. 
'''

import torch
import numpy as np
from typing import Dict, Literal, List, Tuple, Any
from torch import Tensor
import random
import logging

_initialized = False

def init_logging(log_file: str, level=logging.INFO):
    global _initialized
    if _initialized:
        return
    _initialized = True

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()  # optional
        ]
    )

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


logger = get_logger("VgscUtils")

# Mapping from string representations of data types to PyTorch data types
TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "float": torch.float,
    "float64": torch.float64,
    "double": torch.double,
    "float16": torch.float16,
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "long": torch.long,
    "int32": torch.int32,
    "int": torch.int,
    "int16": torch.int16,
    "short": torch.short,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def min_max_normalize(tensor: Tensor) -> Tensor:
    """Min-max normalize a tensor to the range [0, 1].
    Args:
        tensor (Tensor): The input tensor to be normalized.
    Returns:
        Tensor: The min-max normalized tensor.
    """
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    if max_val - min_val < 1e-8:
        return torch.zeros_like(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

def deep_update_object(target_obj: object, overrides_dict: dict) -> object:
    for key, value in overrides_dict.items():
        if hasattr(target_obj, key):
            target_attr = getattr(target_obj, key)
            if isinstance(value, dict) and hasattr(target_attr, '__dict__'):
                deep_update_object(target_attr, value)
            else:
                try:
                    setattr(target_obj, key, value)
                except AttributeError:
                    pass
    return target_obj

def get_dtype_for_bitdepth(bitdepth: int) -> Dict[str, any]:
    """Get the appropriate data type for a given bit depth.
    Args:
        bitdepth (int): The bit depth of the data.
    Returns:
        Dict[str, any]: A dictionary containing the data types for PyTorch, NumPy, and the number of bytes per element.
    Raises:
        ValueError: If the bit depth is not supported.
    """
    if bitdepth <= 8:
        return {"torch": torch.uint8, "numpy": np.uint8, "bytes": 1}
    elif bitdepth < 16:
        # less than 16-bit data is stored in 16-bit integers
        return {"torch": torch.int16, "numpy": np.uint16, "bytes": 2}
    elif bitdepth == 16:
        # torch.uint16 is not supported, use int32 instead
        return {"torch": torch.int32, "numpy": np.uint16, "bytes": 2}
    elif bitdepth < 32:
        # less than 32-bit data is stored in 32-bit integers
        return {"torch": torch.int32, "numpy": np.uint32, "bytes": 4}
    else:
        raise ValueError(f"Unsupported bitdepth: {bitdepth}. Supported values are no more than 32.")

def bit_split(data: Tensor, l_bit_depth: int, u_bit_depth:int) -> Tuple[Tensor, Tensor]:
    l_dtype = get_dtype_for_bitdepth(l_bit_depth)["torch"]
    u_dtype = get_dtype_for_bitdepth(u_bit_depth)["torch"]
    total_bit_depth = l_bit_depth + u_bit_depth
    data_dtype = get_dtype_for_bitdepth(total_bit_depth)["torch"]
    # Ensure the grid_attr_data is in the correct dtype
    data = data.to(data_dtype)
    # Split the data into lower and upper bits
    data_l = (data & ((1 << l_bit_depth) - 1)).to(l_dtype)  # Lower bits
    data_u = (data >> l_bit_depth & ((1 << u_bit_depth) - 1)).to(u_dtype)  # Upper bits
    return data_l, data_u  

def deep_update_dict(target_dict, overrides_dict):
    for key, value in overrides_dict.items():
        if key in target_dict:
            target_value = target_dict[key]
            if isinstance(value, dict) and isinstance(target_value, dict):
                # If both are dictionaries, recursively update
                target_dict[key] = deep_update_dict(target_value, value)
            else:
                # Otherwise, just set the value
                target_dict[key] = value
        else:
            target_dict[key] = value
            print(f"Warning: Key '{key}' not found in target_dict. Adding it with value: {value}")
    return target_dict

def smart_look_up_dict(dictionary: Dict[str, any], key: str, default_value: any = None) -> any:
    """    Look up a value in a dictionary by key, allowing for partial key matches.
    Args:
        dictionary (Dict[str, any]): The dictionary to look up the value in.
        key (str): The key to look up in the dictionary.
        default_value (any, optional): The default value to return if the key is not found.
    Returns:
        any: The value associated with the key in the dictionary, or the default value if the key is not found.
    Raises:
        KeyError: If the key is not found in the dictionary and no default value is provided.
    Example:
        >>> my_dict = {'a_b_c': 1, 'a_b_d': 2, 'a_b_e': 3}
        >>> look_up_dict_value(my_dict, 'a_b', default_value=0)
        1
        >>> look_up_dict_value(my_dict, 'a_b_f', default_value=0)
        0
    """
    if key in dictionary:
        return dictionary[key]
    key_parts = key.split('_')
    while key_parts:
        key = '_'.join(key_parts)
        if key in dictionary:
            return dictionary[key]
        key_parts.pop()
    # if no key is found, return the default value
    if default_value is not None:
        logger.warning(f"Key '{key}' not found in dictionary. Returning default value: {default_value}.")
        return default_value
    else:
        raise KeyError(f"Key '{key}' not found in dictionary and no default value provided.")

def pad_n_splats(splats: Dict[str, Tensor], n_pad: int) -> Dict[str, Tensor]:
    """Pad the splats to make them square
    Args:
        splats (Dict[str, Tensor]): Dictionary of splats with keys as attributes and values as tensors.
        n_pad (int): Number of splats to pad to make the total count square.
        
    Notes:
        - Since the input to this function may be quantized data, the padding values should be chosen carefully to avoid impacting rendering quality.
        - For attributes like 'opacities' and 'scales', padding uses the minimum value from existing data to avoid affecting rendering quality and maintain distribution.
        - For attributes such as 'sh0', 'shN', 'means', 'quats', it is recommended to pad with zeros.

    Returns:
        Dict[str, Tensor]: Padded splats dictionary.
    """
    assert n_pad >= 0, "n_pad must be non-negative."
    if n_pad == 0:
        return splats
    for attr_name, attr_data in splats.items():
        pad_shape = list(attr_data.shape)
        pad_shape[0] += n_pad
        pad_shape = tuple(pad_shape)
        if attr_name in ["opacities", "scales"]:
            attr_min = torch.min(attr_data)
            pad_attr_data = torch.full(pad_shape, attr_min.item(), dtype=attr_data.dtype, device=attr_data.device)
        else:
            pad_attr_data = torch.zeros(pad_shape, dtype=attr_data.dtype, device=attr_data.device)
        pad_attr_data[:attr_data.shape[0]] = attr_data
        splats[attr_name] = pad_attr_data
    return splats

def drop_n_splats(splats: Dict[str, Tensor], n_drop: int, 
                  sort_rule : Literal['opacties', 'opacities_scales'] ='opacties') -> Dict[str, Tensor]:
    """Drop the lowest n_splats splats.
    Args:
        splats (Dict[str, Tensor]): Dictionary of splats with keys as attributes and values as tensors.
        n_drop (int): Number of splats to drop.
    Returns:
        Dict[str, Tensor]: Dropped splats dictionary.
    """
    assert n_drop >= 0, "n_drop must be non-negative."
    if n_drop == 0:
        return splats
    n_splats = splats["means"].shape[0]
    assert n_drop < n_splats, f"n_drop ({n_drop}) must be less than the number of splats ({n_splats})."
    if sort_rule == 'opacities_scales':
        # sort by splats['opacities'] * splats['scales'] in descending order, drop the rest
        combined_metric = torch.sigmoid(splats["opacities"]) * torch.exp(splats["scales"])
        sorted_indices = torch.argsort(combined_metric, descending=True)
    else:
        # defualtly sort by splats['opacities'] in descending order, drop the rest
        sorted_indices = torch.argsort(splats["opacities"], descending=True)
    for attr_name, attr_data in splats.items():
        splats[attr_name] = attr_data[sorted_indices[:-n_drop]]
    
    return splats

def resize_splats(splats: Dict[str, Tensor], target_num: int) -> Dict[str, Tensor]:
    """Resize the splats to a specified number of splats.
    Args:
        splats (Dict[str, Tensor]): Dictionary of splats with keys as attributes and values as tensors.
        target_num (int): Number of splats to resize to.
    Returns:
        Dict[str, Tensor]: Resized splats dictionary.
    """
    assert target_num >= 0, "target_num must be non-negative."
    n_splats = splats["means"].shape[0]
    if n_splats == target_num:
        return splats
    elif n_splats < target_num:
        logger.warning(f"Resizing splats from {n_splats} to {target_num} by padding {target_num - n_splats} splats with dummy values.")
        return pad_n_splats(splats, target_num - n_splats)    
    else:
        logger.warning(f"Resizing splats from {n_splats} to {target_num} by dropping {n_splats - target_num} splats with lowest opacities.")
        return drop_n_splats(splats, n_splats - target_num)

def get_shN_sub_names(max_degree : int) -> List[str]:
    """
    Generate a list of spherical harmonics sub-names based on the maximum degree.
    Args:
        max_degree (int): The maximum degree of spherical harmonics.
    Returns:
        List[str]: A list of spherical harmonics sub-names in the format "shN_sh{degree}_{level}".
    """
    shN_name_list = []
    for degree in range(1,max_degree + 1):
        for level in range(-degree, degree+1):
            shN_name_list.append(f"shN_sh{degree}_{level}")

    return shN_name_list   

def splats_list_to_dict(splats_list: List[Dict]) -> Dict[str, Any]:
    """Convert a list of splat dictionaries into a sequence of attributes.
    Args:
        splats_list (List[Dict]): List of splat dictionaries, each containing attributes like means, quats, etc.
    Returns:
        Dict[str, Any]: A dictionary where keys are attribute names and values are tensors or numpy arrays of attributes
        stacked across the frames.
    """
    if not splats_list:
        logger.error("Input splats_list is empty. Cannot convert to attribute sequence.")
        return {}
    sample_splat = splats_list[0]
    attribute_names = list(sample_splat.keys())

    splats_dict = {}
    for attr_name in attribute_names:
        # Stack the attributes across frames
        attr_seq = [splat[attr_name] for splat in splats_list if attr_name in splat]
        if isinstance(attr_seq[0], np.ndarray):
            splats_dict[attr_name] = np.stack(attr_seq, axis=0)
        elif isinstance(attr_seq[0], Tensor):
            splats_dict[attr_name] = torch.stack(attr_seq, dim=0)
        else:
            logger.error(f"Unsupported attribute type for {attr_name}. Expected numpy array or PyTorch tensor.")
            raise TypeError(f"Unsupported attribute type for {attr_name}. Expected numpy array or PyTorch tensor.")
    return splats_dict

def splats_dict_to_list(splats_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a dictionary of attributes into a list of splat dictionaries.
    Args:
        splats_dict (Dict[str, Any]): A dictionary where keys are attribute names and values are tensors or numpy arrays of attributes
    Returns:
        List[Dict[str, Any]]: A list of splat dictionaries, each containing attributes like means, quats, etc.
    """
    if not splats_dict:
        logger.error("Input splats_dict is empty. Cannot convert to splats list.")
        return []
    
    n_frames = splats_dict['means'].shape[0]
    splats_list = []
    
    for i in range(n_frames):
        splat = {attr_name: attr_data[i] for attr_name, attr_data in splats_dict.items()}
        splats_list.append(splat)
    
    return splats_list

