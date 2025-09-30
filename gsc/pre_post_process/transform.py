'''
Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
Date: 2025-07-06 18:28:46
LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
LastEditTime: 2025-10-01 00:40:31
FilePath: /VGSC/vgsc/pre_post_process/transform.py
Description: Transform splats parameters between different domains, such as log transform for means, RGB to YCbCr for SH coefficients, and quaternion normalization.

Copyright (c) 2025 by Zhiwei Zhu (zhuzhiwei21@zju.edu.cn), All Rights Reserved. 
'''

from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor
import os
import numpy as np
from gsc.utils.gs_math import (
    log_transform, 
    inverse_log_transform, 
    quaternion_normalize,
    quaternion_positive_w,   
    quaternion_to_euler, 
    euler_to_quaternion,
    quaternion_to_rodrigues,
    rodrigues_to_quaternion,
    )
from gsc.pre_post_process.pca import pca_transform_torch, pca_inverse_transform_torch
from gsc.utils.color import rgb_to_ycbcr_torch, ycbcr_to_rgb_torch
from gsc import get_logger
logger = get_logger("Transformer")


def param_transform(splats_list: List[Dict], **kwargs) -> List[Dict]:
    """
    Transform the splats parameters to new domain
    e.g.  
    for means: apply log transform
    for SH coefficients, transform them from RGB domain to YCbCr domain
    for quats:
        1) normalize to unit quaternion
        2) transform to Euler angles or other equivalent representations (Optional, depends on kwargs)

    Args:                                       
        splats_list (List[Dict]): splats parameters of all frames
        transform_attributes (Dict[str, bool]): indicate attributes whether to transform domain
    Returns:
        List[Dict]: transformed splats parameters of all frames
    kwargs:
        trans_quat_euler (bool): whether to transform quaternions to Euler angles, default is False
        color_standard (str): color standard for RGB to YCbCr transformation, can be "BT709" or "BT470", default is "BT709"
    """
    transformed_splats_list = [] 
    
    for splats in splats_list:
        transformed_splats = {}
        
        # Process all keys in the splats dictionary
        for key, value in splats.items():
            if key == "means": 
                if kwargs.get("trans_means_log", False):
                    transformed_splats[key] = log_transform(value)
                    logger.info(f"Transformed {key} with log transform")
                else:
                    transformed_splats[key] = value
                    
            elif key == "quats":                
                if kwargs.get("trans_quats_norm", False):
                    transformed_splats[key] = quaternion_normalize(value)
                    value = transformed_splats[key]
                    logger.info(f"Transformed {key} to unit quaternions")
                    
                if kwargs.get("trans_quats_posi_w", False):
                    transformed_splats[key] = quaternion_positive_w(value)
                    value = transformed_splats[key]
                    logger.info(f"Transformed {key} to quaternions with positive w")
                
                if kwargs.get("trans_quats_euler", False):
                    transformed_splats[key] = quaternion_to_euler(value)
                    logger.info(f"Transformed {key} to Euler angles")
                elif kwargs.get("trans_quats_rod", False):
                    transformed_splats[key] = quaternion_to_rodrigues(value)
                    logger.info(f"Transformed {key} from Rodrigues vector to quaternions")
                else:
                    transformed_splats[key] = value
                    
            # SH coefficients - transform from RGB to YCbCr
            elif key == 'sh0':
                if kwargs.get("trans_sh0_ycbcr", False):                  
                    transformed_splats[key], color_standard = rgb_to_ycbcr_help(value, **kwargs)
                    logger.info(f"Transformed {key} from RGB to YCbCr using {color_standard} standard")
                else:
                    transformed_splats[key] = value
            elif key == 'shN':
                if kwargs.get("trans_shN_ycbcr", False):                  
                    transformed_splats[key], color_standard = rgb_to_ycbcr_help(value, **kwargs)
                    logger.info(f"Transformed {key} from RGB to YCbCr using {color_standard} standard")
                else:
                    transformed_splats[key] = value
            
            # For all other attributes, keep as is
            else:
                transformed_splats[key] = value
        
        transformed_splats_list.append(transformed_splats)
    
    return transformed_splats_list
    
def param_inverse_transform(splats_list: List[Dict], **kwargs) -> List[Dict]:
    """
    Inverse transform the splats parameters to original domain
    
    Args:
        splats_list (List[Dict]): splats parameters of all frames
        transform_attributes (Dict[str, bool]): indicate attributes whether to transform domain
    Returns:
        List[Dict]: inverse transformed splats parameters of all frames
    """
    inverse_transformed_splats_list = []
  
    for splats in splats_list:
        inverse_transformed_splats = {}
        
        # Process all keys in the splats dictionary
        for key, value in splats.items():
            if key == "means":
                if kwargs.get("trans_means_log", False):
                    # Apply inverse log transform to means
                    inverse_transformed_splats[key] = inverse_log_transform(value)
                    logger.info(f"Inverse transformed {key} with inverse log transform")
                else:
                    inverse_transformed_splats[key] = value

            elif key == "quats":
                if kwargs.get("trans_quats_euler", False):
                    inverse_transformed_splats[key] = euler_to_quaternion(value)
                    logger.info(f"Inverse transformed {key} from Euler angles to quaternions")
                elif kwargs.get("trans_quats_rod", False):
                    # Convert Rodrigues vector back to quaternion
                    inverse_transformed_splats[key] = rodrigues_to_quaternion(value)
                    logger.info(f"Inverse transformed {key} from Rodrigues vector to quaternions")
                else:
                    inverse_transformed_splats[key] = value

            elif key == 'sh0':
                if kwargs.get("trans_sh0_ycbcr", False):                  
                    inverse_transformed_splats[key], color_standard = ycbcr_to_rgb_help(value, **kwargs)
                    logger.info(f"Inverse transformed {key} from YCbCr to RGB using {color_standard} standard")
                else:
                    inverse_transformed_splats[key] = value
            elif key == 'shN':
                if kwargs.get("trans_shN_ycbcr", False):                  
                    inverse_transformed_splats[key], color_standard = ycbcr_to_rgb_help(value, **kwargs)
                    logger.info(f"Inverse transformed {key} from YCbCr to RGB using {color_standard} standard")
                else:
                    inverse_transformed_splats[key] = value
            
            # For all other attributes, keep as is (quaternions already normalized)
            else:
                inverse_transformed_splats[key] = value
        
        inverse_transformed_splats_list.append(inverse_transformed_splats)
    
    return inverse_transformed_splats_list

def rgb_to_ycbcr_help(rgb_data: Tensor, **kwargs) -> Tuple[Tensor, str]:
    
    # Determine the tensor shape to handle both sh0 and shN correctly
    original_shape = rgb_data.shape
    assert original_shape[-1]==3, "The last dimension of RGB data should be 3."
    
    # Reshape to make the RGB dimension the last dimension and all others flattened
    rgb_flat = rgb_data.reshape(-1, 3)
    
    # Convert RGB -> YCbCr 
    color_standard=kwargs.get('color_standard', None)
    if color_standard is None:
        logger.warning("No color standard provided, defaulting to 'BT709'.")
        color_standard = 'BT709'
    ycbcr_flat = rgb_to_ycbcr_torch(rgb_flat, color_standard)   
    # Reshape back to original shape
    ycbcr_data = ycbcr_flat.reshape(original_shape)

    return ycbcr_data, color_standard

def ycbcr_to_rgb_help(ycbcr_data: Tensor, **kwargs) -> Tuple[Tensor, str]:
    
    # Determine the tensor shape to handle both sh0 and shN correctly
    original_shape = ycbcr_data.shape
    last_dim = original_shape[-1]
    
    assert last_dim == 3, "The last dimension of YCbCr data should be 3."
    # Reshape to make the YCbCr dimension the last dimension and all others flattened
    ycbcr_flat = ycbcr_data.reshape(-1, 3)
    # Convert YCbCr -> RGB
    color_standard=kwargs.get('color_standard', None)
    if color_standard is None:
        logger.warning("No color standard provided, defaulting to 'BT709'.")
        color_standard = 'BT709'
    rgb_flat = ycbcr_to_rgb_torch(ycbcr_flat, color_standard)
    # Reshape back to original shape
    rgb_data = rgb_flat.reshape(original_shape)
    
    return rgb_data, color_standard

def seq_sh_pca_transform(splats_list: List[Dict[str, Tensor]], pca_info_dir: str, rank: int) -> List[Dict[str, Tensor]]:
    """
    Apply PCA transform to shN and save the PCA info. 
    Args:
        splats_list (List[Dict[str, Tensor]]): List of splats parameters for all frames.
        pca_info_dir (str): Directory to save PCA information.
        rank (int): Rank of the PCA transformation.
    Returns:
        List[Dict[str, Tensor]]: List of splats parameters with shN transformed to PCA domain.
    """

    npz_dict = {}
    for frame_id, splats in enumerate(splats_list):
        shN = splats["shN"].data
        x = shN.reshape(shN.shape[0], -1)  # [N, d]
        pca_info = pca_transform_torch(x, rank=rank)

        x = pca_info['transformed']
        pca_v = pca_info["pca_v"]
        mean = pca_info["mean"]
        std = pca_info["std"]

        splats["shN"].data = x.reshape(shN.shape[0], -1, 3)
        splats_list[frame_id] = splats

        # Save PCA components with flat keys
        fid = f"frame{frame_id:03d}"
        npz_dict[f"{fid}_pca_v"] = pca_v.detach().cpu().numpy()
        if mean is not None:
            npz_dict[f"{fid}_mean"] = mean.detach().cpu().numpy()
        if std is not None:
            npz_dict[f"{fid}_std"] = std.detach().cpu().numpy()

    # Save all to one .npz file
    pca_info_path = os.path.join(pca_info_dir, "shN_pca_meta.npz")
    np.savez_compressed(pca_info_path, **npz_dict)
    logger.info(f"Sucessfully performed PCA transform on shN and saved PCA info to {pca_info_path}")
    return splats_list

    
def seq_sh_pca_inverse_transform(splats_list: List[Dict[str, Tensor]], pca_info_dir : str, rank : int) -> List[Dict[str, Tensor]]:
    """
    Apply PCA inverse transform to shN.
    Args:
        splats_list (List[Dict[str, Tensor]]): List of splats parameters for all frames.
        pca_info_dir (str): Directory containing PCA information.
        rank (int): Rank of the PCA transformation.
    Returns:
        List[Dict[str, Tensor]]: List of splats parameters with shN transformed back to original domain.
    """
    npz_path = os.path.join(pca_info_dir, "shN_pca_meta.npz")
    npz_dict = np.load(npz_path)
    device = splats_list[0]["shN"].data.device 

    for frame_id, splats in enumerate(splats_list):
        fid = f"frame{frame_id:03d}"
        shN = splats["shN"].data
        if shN.shape[1] > rank//3:
            print(f"Warning: shN data shape {shN.shape[1] * 3} is larger than transform rank {rank}.")
        shN = shN[:, :rank//3, :]  # [N, d, 3]
        pca_info = {}

        # Restore PCA components from flat keys
        pca_info["pca_v"] = torch.from_numpy(npz_dict[f"{fid}_pca_v"]).to(device)

        if f"{fid}_mean" in npz_dict:
            pca_info["mean"] = torch.from_numpy(npz_dict[f"{fid}_mean"]).to(device)
        if f"{fid}_std" in npz_dict:
            pca_info["std"] = torch.from_numpy(npz_dict[f"{fid}_std"]).to(device)
        pca_info["transformed"] = shN.reshape(shN.shape[0], -1).to(device)  # [N, d]
        x = pca_inverse_transform_torch(pca_info)
        splats["shN"].data = x.reshape(shN.shape[0], -1, 3)
        splats_list[frame_id] = splats
    logger.info(f"Sucessfully performed PCA inverse transform on shN using info from {npz_path}")
    return splats_list

