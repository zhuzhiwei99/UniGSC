'''
Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
Date: 2025-07-06 18:28:46
LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
LastEditTime: 2025-07-24 17:26:00
FilePath: /VGSC/vgsc/utils/gs_io.py
Description: Gaussian Splatting I/O utilities for loading and saving PLY files.

Copyright (c) 2025 by Zhiwei Zhu (zhuzhiwei21@zju.edu.cn), All Rights Reserved. 
'''

import numpy as np
import torch
import logging
import os
from plyfile import PlyData, PlyElement
from typing import Union, Literal, Dict, List, Any, Tuple



def _extract_prefixed_attributes(vertex_data: np.ndarray, prefix: str) -> np.ndarray:
    """Extracts and sorts attributes from PLY data based on a prefix."""
    # Find all property names that start with the given prefix
    prop_names = [p[0] for p in vertex_data.dtype.descr if p[0].startswith(prefix)]
    # Sort them numerically based on the suffix
    prop_names.sort(key=lambda x: int(x.split('_')[-1]))
    
    if not prop_names:
        return np.empty((len(vertex_data), 0))

    # Stack the attributes into a single NumPy array
    return np.stack([vertex_data[name] for name in prop_names], axis=1)


def _create_ply_attribute_list(sh_rest_dim: int, quats_dim: int) -> List[tuple]:
    """Creates the structured dtype list for the PLY file elements."""
    # Position
    attrs = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    
    # Spherical Harmonics (DC)
    attrs.extend([(f'f_dc_{i}', 'f4') for i in range(3)])
    
    # Spherical Harmonics (Rest)
    attrs.extend([(f'f_rest_{i}', 'f4') for i in range(sh_rest_dim * 3)])
    
    # Opacity
    attrs.append(('opacity', 'f4'))
    
    # Scale and Rotation
    attrs.extend([(f'scale_{i}', 'f4') for i in range(3)])
    attrs.extend([(f'rot_{i}', 'f4') for i in range(quats_dim)])
    
    return attrs


def _to_numpy(tensor: Any) -> np.ndarray:
    """Converts a torch.Tensor or torch.nn.Parameter to a NumPy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy() 
    return np.asarray(tensor)


# --- Main I/O Functions ---

def load_ply(
    path: str,
    return_type: Literal['torch_nn', 'torch', 'numpy'] = 'numpy'
) -> Union[torch.nn.ParameterDict, Dict[str, Any]]:
    """
    Loads 3D Gaussian Splatting data from a PLY file.

    Args:
        path (str): Path to the PLY file.
        return_type (str): The desired return type. One of 'numpy', 'torch', 'torch_nn'.

    Returns:
        A dictionary or ParameterDict containing the Gaussian attributes.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the PLY file is missing required elements or attributes.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"PLY file not found at: {path}")

    print(f"Loading PLY file from: {path}")
    plydata = PlyData.read(path)
    
    if not plydata.elements:
        raise ValueError("PLY file contains no elements.")
        
    vertex_data = plydata.elements[0].data
    num_points = len(vertex_data)

    # A mapping from internal names to their extraction logic
    attribute_map = {
        "means": np.stack([vertex_data['x'], vertex_data['y'], vertex_data['z']], axis=1),
        "opacities": vertex_data['opacity'],
        "scales": _extract_prefixed_attributes(vertex_data, 'scale_'),
        "quats": _extract_prefixed_attributes(vertex_data, 'rot_'),
        "sh0": _extract_prefixed_attributes(vertex_data, 'f_dc_').reshape(num_points, 3, 1).transpose(0, 2, 1),
        "shN": _extract_prefixed_attributes(vertex_data, 'f_rest_').reshape(num_points, 3, -1).transpose(0, 2, 1),
    }

    # Final conversion to the desired framework
    if return_type == 'numpy':
        return {k: v.astype(np.float32) for k, v in attribute_map.items()}
    
    torch_dict = {k: torch.from_numpy(v.astype(np.float32)) for k, v in attribute_map.items()}
    
    if return_type == 'torch':
        return torch_dict
        
    if return_type == 'torch_nn':
        return torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in torch_dict.items()})

    raise ValueError(f"Invalid return_type: {return_type}. Must be one of 'numpy', 'torch', 'torch_nn'.")


def save_ply(ply_data: Dict[str, Any], save_path: str):
    """
    Saves 3D Gaussian Splatting data to a PLY file.

    Args:
        ply_data (Dict[str, Any]): A dictionary containing Gaussian attributes.
                                   Values can be np.ndarray, torch.Tensor, or torch.nn.Parameter.
        save_path (str): The path to save the output PLY file.
    
    Raises:
        KeyError: If the input dictionary is missing required attribute keys.
        IOError: If the file cannot be written.
    """
    required_keys = {"means", "opacities", "sh0", "shN", "scales", "quats"}
    if not required_keys.issubset(ply_data.keys()):
        missing = required_keys - ply_data.keys()
        raise KeyError(f"Input dictionary is missing required keys: {missing}")

    # Convert all inputs to NumPy arrays for robust handling
    numpy_dict = {key: _to_numpy(value) for key, value in ply_data.items()}

    num_points = len(numpy_dict['means'])
    # Create the list of attributes for the PLY file header
    dtype_full = _create_ply_attribute_list(sh_rest_dim = numpy_dict['shN'].shape[-2], quats_dim = numpy_dict['quats'].shape[-1])

    # Flatten attributes and concatenate in the correct order
    elements = np.empty(num_points, dtype=dtype_full)
    attributes = np.concatenate(
        (
            numpy_dict['means'],
            numpy_dict['sh0'].transpose(0, 2, 1).reshape(num_points, -1),
            numpy_dict['shN'].transpose(0, 2, 1).reshape(num_points, -1),
            numpy_dict['opacities'].reshape(num_points, -1),
            numpy_dict['scales'],
            numpy_dict['quats'],
        ),
        axis=1,
    )

    # Assign concatenated data to the structured array
    for i, name in enumerate(elements.dtype.names):
        elements[name] = attributes[:, i]

    # Create PlyElement and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    
    try:
        PlyData([vertex_element]).write(save_path)
        print(f"Successfully saved PLY file to: {save_path}")
    except IOError as e:
        print(f"Error: Failed to write PLY file to {save_path}: {e}")
        raise

def load_ply_sequence(ply_dir: str, frame_num: int, 
                      return_type: Literal['torch_nn', 'torch', 'numpy'] = 'torch_nn') -> Tuple[List[Dict], List[str]]:
    """
    Checks if the given path is a directory or a file, and loads PLY files accordingly.

    Args:
        ply_path (str): Path to the PLY file or directory.
        frame_num (int): Number of frames to load if it's a directory.

    Returns:
        List[Dict]: List of loaded splats dictionaries.
    """
    assert os.path.isdir(ply_dir), f"Provided path {ply_dir} is not a directory."

    filenames = [f for f in os.listdir(ply_dir) if f.endswith('.ply')]
    filenames.sort()  # Sort filenames to ensure consistent order

    if frame_num is not None: 
        if frame_num <= len(filenames):
            filenames = filenames[:frame_num]
        else:
            raise ValueError(f"Requested frame_num {frame_num} exceeds available frames {len(filenames)}.")
    splats_list = []
    for filename in filenames:
        input_path = os.path.join(ply_dir, filename)
        splats = load_ply(input_path, return_type)
        splats_list.append(splats)
    return splats_list, filenames


def save_ply_sequence(splats_list: List[Dict[str, Any]], save_dir: str):
    """
    Saves a sequence of splats to PLY files in the specified directory.
    Args:
        splats_list (List[Dict[str, Any]]): List of splats dictionaries to save.
        save_dir (str): Directory where the PLY files will be saved.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i, splats in enumerate(splats_list):
        save_path = os.path.join(save_dir, f"frame_{i:03d}.ply")
        save_ply(splats, save_path)
        print(f"Saved frame {i} to {save_path}") 
        
def format_splats(splats: Dict[str, Any], 
                 return_type: Literal['torch_nn', 'torch', 'numpy'] = 'torch',
                 device=None) -> Dict[str, Any]:
    """
    Formats a single splats dictionary into a list of dictionaries for consistency.
    
    Args:
        splats (Dict[str, Any]): The splats dictionary to format.
        return_type (str): The desired return type. One of 'numpy', 'torch', 'torch_nn'.
    Returns:
        List[Dict]: A list containing the formatted splats dictionary.
    """
    if return_type not in ['numpy', 'torch', 'torch_nn']:
        raise ValueError(f"Invalid return_type: {return_type}. Must be one of 'numpy', 'torch', 'torch_nn'.")
    formatted = {}
    for key in splats:
        val_np = _to_numpy(splats[key])
        if return_type == 'numpy':
            formatted[key] = val_np
        elif return_type == 'torch':
            t = torch.from_numpy(val_np)
            formatted[key] = t.to(device) if device else t
        elif return_type == 'torch_nn':
            t = torch.from_numpy(val_np.astype(np.float32))
            formatted[key] = torch.nn.Parameter(t.to(device) if device else t)
    return formatted

if __name__ == "__main__":
    from gsc import pad_n_splats
    ply_path = '/work/Users/zhuzhiwei/project/3dRecon/compress/zju_gsc/VGSC/data/GSC_splats/m71763_bartender_stable/track/frame000.ply'
    splats = load_ply(ply_path, return_type='torch')
    # splats_double = {}
    # for key, value in splats.items():
    #     splats_double[key] = torch.cat([value, value], dim=0)
    # save_ply(splats_double, '/work/Users/zhuzhiwei/project/3dRecon/compress/zju_gsc/VGSC/frame000_double.ply')
    splats_padded = pad_n_splats(splats, 1000)
    save_path = '/work/Users/zhuzhiwei/project/3dRecon/compress/zju_gsc/VGSC/data/GSC_splats/m71763_bartender_stable/track_padded_1000/frame000.ply'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_ply(splats_padded, save_path)
