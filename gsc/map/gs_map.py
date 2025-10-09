from typing import Dict, List, Tuple, Literal, Optional
import math
import torch
import numpy as np
from torch import Tensor
from gsc.map import morton_code
from gsc import get_logger, bit_split, min_max_normalize
from gsc.quantize import quantize_torch
from gsc.quantize.gs_quantize import get_attr_min_range
from torchpq.clustering import KMeans
logger = get_logger("Mapper")


def softmax_dict(input_dict):
    keys = list(input_dict.keys())
    values = np.array(list(input_dict.values()))
    exp_values = np.exp(values)
    softmax_values = exp_values / np.sum(exp_values) * len(keys)
    result_dict = {key: float(value) for key, value in zip(keys, softmax_values)}
    return result_dict

def sort_splats_plas(
        splats: Dict[str, Tensor], 
        verbose: bool = True, 
        sort_with_shN: bool = False,
        weight_dict: Dict[str, float] = None,
        ) -> Tensor:
    """Sort splats with Parallel Linear Assignment Sorting from the paper `Compact 3D Scene Representation via
    Self-Organizing Gaussian Grids <https://arxiv.org/pdf/2312.13299>`_.

    .. warning::
        PLAS must installed to use sorting.

    Args:
        splats (Dict[str, Tensor]): splats
        verbose (bool, optional): Whether to logger.info verbose information. Default to True.
        return_indices (bool, optional): Whether to return sorted indices. Default to False.
        sort_with_shN (bool, optional): Whether to consider shN when sorting. Default to False.

    Returns:
        Dict[str, Tensor]: sorted splats
    """
    try:
        from plas import sort_with_plas
    except:
        raise ImportError(
            "Please install PLAS with 'pip install git+https://github.com/fraunhoferhhi/PLAS.git' to use sorting"
        )

    n_gs = len(splats["means"])
    n_sidelen = int(n_gs**0.5)
    assert n_sidelen**2 == n_gs, "Must be a perfect square"

    if not sort_with_shN:
        sort_keys = [k for k in splats if k != "shN"]
    else:
        sort_keys = [k for k in splats]
    if weight_dict is not None:
        weight_dict = softmax_dict(weight_dict)
        params_to_sort = torch.cat([min_max_normalize(splats[k].reshape(n_gs, -1)) * weight_dict[k] for k in sort_keys], dim=-1)
    else:
        params_to_sort = torch.cat([min_max_normalize(splats[k].reshape(n_gs, -1)) for k in sort_keys], dim=-1)
    shuffled_indices = torch.randperm(
        params_to_sort.shape[0], device=params_to_sort.device
    )
    params_to_sort = params_to_sort[shuffled_indices]
    grid = params_to_sort.reshape((n_sidelen, n_sidelen, -1))
    _, sorted_indices = sort_with_plas(
        grid.permute(2, 0, 1), improvement_break=1e-4, verbose=verbose
    )
    sorted_indices = sorted_indices.squeeze().flatten()
    sorted_indices = shuffled_indices[sorted_indices]
    for k, v in splats.items():
        splats[k] = v[sorted_indices]
    
    return  sorted_indices

    

def sort_points_by_space_curve(
    points: Tensor,
    pos_bits: int = 21,
    curve_type: Literal['morton', 'hilbert'] = 'morton'
) -> Tensor:
    """Sorts 3D positions based on space-filling curves (Morton or Hilbert).

    Args:
        points (Tensor): Input positions of shape (N, 3).
        pos_bits (int): Number of quantization bits for position encoding.
        curve_type (Literal['morton', 'hilbert']): Type of space-filling curve to use.

    Returns:
        Tensor: Indices that would sort the positions by the chosen curve.
    """
    assert points.dim() == 2 and points.size(1) == 3, "Input points should be of shape (N, 3)"
    if torch.is_floating_point(points):
        logger.warning(
            "Input points are floating point. Quantizing to integer space for sorting."
        )
        # Quantize the points to integer space
        min_val, range_val = get_attr_min_range(attr_data=points, attr_name='means', keep_spatial=True)
        xyz_q = quantize_torch(points, pos_bits, min_val, range_val)
    else:
        xyz_q = points

    if curve_type == 'morton':
        return morton_code(xyz_q, pos_bits).argsort()
    else:
        raise ValueError(f"Unsupported curve type: {curve_type}. Use 'morton' or 'hilbert'.")
  