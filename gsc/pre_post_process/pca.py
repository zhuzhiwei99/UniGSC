'''
Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
Date: 2025-07-06 18:28:46
LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
LastEditTime: 2025-07-08 13:35:34
FilePath: /zju_gsc/vgsc/vgsc/pre_post_process/pca.py
Description: 

Copyright (c) 2025 by Zhiwei Zhu (zhuzhiwei21@zju.edu.cn), All Rights Reserved. 
'''

import numpy as np
import torch
from typing import Dict, Literal, Optional
from torch import Tensor

def pca_transform_numpy(matrix: np.ndarray, 
                        rank: int = 21,
                        standardize: Optional[Literal['norm', 'center']] ='norm') -> Dict[str, np.ndarray]:
    """
    Perform PCA on a 2D numpy array (N, D) and return the transformed data along with PCA components.
    
    Args:
        matrix (np.ndarray): Input array of shape (N, D) where N is the number of samples and D is the number of features.
        rank (int): Number of principal components to retain.
        standardize (Optional[Literal['norm', 'center']]): If 'norm', standardize the data to have zero mean and unit variance; if 'center', center the data to have zero mean; if None, do not standardize.
    Returns:
        Dict[str, np.ndarray]: A dictionary containing:
            - 'transformed': The PCA transformed data of shape (N, rank).
            - 'pca_v': The PCA components of shape (rank, D).
            - 'mean': The mean of the original data if standardization was applied, otherwise None.
            - 'std': The standard deviation of the original data if standardization was applied, otherwise None.
            - 'rank': The number of principal components retained.
    """
    assert matrix.ndim == 2, "Input matrix must be 2D (N, D)"
    assert rank > 0, "Rank must be a positive integer"
    assert rank <= matrix.shape[1], "Rank must be less than or equal to the number of features"

    if standardize == 'norm':
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)
        matrix = (matrix - mean) / std
    elif standardize == 'center':
        mean = np.mean(matrix, axis=0)
        std = None
        matrix = matrix - mean
    else:
        print("No standardization applied, using original data.")
        mean = None
        std = None

    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    pca_v = Vt[:rank, :]  #  (rank, n_features)
    transformed = np.dot(matrix, pca_v.T)  #  (N, rank)

    pca_info = {
        'transformed': transformed,
        'pca_v': pca_v,
        'mean': mean,
        'std': std,
        'rank': rank,
    }
    return pca_info


def pca_inverse_transform_numpy(pca_info: Dict[str, np.ndarray]) -> np.ndarray:
    transformed = pca_info['transformed']
    pca_v = pca_info['pca_v']
    mean = pca_info['mean'] if 'mean' in pca_info else None
    std = pca_info['std'] if 'std' in pca_info else None

    matrix = np.dot(transformed, pca_v)  # (N, n_features)
    if mean is not None and std is not None:
        matrix = matrix * std + mean
    elif mean is not None:
        matrix = matrix + mean
    return matrix


def pca_transform_torch(matrix: Tensor, 
                rank: int = 21,
                standardize: Optional[Literal['norm', 'center']] ='norm') -> Dict[str, Tensor]:
    """Perform PCA on a 2D tensor (N, D) and return the transformed data along with PCA components.
    
    Args:
        matrix (Tensor): Input tensor of shape (N, D) where N is the number of samples and D is the number of features.
        rank (int): Number of principal components to retain.
        standardize (Optional[Literal['norm', 'center']]): If 'norm', standardize the data to have zero mean and unit variance; if 'center', center the data to have zero mean; if None, do not standardize.
    Returns:
        Dict[str, Tensor]: A dictionary containing:
            - 'transformed': The PCA transformed data of shape (N, rank).
            - 'pca_v': The PCA components of shape (rank, D).
            - 'mean': The mean of the original data if standardization was applied, otherwise None.
            - 'std': The standard deviation of the original data if standardization was applied, otherwise None.
            - 'rank': The number of principal components retained.
    """
    assert matrix.ndim == 2, "Input matrix must be 2D (N, D)"
    assert rank > 0, "Rank must be a positive integer"
    assert rank <= matrix.shape[1], "Rank must be less than or equal to the number of features"

    if standardize=='norm':
        mean = matrix.mean(dim=0) 
        std = matrix.std(dim=0)
        matrix = (matrix - mean) / std
    elif standardize=='center':
        mean = matrix.mean(dim=0) 
        std = None
        matrix = matrix - mean
    else:
        print("No standardization applied, using original data.")
        mean = None
        std = None

    U, S, Vt = torch.linalg.svd(matrix, full_matrices=False)

    pca_v = Vt[:rank, :]  # (rank, n_features)
    
    transformed = matrix @ pca_v.T  # (N, rank)

    
    pca_info = {
        'transformed': transformed,
        'pca_v': pca_v,
        'mean': mean,
        'std': std,
        'rank': rank
    }
    return pca_info

def pca_inverse_transform_torch(pca_info: dict) -> Tensor:    

    transformed = pca_info['transformed']
    pca_v = pca_info['pca_v']
    mean = pca_info['mean'] if 'mean' in pca_info else None
    std = pca_info['std'] if 'std' in pca_info else None
    
    matrix = transformed @ pca_v  # (N, n_features)
    if mean is not None and std is not None:
        matrix = matrix * std + mean
    elif mean is not None:
        matrix = matrix + mean
    else:
        matrix = matrix 
    return matrix
