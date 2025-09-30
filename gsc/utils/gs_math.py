'''
Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
Date: 2025-07-06 18:28:46
LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
LastEditTime: 2025-09-26 19:44:37
FilePath: /VGSC/vgsc/utils/gs_math.py
Description: Mathematical utilities for Gaussian Splatting, including RGB to SH conversion, quaternion normalization, depth to points and normals conversion, and more.

Copyright (c) 2025 by Zhiwei Zhu (zhuzhiwei21@zju.edu.cn), All Rights Reserved. 
'''

import math
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np

def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def normalized_quat_to_rotmat(quats: Tensor) -> Tensor:
    """Convert normalized quaternion to rotation matrix.

    Args:
        quats: Normalized quaternion in wxyz convension. (..., 4)

    Returns:
        Rotation matrix (..., 3, 3)
    """
    assert quats.shape[-1] == 4, quats.shape
    w, x, y, z = torch.unbind(quats, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return mat.reshape(quats.shape[:-1] + (3, 3))

def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)



def log_transform(x):
    if isinstance(x, Tensor):
        return torch.sign(x) * torch.log1p(torch.abs(x))
    elif isinstance(x, np.ndarray):
        return np.sign(x) * np.log1p(np.abs(x))
    else:
        raise TypeError(f"Unsupported type for log_transform: {type(x)}")
    

def inverse_log_transform(y):
    if isinstance(y, Tensor):
        return torch.sign(y) * torch.expm1(torch.abs(y))
    elif isinstance(y, np.ndarray):
        return np.sign(y) * np.expm1(np.abs(y)) 
    else:
        raise TypeError(f"Unsupported type for inverse_log_transform: {type(y)}")


def depth_to_points(
    depths: Tensor, camtoworlds: Tensor, Ks: Tensor, z_depth: bool = True
) -> Tensor:
    """Convert depth maps to 3D points

    Args:
        depths: Depth maps [..., H, W, 1]
        camtoworlds: Camera-to-world transformation matrices [..., 4, 4]
        Ks: Camera intrinsics [..., 3, 3]
        z_depth: Whether the depth is in z-depth (True) or ray depth (False)

    Returns:
        points: 3D points in the world coordinate system [..., H, W, 3]
    """
    assert depths.shape[-1] == 1, f"Invalid depth shape: {depths.shape}"
    assert camtoworlds.shape[-2:] == (
        4,
        4,
    ), f"Invalid viewmats shape: {camtoworlds.shape}"
    assert Ks.shape[-2:] == (3, 3), f"Invalid Ks shape: {Ks.shape}"
    assert (
        depths.shape[:-3] == camtoworlds.shape[:-2] == Ks.shape[:-2]
    ), f"Shape mismatch! depths: {depths.shape}, viewmats: {camtoworlds.shape}, Ks: {Ks.shape}"

    device = depths.device
    height, width = depths.shape[-3:-1]

    x, y = torch.meshgrid(
        torch.arange(width, device=device),
        torch.arange(height, device=device),
        indexing="xy",
    )  # [H, W]

    fx = Ks[..., 0, 0]  # [...]
    fy = Ks[..., 1, 1]  # [...]
    cx = Ks[..., 0, 2]  # [...]
    cy = Ks[..., 1, 2]  # [...]

    # camera directions in camera coordinates
    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cx[..., None, None] + 0.5) / fx[..., None, None],
                (y - cy[..., None, None] + 0.5) / fy[..., None, None],
            ],
            dim=-1,
        ),
        (0, 1),
        value=1.0,
    )  # [..., H, W, 3]

    # ray directions in world coordinates
    directions = torch.einsum(
        "...ij,...hwj->...hwi", camtoworlds[..., :3, :3], camera_dirs
    )  # [..., H, W, 3]
    origins = camtoworlds[..., :3, -1]  # [..., 3]

    if not z_depth:
        directions = F.normalize(directions, dim=-1)

    points = origins[..., None, None, :] + depths * directions
    return points


def depth_to_normal(
    depths: Tensor, camtoworlds: Tensor, Ks: Tensor, z_depth: bool = True
) -> Tensor:
    """Convert depth maps to surface normals

    Args:
        depths: Depth maps [..., H, W, 1]
        camtoworlds: Camera-to-world transformation matrices [..., 4, 4]
        Ks: Camera intrinsics [..., 3, 3]
        z_depth: Whether the depth is in z-depth (True) or ray depth (False)

    Returns:
        normals: Surface normals in the world coordinate system [..., H, W, 3]
    """
    points = depth_to_points(depths, camtoworlds, Ks, z_depth=z_depth)  # [..., H, W, 3]
    dx = torch.cat(
        [points[..., 2:, 1:-1, :] - points[..., :-2, 1:-1, :]], dim=-3
    )  # [..., H-2, W-2, 3]
    dy = torch.cat(
        [points[..., 1:-1, 2:, :] - points[..., 1:-1, :-2, :]], dim=-2
    )  # [..., H-2, W-2, 3]
    normals = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)  # [..., H-2, W-2, 3]
    normals = F.pad(normals, (0, 0, 1, 1, 1, 1), value=0.0)  # [..., H, W, 3]
    return normals


def get_projection_matrix(znear, zfar, fovX, fovY, device="cuda"):
    """Create OpenGL-style projection matrix"""
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, device=device)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def quaternion_normalize(quats: Tensor) -> Tensor:
    """
    Normalize the quaternion to ensure it is a unit quaternion.
    Args:
        quats: Quaternion in wxyz convention (..., 4)
    Returns:
        quats: Normalized quaternion in wxyz convention (..., 4)
    """
    return F.normalize(quats, dim=-1)

def quaternion_positive_w(quats: Tensor) -> Tensor:
    """
    Ensure the quaternion has a positive w component.
    Args:
        quats: Quaternion in wxyz convention (..., 4)
    Returns:
        quats: Quaternion with positive w component (..., 4)
    """
    w = quats[..., 0]
    mask = w < 0
    quats[mask] *= -1
    return quats

def quaternion_to_euler(quats: Tensor) -> Tensor:
    """
    quats: (N, 4) tensor, where each row is (w, x, y, z)
    return: (N, 3) tensor, where each row is (roll, pitch, yaw)
    """
    quats = quaternion_normalize(quats)
    w, x, y, z = quats[..., 0], quats[..., 1], quats[..., 2], quats[..., 3]

    # Roll (x-axis)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis)
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * (torch.pi / 2),
        torch.asin(sinp)
    )

    # Yaw (z-axis)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)

def euler_to_quaternion(euler_angles: Tensor) -> Tensor:
    """
    Convert Euler angles (XYZ order: roll, pitch, yaw) to quaternion.
    Args:
        euler_angles: Tensor of shape (..., 3) in radians: (roll, pitch, yaw)
    Returns:
        quats: Tensor of shape (..., 4) in wxyz order
    """
    roll = euler_angles[..., 0] / 2  # X
    pitch = euler_angles[..., 1] / 2  # Y
    yaw = euler_angles[..., 2] / 2  # Z

    cr = torch.cos(roll)
    sr = torch.sin(roll)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack([w, x, y, z], dim=-1)


def quaternion_to_rodrigues(quats: Tensor) -> Tensor:
    """
    Convert quaternion to Rodrigues vector.
    Args:
        quats: Quaternion in wxyz convention (..., 4)
    Returns:
        rod: Rodrigues vector (..., 3)
    """
    quats = quaternion_normalize(quats)
    w = quats[..., 0:1]
    xyz = quats[..., 1:]

    theta = 2.0 * torch.acos(torch.clamp(w, -1.0, 1.0))  # Clamp to avoid NaNs
    sin_half_theta = torch.sqrt(1.0 - w ** 2)

    # Handle division safely
    small_angle_mask = sin_half_theta < 1e-6
    u = torch.where(small_angle_mask, xyz, xyz / (sin_half_theta + 1e-12))
    rod = u * theta

    return rod


def rodrigues_to_quaternion(rod: Tensor) -> Tensor:
    """
    Convert Rodrigues vector to quaternion.
    Args:
        rod: Rodrigues vector (..., 3)
    Returns:
        quats: Quaternion in wxyz convention (..., 4)
    """
    theta = torch.norm(rod, dim=-1, keepdim=True)
    half_theta = theta / 2.0
    sin_half_theta = torch.sin(half_theta)
    
    w = torch.cos(half_theta)
    v = rod * (sin_half_theta / (theta + 1e-12))  # Avoid division by zero
    
    quats = torch.cat([w, v], dim=-1)
    return quaternion_normalize(quats)  


if __name__ == "__main__":
    # Example usage
    # Example quaternion [ 0.6685, -0.6618, -0.2519, -0.2272]
    quats = torch.tensor([0.8788, -0.1767,  0.4290,  0.1113])  
    print("Original quaternion:", quats)
    quats = quaternion_normalize(quats)
    print("Normalized quaternion:", quats)

    rod = quaternion_to_rodrigues(quats)
    print("Rodrigues vector:", rod)
    quat_from_rod = rodrigues_to_quaternion(rod)
    print("Quaternion from Rodrigues vector:", quat_from_rod)
    
    euler = quaternion_to_euler(quats)
    print("Euler angles from quaternion:", euler)
    quat_from_euler = euler_to_quaternion(euler)
    print("Euler angles from Rodrigues vector:", quat_from_euler)
