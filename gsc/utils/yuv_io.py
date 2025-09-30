'''
Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
Date: 2025-07-06 18:28:46
LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
LastEditTime: 2025-08-12 09:45:31
FilePath: /VGSC/vgsc/gs_yuv_io.py
Description: Handles saving and loading YUV files with metadata, including chroma subsampling and upsampling methods.

Copyright (c) 2025 by Zhiwei Zhu (zhuzhiwei21@zju.edu.cn), All Rights Reserved. 
'''
import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Dict, Tuple
from gsc.quantize import quantize_torch
from gsc.config import PIXEL_FORMATS, SUBSAMPLING_METHODS, UPSAMPLING_METHODS
from gsc import get_dtype_for_bitdepth, get_logger
logger = get_logger("YUVDataHandler")

@dataclass
class YUVDataHandler:
    """
    Handles saving tensor data to YUV files with metadata, and loading it back.

    This class encapsulates the logic for different attribute types, such as standard
    quantization, 16-bit splitting, or multi-component handling.
    """
    yuv_dir: Path
    chroma_sub_method: SUBSAMPLING_METHODS = field(default="bicubic")
    chroma_up_method: UPSAMPLING_METHODS = field(default="bicubic")

    def __post_init__(self):
        """Create the directory after initialization and set up method maps."""
        # Ensure the YUV directory exists
        self.yuv_dir = Path(self.yuv_dir)
        if not self.yuv_dir.exists():
            logger.info(f"Error: YUV directory does not exist: {self.yuv_dir}.")
            raise FileNotFoundError(f"YUV directory does not exist: {self.yuv_dir}.")

    #region Main Public Methods
    def _get_path(self, video_name: str, width: int, height: int, 
                   bit_depth: int, pix_fmt: str,
                   prefix: str = "", suffix: str = "",                
                   ) -> Path:
        """Generates standard file paths for an attribute component."""
        bit_suffix = f"{bit_depth}le" if bit_depth > 8 else ""
        base_file_name = f"{video_name}_{width}x{height}_{pix_fmt}{bit_suffix}"
        yuv_path = self.yuv_dir / f"{prefix}{base_file_name}{suffix}.yuv"
        return yuv_path
    
    def save_all_to_yuv(self, splats_videos : Dict[str, Tensor],
                        bit_depth_dict : Dict[str, int],
                        pix_fmt_dict: Dict[str, str],
                        prefix: str = "",
                        suffix: str = "") -> None:
        """Saves all attributes in `splats_video_data` to their respective YUV files."""
        for video_name, video in splats_videos.items():
            pix_fmt = pix_fmt_dict.get(video_name, None)
            bit_depth = bit_depth_dict.get(video_name, None) 
            if pix_fmt is None:
                raise ValueError(f"Pixel format for '{video_name}' is not specified in pix_fmt_dict.")
            if bit_depth is None:
                raise ValueError(f"Bit depth for '{video_name}' is not specified in bit_depth_dict.")
            height, width = video.shape[1], video.shape[2]
            
            yuv_path = self._get_path(video_name, width, height, 
                                          bit_depth, pix_fmt, 
                                          prefix=prefix, suffix=suffix)
            logger.info(f"Saving '{video_name}' to YUV file: {yuv_path}")
            save_tensor_to_yuv(video, str(yuv_path), bit_depth, pix_fmt, self.chroma_sub_method)

            
    def load_all_from_yuv(self, meta: Dict[str, Any],
                          prefix: str = "",
                          suffix: str = "") -> Dict[str, Tensor]:
        """Loads attributes specified by `video_names` from decoded YUV files."""
        import os
        loaded_data = {}
        width = meta.get('width', None)
        height = meta.get('height', None)
        pix_fmt_dict = meta.get('pix_fmt_dict', None)
        bit_depth_dict = meta.get('bit_depth_dict', None)
        missing = [name for name, value in [
            ('pix_fmt_dict', pix_fmt_dict),
            ('bit_depth_dict', bit_depth_dict),
            ('width', width),
            ('height', height),
        ] if value is None]

        if missing:
            raise ValueError(f"Metadata missing required fields: {', '.join(missing)}")
        for video_name, pix_fmt in pix_fmt_dict.items():
            bit_depth = bit_depth_dict[video_name]
            decoded_yuv_path = self._get_path(video_name, width, height,
                                                bit_depth, pix_fmt,
                                                prefix, suffix)
            video = load_yuv_to_tensor(str(decoded_yuv_path), 
                                       height, width, 
                                       bit_depth, bit_depth, 
                                       pix_fmt, self.chroma_up_method)
            loaded_data[video_name] = video
        return loaded_data


def save_tensor_to_yuv(
    video: Tensor,
    yuv_file_path: str,
    bitdepth: int = 8,
    pix_fmt: PIXEL_FORMATS = None,
    chroma_sub_method: SUBSAMPLING_METHODS = "bicubic"
) -> None:
    """
    Save a tensor to a YUV file in planar format with a specified bitdepth.

    Args:
        video (torch.Tensor): Video tensor in the shape of [T, H, W, 3].
                              If float, values are assumed to be in the [0, 1] range.
                              If integer, values are assumed to be in the correct range.
                              For yuv420p, U and V planes are expected at full resolution;
                              they will be subsampled during the save process.
        yuv_file_path (str): Path to save the output YUV file.
        bitdepth (int): The target bitdepth for the output file (e.g., 8 or 10).
        pix_fmt (str): Chroma subsampling format.
    """
    if not isinstance(video, Tensor):
        raise TypeError(f"Input video must be a torch.Tensor, but got {type(video)}")

    logger.info(
        f"Attempting to save video to '{yuv_file_path}' with format {pix_fmt} "
        f"and {bitdepth}-bit depth."
    )

    dtypes = get_dtype_for_bitdepth(bitdepth)
    target_torch_dtype = dtypes["torch"]
    target_numpy_dtype = dtypes["numpy"]

    # --- Tensor Preparation ---
    # Ensure tensor is on CPU and has the correct integer data type.
    video = video.cpu()

    if video.dtype != target_torch_dtype:
        logger.warning(f"Input tensor dtype is {video.dtype}, but target is {target_torch_dtype}. Will quantize the input tensor and convert it to the target dtype.")
        min_val = torch.min(video)
        range_val = torch.max(video) - min_val
        video = quantize_torch(video, bitdepth, min_val=min_val, range_val=range_val)

    n_frames, H, W, _ = video.shape

    # --- File Writing ---
    try:
        with open(yuv_file_path, 'wb') as f:
            for t in range(n_frames):
                frame = video[t]  # Shape: (H, W, C)
                channels = frame.shape[-1]
                y_plane = frame[..., 0]

                if pix_fmt == "yuv400p":
                    if channels == 3:
                        logger.warning(
                            f"Frame {t}: 3 channels detected, but pix_fmt is 'yuv400p'. "
                            "Only the Y channel will be written."
                        )
                    f.write(y_plane.numpy().astype(target_numpy_dtype).tobytes())
                    continue

                if pix_fmt == "yuv420p":
                    if channels == 3:
                        y_plane, u_plane, v_plane = chroma_subsample_frame(frame, chroma_sub_method)
                    else:
                        logger.warning(
                            f"Frame {t}: {channels} channels detected, expected 3 for 'yuv420p'. "
                            "Using mid-depth padding for U and V planes."
                        )
                        u_plane = v_plane = torch.full(
                            (H // 2, W // 2),
                            1 << (bitdepth - 1),
                            dtype=target_torch_dtype
                        )

                elif pix_fmt == "yuv444p":
                    if channels == 3:
                        u_plane = frame[..., 1]
                        v_plane = frame[..., 2]
                    else:
                        logger.warning(
                            f"Frame {t}: {channels} channels detected, expected 3 for 'yuv444p'. "
                            "Using mid-depth padding for U and V planes."
                        )
                        u_plane = v_plane = torch.full(
                            (H, W),
                            1 << (bitdepth - 1),
                            dtype=target_torch_dtype
                        )

                else:
                    raise ValueError(f"Unsupported pixel format: {pix_fmt}")

                # Write planes
                for plane in (y_plane, u_plane, v_plane):
                    f.write(plane.numpy().astype(target_numpy_dtype).tobytes())

        logger.info(f"Successfully saved YUV file to: {yuv_file_path}")

    except IOError as e:
        logger.error(f"Failed to write to file {yuv_file_path}: {e}")
        raise



def load_yuv_to_tensor(
    yuv_file: str,
    height: int,
    width: int,
    input_bitdepth: int = 8,
    output_bitdepth: int = 8,
    pix_fmt: PIXEL_FORMATS = None,
    chroma_up_method: UPSAMPLING_METHODS = "bicubic"
) -> Tensor:
    """
    Load a raw YUV file into a PyTorch tensor with specified input and output bitdepths.

    Reads planar YUV data, reconstructs it into a tensor of shape [T, H, W, 3],
    and handles bitdepth conversion via bit-shifting.

    Args:
        yuv_file (str): Path to the input raw YUV file.
        height (int): Height of the video frames.
        width (int): Width of the video frames.
        pix_fmt (Literal["yuv420p", "yuv444p", "yuv400p"]): Pixel format of the YUV file.
        input_bitdepth (int): The bitdepth of the source YUV file (e.g., 8 or 10).
        output_bitdepth (int): The desired bitdepth for the output tensor. If
                               output_bitdepth < input_bitdepth, a right bit-shift
                               is applied.

    Returns:
        Tensor: A tensor containing the video data in shape [T, H, W, 3] with an
                integer dtype corresponding to `output_bitdepth`.

    Raises:
        FileNotFoundError: If the yuv_file does not exist.
        ValueError: If file size is inconsistent, or bitdepths are unsupported.
    """
    if not os.path.exists(yuv_file):
        raise FileNotFoundError(f"YUV file not found: {yuv_file}")

    logger.info(
        f"Loading YUV file '{os.path.basename(yuv_file)}' with format {pix_fmt}, "
        f"{input_bitdepth}-bit input -> {output_bitdepth}-bit output."
    )

    # --- Parameter and File Validation ---
    input_dtypes = get_dtype_for_bitdepth(input_bitdepth)
    output_dtypes = get_dtype_for_bitdepth(output_bitdepth)
    
    read_dtype = input_dtypes["torch"]
    pix_bytes = input_dtypes["bytes"]
    
    file_size = os.path.getsize(yuv_file)

    if pix_fmt == "yuv444p":
        bytes_per_frame = pix_bytes * height * width * 3
        chroma_h, chroma_w = height, width
    elif pix_fmt == "yuv420p":
        if height % 2 != 0 or width % 2 != 0:
            logger.warning(f"Height ({height}) or Width ({width}) is odd for YUV420p.")
        bytes_per_frame = pix_bytes * int(height * width * 1.5)
        chroma_h, chroma_w = height // 2, width // 2
    elif pix_fmt == "yuv400p":
        bytes_per_frame = pix_bytes * height * width
        chroma_h, chroma_w = 0, 0
    else:
        raise ValueError(f"Unsupported pix_fmt: {pix_fmt}")

    if bytes_per_frame == 0:
        raise ValueError("Calculated bytes_per_frame is zero. Check dimensions.")
    if file_size % bytes_per_frame != 0:
        raise ValueError(
            f"File size {file_size} is not a multiple of frame size {bytes_per_frame} "
            f"for H={height}, W={width}, pix_fmt={pix_fmt}, bitdepth={input_bitdepth}."
        )

    num_frames = file_size // bytes_per_frame
    if num_frames == 0:
        logger.warning(f"YUV file is empty or too small for one frame: {yuv_file}")
        return torch.empty((0, height, width, 3), dtype=output_dtypes["torch"])

    logger.info(f"Reading {num_frames} frames from file.")

    # --- Reading and Reconstructing Frames ---
    y_plane_size = pix_bytes * height * width
    chroma_plane_size = pix_bytes * chroma_h * chroma_w
    all_frames = []

    try:
        with open(yuv_file, 'rb') as f:
            for i in range(num_frames):
                y_bytes = f.read(y_plane_size)
                y_plane = torch.frombuffer(y_bytes, dtype=read_dtype).reshape(height, width)

                if pix_fmt == "yuv400p":
                    frame_tensor = y_plane.unsqueeze(-1)  # Shape: (H, W, 1)
                    all_frames.append(frame_tensor)
                    continue
                else:
                    u_bytes = f.read(chroma_plane_size)
                    u_plane = torch.frombuffer(u_bytes, dtype=read_dtype).reshape(chroma_h, chroma_w)
                    
                    v_bytes = f.read(chroma_plane_size)
                    v_plane = torch.frombuffer(v_bytes, dtype=read_dtype).reshape(chroma_h, chroma_w)

                    # Upsample chroma for yuv420p
                    if pix_fmt == "yuv420p":
                        # Convert to float for interpolation, then back to original int type
                        u_plane, v_plane = chroma_upsample_frame(
                            y_plane, u_plane, v_plane, method=chroma_up_method,
                            bit_depth=input_bitdepth
                        )

                # Stack planes to form a frame [H, W, 3]
                frame_tensor = torch.stack([y_plane, u_plane, v_plane], dim=-1)
                all_frames.append(frame_tensor)
    
    except (IOError, RuntimeError) as e:
        logger.error(f"Failed to read or process file {yuv_file}: {e}")
        raise

    video_tensor = torch.stack(all_frames, dim=0)

    # --- Bitdepth Conversion ---
    if input_bitdepth != output_bitdepth:
        logger.info(f"Converting from {input_bitdepth}-bit to {output_bitdepth}-bit.")
        if output_bitdepth < input_bitdepth:
            shift = input_bitdepth - output_bitdepth
            video_tensor = video_tensor >> shift
        else: # output_bitdepth > input_bitdepth
            shift = output_bitdepth - input_bitdepth
            video_tensor = video_tensor << shift

    # Cast to the final target dtype
    return video_tensor.to(output_dtypes["torch"])

def _create_gaussian_kernel(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Creates a 2D Gaussian kernel for filtering."""
    center = kernel_size // 2
    x = torch.arange(kernel_size, device=device, dtype=torch.float32) - center
    x_grid, y_grid = torch.meshgrid(x, x, indexing='ij')
    kernel = torch.exp(-(x_grid.pow(2) + y_grid.pow(2)) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    # Reshape for conv2d: [out_channels, in_channels, H, W]
    return kernel.view(1, 1, kernel_size, kernel_size)


def chroma_subsample_frame(
    frame_444: torch.Tensor,
    method: SUBSAMPLING_METHODS = "bicubic"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs chroma downsampling on a single YUV 4:4:4 frame to produce 4:2:0 planes.

    This function takes a single frame of shape [H, W, 3] and returns the full-resolution
    Y plane, and the half-resolution U and V planes using one of several methods.

    Args:
        frame_444 (torch.Tensor): Input frame tensor with shape [H, W, 3],
                                  representing YUV 4:4:4 data. Can be uint8 or float.
        method (SUBSAMPLING_METHODS): The algorithm to use for downsampling chroma.
            - "skip": Fastest, but prone to aliasing. Simply picks every
                                 second pixel.
            - "average_pool": Recommended. Averages 2x2 blocks, providing good
                              aliasing prevention and speed.
            - "gaussian_blur": High-quality. Applies a Gaussian blur before
                               subsampling to reduce aliasing smoothly.
            - "bicubic": High-quality. Uses bicubic interpolation to resize
                                the chroma planes.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - y_plane (torch.Tensor): Y plane with shape [H_out, W_out].
            - u_plane_420 (torch.Tensor): Downsampled U plane with shape [H_out/2, W_out/2].
            - v_plane_420 (torch.Tensor): Downsampled V plane with shape [H_out/2, W_out/2].
        H_out and W_out are the original dimensions cropped to the nearest even number.
    """
    # --- 1. Input Validation and Preparation ---
    if not frame_444.dim() == 3 or not frame_444.shape[2] == 3:
        raise ValueError("Input frame must be a tensor with shape [H, W, 3].")

    original_dtype = frame_444.dtype
    h, w, _ = frame_444.shape

    # Crop to even dimensions, required for 4:2:0 subsampling
    h_out, w_out = h - (h % 2), w - (w % 2)
    if h != h_out or w != w_out:
        logger.warning(f"Input frame cropped from ({h}, {w}) to ({h_out}, {w_out}) for even dimensions.")
        frame_444 = frame_444[:h_out, :w_out, :]
    
    # Separate Y, U, V planes. Y plane is not processed further.
    y_plane = frame_444[:, :, 0]
    u_plane_444 = frame_444[:, :, 1]
    v_plane_444 = frame_444[:, :, 2]
    
    # --- 2. Chroma Downsampling Logic ---
    logger.info(f"Applying '{method}' method for chroma subsampling.")

    if method == "skip":
        u_plane_420 = u_plane_444[::2, ::2]
        v_plane_420 = v_plane_444[::2, ::2]
    else:
        # For advanced methods, convert to float and add batch/channel dims for PyTorch functions
        u_ch_first = u_plane_444.float().unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
        v_ch_first = v_plane_444.float().unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]

        if method == "average_pool":
            u_down = F.avg_pool2d(u_ch_first, kernel_size=2, stride=2)
            v_down = F.avg_pool2d(v_ch_first, kernel_size=2, stride=2)
        
        elif method == "bicubic":
            u_down = F.interpolate(u_ch_first, scale_factor=0.5, mode='bicubic', align_corners=False)
            v_down = F.interpolate(v_ch_first, scale_factor=0.5, mode='bicubic', align_corners=False)
        
        elif method == "gaussian_blur":
            kernel = _create_gaussian_kernel(kernel_size=5, sigma=1.0, device=frame_444.device)
            # Pad to maintain size, then convolve
            u_filtered = F.conv2d(u_ch_first, kernel, padding='same')
            v_filtered = F.conv2d(v_ch_first, kernel, padding='same')
            # Subsample after filtering
            u_down = u_filtered[:, :, ::2, ::2]
            v_down = v_filtered[:, :, ::2, ::2]
        else:
            # This path should not be reachable due to Literal typing but is good practice
            raise ValueError(f"Unsupported subsampling method: {method}")

        # Remove batch/channel dimensions
        u_plane_420 = u_down.squeeze(0).squeeze(0)
        v_plane_420 = v_down.squeeze(0).squeeze(0)

    # --- 3. Final Type Conversion ---
    if not original_dtype.is_floating_point:
        y_plane = y_plane.round().to(original_dtype)
        u_plane_420 = u_plane_420.round().clamp(0, 255).to(original_dtype)
        v_plane_420 = v_plane_420.round().clamp(0, 255).to(original_dtype)

    return y_plane, u_plane_420, v_plane_420


def chroma_upsample_frame(
    y_plane: torch.Tensor,
    u_plane_420: torch.Tensor,
    v_plane_420: torch.Tensor,
    method: UPSAMPLING_METHODS = "bicubic",
    bit_depth: int = 10
) -> torch.Tensor:
    """
    Performs chroma upsampling on YUV 4:2:0 planes to produce a single YUV 4:4:4 frame.

    This function takes a full-resolution Y plane and half-resolution U and V planes,
    and returns a single combined frame tensor of shape [H, W, 3].

    Args:
        y_plane (torch.Tensor): Full-resolution Y plane with shape [H, W].
        u_plane_420 (torch.Tensor): Half-resolution U plane with shape [H/2, W/2].
        v_plane_420 (torch.Tensor): Half-resolution V plane with shape [H/2, W/2].
        method (UPSAMPLING_METHODS): The algorithm to use for upsampling chroma.
            - "nearest": Fastest, but produces blocky artifacts.
            - "bilinear": Good balance of speed and quality, produces smooth results.
            - "bicubic": Recommended for high quality. Slower but better at
                         preserving details.

    Returns:
        torch.Tensor: A combined YUV 4:4:4 frame tensor with shape [H, W, 3].
    """
    # --- 1. Input Validation and Preparation ---
    if not (y_plane.dim() == 2 and u_plane_420.dim() == 2 and v_plane_420.dim() == 2):
        raise ValueError("Input planes must be 2D tensors.")

    h, w = y_plane.shape
    h_chroma, w_chroma = u_plane_420.shape

    if h != h_chroma * 2 or w != w_chroma * 2:
        raise ValueError(
            f"Luma dimensions ({h}, {w}) must be exactly double the chroma "
            f"dimensions ({h_chroma}, {w_chroma})."
        )

    original_dtype = y_plane.dtype
    
    # --- 2. Chroma Upsampling Logic ---
    logger.info(f"Applying '{method}' method for chroma upsampling.")
    
    # Prepare chroma planes for interpolation: add batch and channel dims
    u_to_interp = u_plane_420.float().unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H/2, W/2]
    v_to_interp = v_plane_420.float().unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H/2, W/2]

    # `align_corners=False` is generally recommended for image resizing
    u_plane_444 = F.interpolate(u_to_interp, size=(h, w), mode=method, align_corners=False)
    v_plane_444 = F.interpolate(v_to_interp, size=(h, w), mode=method, align_corners=False)
    
    # Remove batch and channel dimensions
    u_plane_444 = u_plane_444.squeeze(0).squeeze(0)
    v_plane_444 = v_plane_444.squeeze(0).squeeze(0)

    # --- 3. Final Type Conversion ---
    if not original_dtype.is_floating_point:
        # Clamp values to the valid range for uint8 before converting
        max_val = (1 << bit_depth) - 1
        u_plane_444 = u_plane_444.round().clamp(0, max_val).to(original_dtype)
        v_plane_444 = v_plane_444.round().clamp(0, max_val).to(original_dtype)

    return u_plane_444, v_plane_444