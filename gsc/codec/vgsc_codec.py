'''
Author: Zhiwei Zhu (zhuzhiwei21@zju.e:u.cn)
Date: 2025-07-06 18:28:46?
LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
LastEditTime: 2025-10-01 01:41:13
FilePath: /VGSC/gsc/codec/vgsc_codec.py
Description: T,e VgscC)dec class provides methods for encoding and decoding splat frames into video bitstreams.
Copyright (c) 2025 by Zhiwei Zhu (zhuzhiwei21@zju.edu.cn), All Rights Reserved. 
'''

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Literal, Tuple
from dataclasses import dataclass, field
import copy
import time
import torch
from torch import Tensor
from gsc import (
    get_dtype_for_bitdepth, 
    get_shN_sub_names, 
    resize_splats, 
    smart_look_up_dict, 
    splats_list_to_dict,
    splats_dict_to_list,
    )
from gsc.utils.file import force_make_dirs, safe_make_dirs, smart_save_meta, smart_load_meta
from gsc.config import (
    DEC_SUFFIX, CODEC_META_NAME,
    SUBSAMPLING_METHODS, UPSAMPLING_METHODS,
    DEFAULT_QP, DEFAULT_BIT_DEPTH  
    )
from gsc.utils.yuv_io import YUVDataHandler
from .video_codec import VideoCodec
from gsc import get_logger, bit_split
logger = get_logger("VgscCodec")


@dataclass
class VgscCodec:
    """
    VgscCodec is a class for compressing and decompressing splat frames into video bitstreams.

    This class takes a list of splat frames, reorganizes them into attribute videos,
    and then uses a YUVDataHandler to save them as raw YUV and a VideoCodec to 
    compress them into bitstreams.
    """ 
    # Experimental parameters

    # sort parameters
    sort_type: str = "plas"
    sort_with_shN: bool = True # PLAS sort with shN
    weight_dict: Dict[str, float]= None # Weight for each attribute in sorting, e.g., {"means": 1.0, "opacities": 0.5, ...}
    # chroma subsampling and upsampling methods
    chroma_sub_method: SUBSAMPLING_METHODS = "bicubic"  # Method for chroma subsampling
    chroma_up_method: UPSAMPLING_METHODS = "bicubic"  # Method for chroma upsampling
    
    # video codec parameters
    video_codec_type: str = "ffmpeg"  # or "hm"
    encoder_path: str = "ffmpeg"  # Path to the video encoder executable
    decoder_path: str = "ffmpeg"  # Path to the video decoder executable
    encode_config_path: Optional[str] = None  # Path to the encoding configuration file
    decode_config_path: Optional[str] = None  # Path to the decoding configuration file
    all_intra: bool = False  # If True, all frames are intra-coded
    gop_size: int = 16  # Group of Pictures size for compression
    qp_config: Dict[str, Any]= field(default_factory=lambda: copy.deepcopy(DEFAULT_QP))  # QP for each attribute, e.g., {"means": 4, "opacities": 4, ...
    bit_depth_config: Dict[str, Any]= field(default_factory=lambda: copy.deepcopy(DEFAULT_BIT_DEPTH))  # Bit depth for each attribute, e.g., {"means": 8, "opacities": 8, ...}
    pix_fmt_config: Optional[Dict[str, Dict[str, Any]]]= None  # Pixel format for each attribute
    means_no_split: bool = False  # If True, do not split means into lower and upper parts
       
            
    def _setup_directories(self, result_dir: Path) -> None:
        """Creates the working directories."""
        logger.info("Setting up working directories...")
        self.result_dir = Path(result_dir)
        self.compress_dir = self.result_dir / "compression"
        self.yuv_dir = self.result_dir / "yuv"
        
        safe_make_dirs(self.compress_dir)
        force_make_dirs(self.yuv_dir)
        logger.info("Working directories are ready.")

        
    def encode(self, splats_list: List[Dict[str, Tensor]], compress_dir: Path) -> None:
        """
        Encodes a list of splat frames into a compressed video bitstream.
        
        1. Reorganizes the splats list into a dictionary of attributes.
        2. Converts the attribute dictionary to a grid format.
        3. Saves the attribute videos to YUV format using YUVDataHandler.
        4. Encodes each YUV file to a bitstream using VideoCodec.
        
        Args:
            splats_list (List[Dict[str, Tensor]]): List of splat frames, where each frame is a dictionary of attributes.
            compress_dir (Path): Directory where the compressed files will be saved.
            gop_id (int): Group of Pictures ID for the compression process.
        Returns:
            None: The function saves the compressed files to the specified directory.  
        """
        self._setup_directories(Path(compress_dir).parent)
        # Check Everything is set up correctly
        if not splats_list:
            logger.error("Input splats_list is empty. Nothing to compress.")
            return

        logger.info("--- Starting Compression Process ---")

        # 1. Reorganize splats list into attribute videos
        logger.info("Reorganizing splats list into attribute dictionary.")
        attr_dict, n_sidelen = self._reorganize(splats_list)
        width = n_sidelen
        height = n_sidelen
          
        # 2. Convert the attribute dictionary to grid format
        logger.info("Converting attribute dictionary to grid format.")
        splats_videos = self._attribute_dict_to_grid(attr_dict, n_sidelen, n_sidelen)
        
        # 3. Setup YUV handler and save to YUV
        logger.info("Saving attribute videos to YUV format.")
        yuv_data_handler = YUVDataHandler(self.yuv_dir, chroma_sub_method=self.chroma_sub_method)
        self._build_attribute_configs(splats_videos)
        yuv_data_handler.save_all_to_yuv(splats_videos, 
                                         bit_depth_dict=self.attribute_configs['bit_depth_dict'], 
                                         pix_fmt_dict=self.attribute_configs['pix_fmt_dict'])
        self._save_metadata(self.compress_dir, 
                            prefix="",
                            width=width,
                            height=height,
                            bit_depth_dict=self.attribute_configs['bit_depth_dict'], 
                            pix_fmt_dict=self.attribute_configs['pix_fmt_dict'],)
           
        # 4. Encode each YUV to a bitstream
        logger.info("Encoding YUV files to bitstreams.")
        video_codec = VideoCodec(self.video_codec_type,
                                 encoder_path=self.encoder_path,
                                 decoder_path=self.decoder_path,
                                 encode_config_path=self.encode_config_path,
                                 decode_config_path=self.decode_config_path,)
        yuv_files = [f for f in self.yuv_dir.glob("*.yuv")]
        if not yuv_files:
            logger.error(f"No YUV files found in {self.yuv_dir}. Cannot compress.")
            return
        for yuv_path in yuv_files:
            yuv_name = yuv_path.stem
            bitstream_path = self.compress_dir / f"{yuv_name}.mp4"
            config_params = self._get_encode_configs(yuv_name, width, height, frame_num=len(splats_list))
            video_codec.encode(
                input_yuv=yuv_path,
                output_bitstream=bitstream_path,
                config_params=config_params
            )
        logger.info("✅ Compression Process Finished.")
         

    def decode(self, compress_dir: Path) -> List[Dict[str, Any]]:
        """
        Decodes a compressed video bitstream back into a list of splat frames.

        1. Decodes the bitstreams to YUV files using VideoCodec.
        2. Loads metadata and the YUV files using YUVDataHandler.
        3. Converts the loaded grid videos to an attribute dictionary.
        4. Deorganizes the attribute dictionary back to a list of splats.
       
        """
        logger.info("--- Starting Decompression Process ---")
        
        compress_dir = Path(compress_dir)
        if not compress_dir.exists():
            raise FileNotFoundError(f"Compression directory {self.compress_dir} does not exist. Please run video_encode first.")
        
        yuv_dir =  compress_dir.parent / "yuv"
        meta = self._load_metadata(compress_dir, prefix="")
        # 1. Decode bitstreams to YUV
        logger.info("Decoding bitstreams to YUV files.")
        video_codec = VideoCodec(self.video_codec_type,
                            encoder_path=self.encoder_path,
                            decoder_path=self.decoder_path,
                            encode_config_path=self.encode_config_path,
                            decode_config_path=self.decode_config_path,)
        bin_files = [f for f in compress_dir.glob("*.mp4")]
        if not bin_files:
            logger.error(f"No compressed files found in {compress_dir}. Cannot decompress.")
            return []
        for bin_path in bin_files:
            yuv_name = bin_path.stem
            yuv_path = yuv_dir / f"{yuv_name}{DEC_SUFFIX}.yuv"
            if not yuv_path.exists():
                video_codec.decode(input_bitstream=bin_path,
                                   output_yuv=yuv_path,
                                   config_params=self._get_decode_configs(yuv_name, meta))
            else:
                logger.info(f"YUV file already exists: {yuv_path}. Skipping decoding.")
                
        # 2. Load from YUV and metadata
        yuv_data_handler = YUVDataHandler(yuv_dir, chroma_up_method=self.chroma_up_method)
        rec_splats_videos = yuv_data_handler.load_all_from_yuv(meta, 
                                                               prefix="",
                                                               suffix=DEC_SUFFIX)
        
        # 3. Convert the loaded grid videos to attribute dictionary
        rec_attr_dict = self._grid_to_attribute_dict(rec_splats_videos, meta)   

        # 4. Deorganize back to splat list
        splats_list = self._deorganize(rec_attr_dict)
        logger.info("✅ Decompression Process Finished.")

        return splats_list
    
    def _resize_splats_list(self, splats_list: List[Dict[str, Tensor]], target_num: Optional[int]=None, block_size: Optional[int]=None, pad : bool=False) -> Tuple[List[Dict[str, Tensor]], int]:
        """Resize the number of splats in each frame to the target number of splats.
        
        Args:
            splats_list (List[Dict[str, Tensor]]): List of splat dictionaries, each containing attributes like means, quats, etc.
            target_num (Optional[int]): The target number of splats. If None, use the maximum number of splats in the list.
        Returns:
            List[Dict[str, Tensor]]: List of splat dictionaries with the number of splats resized to the target number.
        """
        if not splats_list:
            logger.error("Input splats_list is empty. Cannot resize splats.")
            return []
        
        if target_num is None:
            target_num = max(splat['means'].shape[0] for splat in splats_list)
        
        if pad:
            ### padding splats to a square grid
            n_sidelen = int(target_num**0.5 + 1)  # Calculate the side length of the square grid
            if n_sidelen % 8 != 0:
                n_sidelen = 8 * (n_sidelen // 8 + 1)  # Video codec requires the side length to be a multiple of 8
            if block_size is not None and n_sidelen % block_size != 0:
                n_sidelen = block_size * (n_sidelen // block_size + 1)
        else:
            ### crop splats to a square grid (recommended)
            n_sidelen = int(target_num**0.5) 
            if n_sidelen % 8 != 0:
                n_sidelen = 8 * (n_sidelen // 8)  # Video codec requires the side length to be a multiple of 8
            if block_size is not None and n_sidelen % block_size != 0:
                n_sidelen = block_size * (n_sidelen // block_size)
            
        target_num = n_sidelen * n_sidelen  # The target number of splats is the square of the side length
               
        resized_splats_list = [resize_splats(splat, target_num) for splat in splats_list]    
        return resized_splats_list, n_sidelen

    
    def _sort_splats(self, splats_to_be_sorted: Dict[str, Tensor]) -> Tensor:
        """Organize the list of splats into several sequences of attributs

        Args:
            splats_list (List[Dict]): List of splat dictionaries, each containing attributes like means, quats, etc.
            frame_id (int): The index of the frame to sort. Defaults to 0.
        Returns:
            Tensor: Sorted indices of the splats for the specified frame.

        """
        from gsc.map.gs_map import (
            sort_splats_plas,  
            sort_points_by_space_curve,
        )
        start_time = time.time()
        # Use the appropriate sorting function based on the sort_type
        if self.sort_type =='plas':
            sorted_indices = sort_splats_plas(splats_to_be_sorted, sort_with_shN=self.sort_with_shN)                                           
        else:
            sorted_indices = sort_points_by_space_curve(
                splats_to_be_sorted['means'], 
                curve_type=self.sort_type,)

        logger.info(f"Splats sorting ({self.sort_type}) took {time.time() - start_time:.2f} seconds.")
        return sorted_indices

    
    def _reorganize(self, splats_list: List[Dict]) -> Tuple[Dict[str, Tensor], int]:
        """Reorganize a list of splat dictionaries into a sequence of attributes.
        Args:
            splats_list (List[Dict]): List of splat dictionaries, each containing attributes like means, quats, etc.
        Returns:
            int: The side length of the padded square grid of Gaussians.
        """
        # resize the splats to a fixed number of splats
        splats_list, n_sidelen = self._resize_splats_list(splats_list)
        # splat list to sequence of attributes
        attr_dict = self._splats_list_to_attribute_dict(splats_list)
        if not self.all_intra: # random access
            logger.info("Using random access mode for compression.")
            gop_size = self.gop_size
            frame_num = len(splats_list)
            num_gop = (frame_num + gop_size - 1) // gop_size
            for gop_id in range(num_gop):
                gop_start_frame_id = gop_id * gop_size
                gop_end_frame_id = min(gop_start_frame_id + gop_size, frame_num)
                # sort the splats with the first frame index of the GOP
                sorted_indices = self._sort_splats(splats_list[gop_start_frame_id])
                # use indices to sort the sequences of attributes
                for fr_id in range(gop_start_frame_id, gop_end_frame_id):
                    for attr_name, attr_data in attr_dict.items():
                        attr_dict[attr_name][fr_id] = attr_data[fr_id][sorted_indices, ...]
        else: # all intra
            logger.info("Using all intra mode for compression.")
            # sort the splats with each frame index
            for fr_id, _ in enumerate(splats_list):
                sorted_indices = self._sort_splats(splats_list[fr_id])
                for attr_name, attr_data in attr_dict.items():
                    attr_dict[attr_name][fr_id] = attr_data[fr_id][sorted_indices, ...]

        return attr_dict, n_sidelen
    
    def _deorganize(self, attr_dict: Dict[str, Tensor]) -> List[Dict]:
        """_summary_

        Args:
            attr_dict (Dict[str, Tensor]): _description_

        Returns:
            List[Dict]: _description_
        """
        base_attr_dict = {}
        base_attr_dict['means'] = attr_dict.get('means_l', None) # [T, N, 3]
        if base_attr_dict['means'] is None:
            base_attr_dict['means'] = attr_dict.get('means', None)
        if base_attr_dict['means'] is None:
            raise ValueError("Attribute 'means' is missing in the attribute dictionary. Cannot deorganize splats.")
        base_attr_dict['scales'] = attr_dict['scales'] # [T, N, 3]
        base_attr_dict['sh0'] = attr_dict['sh0'].unsqueeze(dim=2) # [T, N, 1, 3]
        base_attr_dict['opacities'] = attr_dict['opacities'][..., 0]  # [T, N]  
        if 'quats_w' in attr_dict:
            # If quaternions are split into w and xyz components, we need to combine them
            base_attr_dict['quats'] = torch.cat([attr_dict['quats_w'][..., 0].unsqueeze(dim=2), attr_dict['quats_xyz']], dim=-1) # [T, N, 4]
        else:
            base_attr_dict['quats'] = attr_dict['quats_xyz']  # [T, N, 3] 
        shN_names = get_shN_sub_names(max_degree=3)
        exist_sh_names = [key for key in attr_dict.keys() if key.startswith('shN_')]
        shN_names = shN_names[:len(exist_sh_names)]  # Ensure we only use available names
        sh_videos = []
        for shN_name in shN_names:
            sh_videos.append(attr_dict[shN_name])
        
        # Stack along a new dimension to form [T, N, C_idx, Channels]
        # We add a new dimension at index 3 for stacking. 
        base_attr_dict['shN'] = torch.stack(sh_videos, dim=2) # [T, N, C_idx, 3]

        splats_list = splats_dict_to_list(base_attr_dict)
        
        return splats_list
  
    
    def _splats_list_to_attribute_dict(self, splats_list: List[Dict]) -> Dict[str, Tensor]:
        """Convert a list of splat dictionaries into a sequence of attributes.
        Args:
            splats_list (List[Dict]): List of splat dictionaries, each containing attributes like means, quats, etc.
        Returns:
            Dict[str, Tensor]: A dictionary where keys are attribute names and values are tensors of attributes
            stacked across the frames.
        """
        base_attr_dict = splats_list_to_dict(splats_list)

        attr_dict = {}
        for attr_name, attr_seq in base_attr_dict.items():
            # Organize the attributes to different groups      
            if attr_name == "quats": 
                q_dim = attr_seq.shape[-1] # [T, N, 3] or [T, N, 4]
                if q_dim == 3:
                    # If the dimension is 3, we can store the parameters directly 
                    attr_dict[f'{attr_name}_xyz'] = attr_seq
                elif q_dim == 4:
                    # If the dimension is 4, we need to split into w and xyz components
                    video_w = attr_seq[..., 0:1]
                    video_xyz = attr_seq[..., 1:4]
                    attr_dict[f'{attr_name}_w'] = video_w
                    attr_dict[f'{attr_name}_xyz'] = video_xyz
                else:
                    raise ValueError(f"Unsupported quaternion dimension {q_dim}. Expected 3 or 4.")                
            elif attr_name == "shN":        
                shN_names = get_shN_sub_names(max_degree=3)
                if attr_seq.shape[-2] != len(shN_names):  # [T, N, x_dim, 3]
                    logger.warning(f"shN_video shape {attr_seq.shape} does not match expected number of spherical harmonics coefficients {len(shN_names)}.")
                    shN_names = shN_names[:attr_seq.shape[-2]]  # Adjust to the actual number of coefficients
                for i, shN_name in enumerate(shN_names):
                    attr_dict[shN_name] = attr_seq[..., i, :] # [T, N, 3]
            elif attr_name == "sh0":
                attr_dict[attr_name] = attr_seq.squeeze(dim=-2)  # [T, N, 3]
            elif attr_name == "opacities":
                attr_dict[attr_name] = attr_seq.unsqueeze(dim=-1)  # [T, N, 1]
            else:
                attr_dict[attr_name] = attr_seq
        
        return attr_dict

    def _attribute_dict_to_grid(self, attr_dict: Dict[str, Tensor], width: int, height: int,) -> Tuple[Dict[str, Tensor]]:
        """Reshape the splats list into a grid format for video representation.
        Args:
            attr_dict (Dict[str, Tensor]): Dictionary of splat attributes, where keys are attribute names
            and values are tensors of attributes stacked across frames.
            width (int): Width of the grid.
            height (int): Height of the grid.
            means_u_bit_depth (int): Bit depth for the upper part of the means attribute.
        Returns:
            Dict[str, Tensor]: A dictionary where keys are attribute names and values are tensors of attributes
            reshaped into a grid format.
        """
        splats_videos = {}
        n_frames = attr_dict["means"].shape[0]
        for attr_name, attr_data in attr_dict.items():
            ori_shape = list(attr_data.shape)
            new_shape = [n_frames, width, height] + ori_shape[2:]
            grid_attr_data = attr_data.reshape(new_shape)
            if attr_name == "means" and self.means_no_split is False:
                l_bit_depth = smart_look_up_dict(self.bit_depth_config, 'means_l')
                u_bit_depth = smart_look_up_dict(self.bit_depth_config, 'means_u')
                # For means, we split into lower and upper according to the specified bit depth
                video_l, video_u = bit_split(grid_attr_data, l_bit_depth, u_bit_depth)  # [T, H, W, 3], [T, H, W, 3]
                splats_videos['means_l'] = video_l
                splats_videos['means_u'] = video_u            
            else:
                splats_videos[attr_name] = grid_attr_data.view(n_frames, width,  height, -1)  # [T, H, W, C]
        return splats_videos

    
    def _grid_to_attribute_dict(
        self,
        splats_videos: Dict[str, Tensor],
        meta: Dict[str, Any],
    ) -> Dict[str, Tensor]:
        """
        Reconstructs the original attribute dictionary from the video grid format.
        This is the inverse function of `_attribute_dict_to_grid`.

        Args:
            splats_videos (Dict[str, Tensor]): A dictionary where keys are attribute video names
                                               (e.g., 'means_l', 'scales') and values are tensors
                                               in video grid format [T, H, W, C].
            ori_shape_dict (Dict[str, Tuple[int, ...]]): A dictionary mapping original attribute names
                                                         to their original shapes.

        Returns:
            Dict[str, Tensor]: The reconstructed dictionary of splat attributes in their
                               original list format (e.g., [T, N, C] or [T*N, C]).
        """
        attr_dict = {}
        bit_depth_dict = meta.get('bit_depth_dict', {})
        # Iterate over the original attribute names to ensure we reconstruct everything
        
        for attr_name, bit_depth in bit_depth_dict.items():
            dtype = get_dtype_for_bitdepth(bit_depth)["torch"]
            if attr_name== 'means_l':
                # Reconstruct 'means' from lower and upper 8-bit videos
                means_dtype = get_dtype_for_bitdepth(bit_depth*2)['torch']
                video_l = splats_videos['means_l'].to(means_dtype)
                video_u = splats_videos['means_u'].to(means_dtype)
                # Combine them back: upper_bits << l_bit_depth | lower_bits
                grid_attr_data = (video_u << bit_depth) | video_l
            elif attr_name == 'means_u':
                continue  # Skip 'means_u' since it's already handled with 'means_l'            
            else:
                # For all other attributes, they are stored directly
                if attr_name not in splats_videos:
                    raise KeyError(f"Attribute '{attr_name}' not found in the provided video dictionary.")
                grid_attr_data = splats_videos[attr_name].to(dtype)

            grid_shape = list(grid_attr_data.shape) # [N_frames, H, W, C]
            # Reshape to [N_frames, N_gaussians, C]
            n_frames, height, width = grid_shape[:3]
            n_gaussians = height * width
            new_shape = [n_frames, n_gaussians] + grid_shape[3:]
            attr_dict[attr_name] = grid_attr_data.reshape(new_shape)
            
        return attr_dict
    
    def _save_metadata(self, compress_dir: Path, prefix: str= "", **kwargs) -> None:
        """Saves metadata to a JSON file in the specified directory."""
        total_meta={}
        meta_path = compress_dir / f"{prefix}{CODEC_META_NAME}"
        width = kwargs.get('width', 0)
        height = kwargs.get('height', 0)
        pix_fmt_dict = kwargs.get('pix_fmt_dict', {})
        bit_depth_dict = kwargs.get('bit_depth_dict', {})
        total_meta['width'] = width
        total_meta['height'] = height
        total_meta['pix_fmt_dict'] = pix_fmt_dict
        total_meta['bit_depth_dict'] = bit_depth_dict
        
        smart_save_meta(total_meta, meta_path)
        logger.info(f"Metadata saved to {meta_path}")    
        
    def _load_metadata(self, compress_dir: Path, prefix: str= "") -> Dict[str, Any]:
        """Loads metadata from a JSON file in the specified directory."""
        meta_path = compress_dir / f"{prefix}{CODEC_META_NAME}"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        
        meta = smart_load_meta(meta_path)
        logger.info(f"Metadata loaded from {meta_path}")
        
        return meta
    
    def _build_attribute_configs(self, splats_videos: List[str]) -> None:
        """Builds the attribute configurations for encoding.
        This function creates dictionaries for pixel formats, bit depths, and quantization parameters
        based on the provided splat videos.
        Args:
            splats_videos (List[str]): List of attribute video names.
        Returns:
            None: The function sets up the `attribute_configs` attribute with dictionaries for pixel formats,
            bit depths, and quantization parameters.
        """
        pix_fmt_dict = {}
        bit_depth_dict = {}
        qp_dict = {}
        for attr_name, attr_video in splats_videos.items():
            qp_dict[attr_name] = smart_look_up_dict(self.qp_config, attr_name, 'pix_fmt')
            bit_depth_dict[attr_name] = smart_look_up_dict(self.bit_depth_config, attr_name, 'bit_depth')
            if self.pix_fmt_config:
                pix_fmt_dict[attr_name] = smart_look_up_dict(self.pix_fmt_config, attr_name, 'pix_fmt')
            else:
                # If no specific pixel format is provided, derive it according to the video shape
                logger.warning(f"No specific pixel format provided for {attr_name}. Deriving from video shape.")
                component_num = attr_video.shape[-1]
                if component_num == 1:
                    pix_fmt_dict[attr_name] = 'yuv400p'
                elif component_num == 3:
                    pix_fmt_dict[attr_name] = 'yuv444p'
                else:
                    raise ValueError(f"Unsupported number of components {component_num} for attribute {attr_name}. Expected 1 or 3.")
        self.attribute_configs= {
            'pix_fmt_dict': pix_fmt_dict,
            'bit_depth_dict': bit_depth_dict,
            'qp_dict': qp_dict,
        }
    
    def _get_encode_configs(self, yuv_name: str, width : int, height : int, frame_num : int) -> Dict[str, Any]:
        """Get the configuration parameters for a specific attribute based on its name."""
        configs = {
            'frame_num': frame_num,
            'gop_size': self.gop_size,
            'all_intra': self.all_intra,
            'width': width,
            'height': height,
            'pix_fmt': smart_look_up_dict(self.attribute_configs['pix_fmt_dict'], yuv_name),
            'bit_depth': smart_look_up_dict(self.attribute_configs['bit_depth_dict'], yuv_name),
            'qp': smart_look_up_dict(self.attribute_configs['qp_dict'], yuv_name),
        }

        return configs
    
    def _get_decode_configs(self, yuv_name: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Get the configuration parameters for a specific attribute based on its name."""
        configs = {          
            'pix_fmt': smart_look_up_dict(meta['pix_fmt_dict'], yuv_name),
            'bit_depth': smart_look_up_dict(meta['bit_depth_dict'], yuv_name),
        }
        return configs
    
    