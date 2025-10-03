import json
import os
import time
import copy
import glob
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, Any, Literal, Annotated

import numpy as np
import torch
from tqdm import tqdm
import yaml
import imagecodecs
from torch import Tensor
from typing_extensions import Literal
from matplotlib import pyplot as plt
import pandas as pd
# dataset
from datasets.colmap import Dataset, GSCDataset, Parser
# gsplat
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy

# VGSC
from gsc import set_random_seed, init_logging, get_logger, deep_update_dict
from gsc.utils.plot import plot_pie, group_data_auto_prefix
from gsc.utils.file import force_make_dirs, safe_make_dirs, smart_load_meta, smart_save_meta
from gsc.utils.color import COLOR_SPACE_STANDARD
from gsc.utils.gs_io import load_ply, save_ply, load_ply_sequence, format_splats
from gsc.pre_post_process.transform import param_transform, param_inverse_transform, seq_sh_pca_transform, seq_sh_pca_inverse_transform
from gsc.pre_post_process.prune import prune_splats_list
from gsc.codec import VgscCodec, PccCodec
from gsc.quantize.gs_quantize import (
    quantize_splats_list_jointly, 
    dequantize_splats_list_jointly,
    quantize_splats_list_seperately,
    dequantize_splats_list_seperately)
from gsc.config import (
    QUANT_CONFIG_KEYS, QUANT_CONFIG, QUANT_META_NAME, 
    INFO_NAME, ATTRIBUTE_MAP,
    DEFAULT_QP, SORT_TYPES,
    SUBSAMPLING_METHODS, UPSAMPLING_METHODS, 
)
# metrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

PIPE_STAGE_CHOICES = Literal["benchmark", "codec","encode", "decode", "decode_eval", "render", "eval", "preprocess", "quantize"] 

def get_default_qp_config() -> Dict[str, Any]:
    return DEFAULT_QP.copy()


@dataclass
class CodecConfig:
    """
    Configuration for Gaussian Splat Codec.
    Defines preprocessing, transform, quantization, and codec parameters.
    """

    # General codec parameters
    save_rec_ply: bool = False                                          # Save reconstructed PLY files
    frame_num: int = 1                                                  # Number of frames to encode
    gop_size: int = 16                                                  # GOP size for video compression

    # Prune parameters
    prune_type: Optional[str] = None                                    # None, "threshold", "outliers", "ratio"
    prune_thres_opacities: Optional[float] = None                       # Threshold for opacities (e.g., -7.0)
    prune_thres_scales: Optional[float] = None                          # Threshold for scales (e.g., -8.0)

    # Transform parameters
    trans_means_log: bool = False                                         # Log transform for means

    trans_quats_norm: bool = False                                        # Normalize quaternions
    trans_quats_posi_w: bool = False                                      # Ensure quaternion w ≥ 0
    trans_quats_euler: bool = False                                       # Euler angle representation
    trans_quats_rod: bool = False                                         # Rodrigues vector representation

    trans_sh0_ycbcr: bool = False                                         # SH DC: RGB → YCbCr
    trans_shN_ycbcr: bool = False                                         # SH AC: RGB → YCbCr
    color_standard: Optional[COLOR_SPACE_STANDARD] = "BT470"                             # Color space standard (MPEG GSC N00677)
    trans_shN_pca: bool = False                                           # Apply PCA on SH AC coefficients
    shN_rank: int = 21                                                  # PCA rank for SH AC coefficients

    # Quantization parameters
    quant_type: QUANT_CONFIG_KEYS = "video_N01292"                      # Predefined quantization type
    quant_ply_dir: Optional[str] = None                                 # Directory for quantized splats
    quant_config: Optional[Dict[str, Any]] = None                       # Custom quantization config
    quant_per_channel: bool = False                                     # Quantize per channel
    quant_shN_per_channel: bool = False                                 # Quantize SH AC per channel
    keep_spatial: bool = False                                          # Preserve spatial distribution
    quant_seperate: bool = False                                        # Quantize splats separately
    bit_depth_config: Optional[Dict[str, Any]] = None                   # Bit depth config for attributes

    # Codec parameters
    all_intra: bool = False                                         # Enable All-Intra coding mode

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)


@dataclass
class GpccCodecConfig(CodecConfig):
    encoder_path: str = "third_party/pcc_codec/bin/gpcc/tmc3"
    decoder_path: str = "third_party/pcc_codec/bin/gpcc/tmc3"
    encode_config_path: str = "third_party/pcc_codec/cfg/gpcc/mpeg151/JEE6.6/octree-raht/lossless-geom-lossy-attrs/r01/encoder.cfg"
    decode_config_path: str = "third_party/pcc_codec/cfg/gpcc/mpeg151/JEE6.6/octree-raht/lossless-geom-lossy-attrs/r01/decoder.cfg"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "encoder_path": self.encoder_path,
            "decoder_path": self.decoder_path,
            "encode_config_path": self.encode_config_path,
            "decode_config_path": self.decode_config_path,
            "quant_type": self.quant_type,
        }


@dataclass
class VgscCodecConfig(CodecConfig):
    """
    Configuration for VGSC codec.
    Extends CodecConfig with VGSC-specific parameters: sorting, YUV, and video codec.
    """

    # Sorting parameters
    sort_type: SORT_TYPES = "morton"                                         # Sorting method for better spatial locality
    sort_with_shN: bool = True                                               # Include SHN in PLAS sorting
    weight_dict: Optional[Dict[str, float]] = field(default=None)            # Attribute weights for PLAS sorting, e.g., {"means":1.0, "opacities":0.5}

    # YUV parameters
    pix_fmt_config: Optional[Dict[str, str]] = None                          # Pixel format configuration per attribute
    means_no_split: bool = False                                             # Keep means unsplit
    chroma_sub_method: SUBSAMPLING_METHODS = "bicubic"                       # Chroma subsampling method
    chroma_up_method: UPSAMPLING_METHODS = "bicubic"                         # Chroma upsampling method

    # Video codec parameters
    video_codec_type: str = "ffmpeg"                                         # Video codec name: "vvenc", "vtm", "hm", "ffmpeg"
    encoder_path: str = "ffmpeg"                                             # Path to encoder binary
    decoder_path: str = "ffmpeg"                                             # Path to decoder binary
    encode_config_path: Optional[str] = None                                 # Path to encoder config file
    decode_config_path: Optional[str] = None                                 # Path to decoder config file
    qp_config: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_QP))  # QP configuration per attribute
    

    def to_dict(self) -> Dict[str, Any]:
        params_name = [
            "all_intra", "gop_size", "means_no_split",
            'qp_config', 'bit_depth_config', 'pix_fmt_config',
            "sort_type", "sort_with_shN", "weight_dict",
            "chroma_sub_method", "chroma_up_method",
            "video_codec_type", "encoder_path", "decoder_path", 
            "encode_config_path", "decode_config_path",
        ]
        
        return {k: getattr(self, k) for k in params_name if hasattr(self, k)}


@dataclass
class Config:
    # ---------------- Codec ----------------
    codec: CodecConfig = field(default_factory=VgscCodecConfig)
    
    # ---------------- Pipeline ----------------
    pipe_stage: PIPE_STAGE_CHOICES = "codec"  # Current stage of the pipeline
    
    # ---------------- Rendering ----------------
    render_traj_path: str = "interp"         # Path for rendering camera trajectory
    normalize_world_space: bool = True       # Normalize scene to world space
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # ---------------- Training ----------------
    batch_size: int = 1                       # Batch size; learning rates scaled automatically
    near_plane: float = 0.01                  # Near clipping plane
    far_plane: float = 1e10                   # Far clipping plane
    sh_degree: int = 3                        # Spherical harmonics degree
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(default_factory=DefaultStrategy)

    # Rasterization options
    packed: bool = False                       # Use packed mode (less memory, slightly slower)
    sparse_grad: bool = False                  # Use sparse gradients (experimental)
    antialiased: bool = False                  # Enable anti-aliasing (may affect metrics)

    # Depth supervision
    depth_loss: bool = False                   # Enable depth loss (experimental)
    depth_lambda: float = 1e-2                 # Depth loss weight

    # TensorBoard
    tb_every: int = 100                        # Log every N steps
    tb_save_image: bool = False                # Save images to TensorBoard

    # ---------------- Dataset / Scene ----------------
    scene_type: Literal["gsc_static", "gsc_dynamic", "default"] = "default"  # Scene type, default can be used for Mip-NeRF 360, Tanks and Temples, etc.
    test_view_id: Optional[List[int]] = None    # Test view IDs; if None, use default split
    lpips_net: Literal["vgg", "alex"] = "alex"  # Network for LPIPS metric
    data_dir: str = ""                          # Directory of GT images
    data_factor: int = 4                        # Downsample factor
    test_every: int = 8                         # Test image interval
    patch_size: Optional[int] = None            # Random crop size (experimental)

    # ---------------- Output ----------------
    result_dir: str = "results/"
    ply_dir: str = ""                            # Uncompressed splats directory
    ori_render_dir: Optional[str] = None         # Rendered original splats
    compressed_ply_dir: Optional[str] = None     # Compressed splats directory
    frame_num: int = 1                           # Number of frames to process

 
   
    def to_dict(self):
        return self.__dict__

class Runner:
    def __init__(self,  
                 local_rank: int, 
                 world_rank: int, 
                 world_size: int,
                 cfg: Config ) -> None:
        
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"
                 
        # Configure the self.logger
        self.set_up_logging(cfg.pipe_stage)   
        
        # Set up directories for saving results
        self.set_up_directories(cfg.pipe_stage)
         
        # Save the configuration
        self.save_config(cfg.pipe_stage)
        
        # Init splats list
        self.splats_list = None
        
        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # frame num
        self.frame_num = cfg.frame_num

        self.codec = cfg.codec.to_dict()
        
        if isinstance(cfg.codec, VgscCodecConfig):
            self.codec_method = VgscCodec(**self.codec)
        elif isinstance(cfg.codec, GpccCodecConfig):
            self.codec_method = PccCodec(**self.codec)
        else:
            raise TypeError(f"Unknown codec config type: {type(cfg.codec)}")
        
    
    def set_up_logging(self, pipe_stage: str) -> None:
        """Set up logging for the runner."""
        self.log_dir = f"{self.cfg.result_dir}/log"
        safe_make_dirs(self.log_dir)
        if self.local_rank == 0:
            init_logging(f"{self.log_dir}/gs_{pipe_stage}.log")
        self.logger = get_logger(f"GS{pipe_stage}Runner")
        self.logger.info(f"Logging initialized at {self.log_dir}/gs_{pipe_stage}.log")
    
    def set_up_directories(self, pipe_stage: str) -> None:
        # Setup output directories.
        safe_make_dirs(self.cfg.result_dir)
        self.config_dir = f"{self.cfg.result_dir}/config"
        safe_make_dirs(self.config_dir)
        self.result_dir = self.cfg.result_dir
        self.compress_dir = f"{self.result_dir}/compression"
        self.intermediate_dir = f"{self.result_dir}/intermediate"
        self.reconstructed_dir = f"{self.result_dir}/reconstructed"
        self.stats_dir = f"{self.result_dir}/stats"
        safe_make_dirs(self.stats_dir)
        self.render_dir = f"{self.result_dir}/renders"
        if pipe_stage == "codec" or pipe_stage == "benchmark":    
            force_make_dirs(self.compress_dir)
            force_make_dirs(self.intermediate_dir)
            force_make_dirs(self.reconstructed_dir)
            force_make_dirs(self.stats_dir)
            force_make_dirs(self.render_dir)
        elif pipe_stage == "encode":
            force_make_dirs(self.compress_dir)
            force_make_dirs(self.intermediate_dir)
            force_make_dirs(self.stats_dir) 
        elif pipe_stage == "decode" or pipe_stage == "decode_eval":
            safe_make_dirs(self.reconstructed_dir)      
        elif pipe_stage == "render":
            safe_make_dirs(self.render_dir)
        elif pipe_stage == "preprocess":
            safe_make_dirs(self.intermediate_dir)
        elif pipe_stage == "quantize":
            safe_make_dirs(self.intermediate_dir)
            safe_make_dirs(self.compress_dir)
        else:
            safe_make_dirs(self.compress_dir)
            safe_make_dirs(self.intermediate_dir)
            safe_make_dirs(self.reconstructed_dir)
            safe_make_dirs(self.render_dir)

        self.logger.info('Directories set up successfully.')
    
    def save_ply(self):
        self.logger.info(f"Saving reconstructed PLY files to {self.reconstructed_dir}...")
        s_time = time.time()
        for f_id, splats in enumerate(tqdm(self.splats_list, desc="Saving PLY files")):
            save_ply(splats, f"{self.reconstructed_dir}/frame{f_id:03d}.ply")
        duration = time.time() - s_time
        self.logger.info(f"Saving PLY files time: {duration:.2f} seconds.")
        self.save_info(duration, "save_ply_time")

    def save_config(self, pipe_stage: str):
        """Save the configuration to a YAML file."""
        config_path = os.path.join(self.config_dir,f"{pipe_stage}_config.yaml")
        with open(config_path, "w") as f:
            if pipe_stage == "render":
                # Save only the render-related configuration, excluding codec settings
                render_cfg = copy.deepcopy(self.cfg)
                render_cfg.codec = None
                yaml.dump(render_cfg.to_dict(), f, default_flow_style=False)
            else:
                yaml.dump(self.cfg.to_dict(), f, default_flow_style=False)
        self.logger.info(f"Configuration saved to {config_path}")
        
    def save_info(self, info: Any, info_name: str) -> None:
        """Save statistics to a file, appending/updating by info_name."""
        info_path = os.path.join(self.stats_dir, INFO_NAME)
        if os.path.exists(info_path):
            existing_info = smart_load_meta(info_path)
            if not isinstance(existing_info, dict):
                existing_info = {}
        else:
            existing_info = {}
        # if info is float, round to 3 decimal places
        if isinstance(info, float):
            info = round(info, 3)
        existing_info[info_name] = info
        smart_save_meta(existing_info, info_path)
        self.logger.info(f"'{info_name}: {info}' saved to {info_path}")

    def load_ply_sequences(self, ply_dir) -> None:
        """Load PLY sequences from the specified directory."""
        self.logger.info(f"Loading PLY sequences from {ply_dir}")
        self.splats_list, self.ply_filenames = load_ply_sequence(ply_dir, self.frame_num, return_type='torch')
        self.logger.info(f"Successfully loaded PLY sequences from {ply_dir}.")
        
    def load_quant_ply_sequences(self) -> bool:
        """Load quantized PLY sequences from the specified directory."""
        try:   
            self.logger.info(f"Loading quantized PLY sequences from {self.cfg.codec.quant_ply_dir}")
            self.splats_list, self.ply_filenames = load_ply_sequence(
                self.cfg.codec.quant_ply_dir, 
                self.frame_num, 
                return_type='torch'
            )
            self.quant_ply_dir = self.cfg.codec.quant_ply_dir
            self.quant_meta_path = f"{self.cfg.codec.quant_ply_dir}/${QUANT_META_NAME}"
            assert os.path.exists(self.quant_meta_path), (
                f"Quantization metadata not found at {self.quant_meta_path}. "
            )      
            self.logger.info(f"Successfully loaded quantized PLY sequences from {self.cfg.codec.quant_ply_dir}.")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to load quantized PLY sequences: {e}")
            return False

   
    def update_splats_list(
        self, splats_list: List
    ) -> None:
        """Update the splats_list with new splats."""
        assert len(splats_list) == self.frame_num, (
            f"Length of splats_list ({len(splats_list)}) does not match frame_num ({self.frame_num})."
        ) 
        if self.splats_list is not None:  
            for splats, new_splats in zip(self.splats_list, splats_list):
                formatted = format_splats(new_splats, return_type='torch', device=self.device)
                for k in splats:
                    with torch.no_grad():
                        splats[k].data = formatted[k].data
        else:
            self.splats_list = []
            for new_splats in splats_list:
                formatted = format_splats(new_splats, return_type='torch', device=self.device)
                self.splats_list.append(formatted)

        self.logger.info(f"Updated splats_list with {self.frame_num} frames.")
    
    def set_up_datasets(
        self, data_dir: str, frame_num: int, cfg: Config
    ) -> Tuple[List[Dataset], List[Dataset]]:
        if self.cfg.scene_type == "gsc_static":
            assert frame_num == 1, "For gsc_static, frame_num must be 1."
            folders = [data_dir]
        else:
            all_items = sorted(glob.glob(os.path.join(data_dir, "*")))
            folders = [item for item in all_items if os.path.isdir(item)]
            self.logger.info(f"Found {len(folders)} scene folders in {data_dir}.")

        trainset_list = []
        valset_list = []
        for folder in tqdm(folders[:frame_num], desc="Loading colmap results"):
            parser = Parser(
                data_dir=folder,
                factor=cfg.data_factor,
                normalize=cfg.normalize_world_space,
                test_every=cfg.test_every,
            )
            trainset = GSCDataset(
                parser,
                split="train",
                patch_size=cfg.patch_size,
                load_depths=cfg.depth_loss,
                test_view_ids=cfg.test_view_id,
            )
            valset = GSCDataset(
                parser, 
                split="val", 
                test_view_ids=cfg.test_view_id,)                    
            trainset_list.append(trainset)
            valset_list.append(valset)
        self.trainset_list = trainset_list
        self.valset_list = valset_list
        self.cfg.test_view_id = valset.indices

    def preprocess(self) -> None:
        self.logger.info("Preprocessing splats...")
        s_time = time.time()
        # Pruning
        if self.cfg.codec.prune_type is not None:
            self.logger.info("Pruning splats before compression.")
            splats_list = prune_splats_list(
                splats_list=self.splats_list,
                prune_type=self.cfg.codec.prune_type,
                prune_thres_opacities=self.cfg.codec.prune_thres_opacities,
                prune_thres_scales=self.cfg.codec.prune_thres_scales,
            )
            self.update_splats_list(splats_list)       
        # Transform parameters if needed

        self.logger.info("Transforming splats parameters to new domain.")
        splats_list = param_transform(
            splats_list=self.splats_list, 
            trans_means_log=self.cfg.codec.trans_means_log,
            trans_sh0_ycbcr=self.cfg.codec.trans_sh0_ycbcr,
            trans_shN_ycbcr=self.cfg.codec.trans_shN_ycbcr,
            color_standard=self.cfg.codec.color_standard,
            trans_quats_rod=self.cfg.codec.trans_quats_rod,
            trans_quats_euler=self.cfg.codec.trans_quats_euler,
            trans_quats_norm=self.cfg.codec.trans_quats_norm,
            trans_quats_posi_w=self.cfg.codec.trans_quats_posi_w,)    
        self.update_splats_list(splats_list)         
        
        if self.cfg.codec.trans_shN_pca:
            pca_s_time = time.time()
            splats_list=seq_sh_pca_transform(
                splats_list=self.splats_list,
                pca_info_dir=self.compress_dir,
                rank=self.cfg.codec.shN_rank) 
            pca_duration = time.time() - pca_s_time
            self.save_info(pca_duration, "PCA_time")    
            self.update_splats_list(splats_list)

        duration = time.time() - s_time
        self.save_info(duration, "Preprocess_time")
    

    def postprocess(self) -> None:
        self.logger.info("Postprocessing splats...")
        s_time = time.time()
        # 1. Inverse PCA transform splats if needed
        if self.cfg.codec.trans_shN_pca:
            s_time = time.time()
            splats_list = seq_sh_pca_inverse_transform(
                splats_list=self.splats_list,
                pca_info_dir=self.compress_dir,
                rank=self.cfg.codec.shN_rank)
            inverse_pca_duration = time.time() - s_time
            self.save_info(inverse_pca_duration, "Inverse_PCA_time")
            self.update_splats_list(splats_list)
        # 2. Inverse transform parameters if needed
        self.logger.info("Inverse transforming splats parameters to original domain.")
        splats_list = param_inverse_transform(
            splats_list=self.splats_list, 
            trans_means_log=self.cfg.codec.trans_means_log,
            trans_sh0_ycbcr=self.cfg.codec.trans_sh0_ycbcr,
            trans_shN_ycbcr=self.cfg.codec.trans_shN_ycbcr,
            color_standard=self.cfg.codec.color_standard,
            trans_quats_rod=self.cfg.codec.trans_quats_rod,
            trans_quats_euler=self.cfg.codec.trans_quats_euler,)       
        self.update_splats_list(splats_list)

        duration = time.time() - s_time
        self.save_info(duration, "Postprocess_time")
        
    def quantize(self) -> None:
        """Quantize the splats_list and save to disk."""
                        
        self.logger.info("Quantizing splats before compression...")
        s_time = time.time()
        quant_ply_dir = self.intermediate_dir
        self.quant_ply_dir = quant_ply_dir
        
        quant_config = QUANT_CONFIG.get(self.cfg.codec.quant_type, None)
        if quant_config is None:
            raise ValueError(f"Unknown quantization type: {self.cfg.codec.quant_type}. Available types: {list(QUANT_CONFIG.keys())}")
        custom_quant_config = self.cfg.codec.quant_config
        if custom_quant_config is not None:
            self.logger.info(f"Updating quantization configuration using custom configuration: {custom_quant_config}")
            quant_config = deep_update_dict(quant_config, custom_quant_config)
              
        if self.cfg.codec.quant_seperate:
            quant_splats_list, quant_meta = quantize_splats_list_seperately(self.splats_list, 
                                                                     quant_config, 
                                                                     self.cfg.codec.bit_depth_config,
                                                                     self.cfg.codec.quant_per_channel,
                                                                     self.cfg.codec.quant_shN_per_channel,
                                                                     self.cfg.codec.keep_spatial,)
        else:
            quant_splats_list, quant_meta = quantize_splats_list_jointly(self.splats_list, 
                                                                        quant_config, 
                                                                        self.cfg.codec.bit_depth_config,
                                                                        self.cfg.codec.quant_per_channel,
                                                                        self.cfg.codec.quant_shN_per_channel,
                                                                        self.cfg.codec.keep_spatial,)
        self.update_splats_list(quant_splats_list)

        duration = time.time() - s_time
        self.save_info(duration, "Quant_time") # The time for quantization only, excluding saving time
        safe_make_dirs(quant_ply_dir)
        # Save quantized splats to disk
        for f_id, splats in enumerate(quant_splats_list):
            quant_ply_path = f"{quant_ply_dir}/{self.ply_filenames[f_id]}"
            save_ply(splats, quant_ply_path)
        self.logger.info(f"Quantized splats saved to {quant_ply_dir}.")
        # Save quantization metadata
        quant_meta_path = f"{self.compress_dir}/{QUANT_META_NAME}"
        smart_save_meta(quant_meta, quant_meta_path)
        self.logger.info(f"Quantization metadata saved to {quant_meta_path}.")
        self.logger.info(f"Quantization metadata:\n {quant_meta}")

    
    def dequantize(self) -> None:
        """De-quantize the splats_list and save to disk."""
        self.logger.info("De-quantizing splats after compression...")
        s_time = time.time()
        quant_meta_path = f"{self.compress_dir}/{QUANT_META_NAME}"
        quant_meta = smart_load_meta(quant_meta_path)
        if self.cfg.codec.quant_seperate:
            dequant_splats_list = dequantize_splats_list_seperately(self.splats_list, quant_meta)
        else:
            dequant_splats_list = dequantize_splats_list_jointly(self.splats_list, quant_meta)
        self.update_splats_list(dequant_splats_list)
        
        duration = time.time() - s_time
        self.save_info(duration, "Dequant_time")
            
    def vgsc_encode(self):
        self.logger.info("Running VGSC encoding...")  
        s_time = time.time()        
        self.codec_method.encode(self.splats_list, self.compress_dir)
        duration = time.time() - s_time
        self.save_info(duration, "vgsc_enc_time")
        
    def vgsc_decode(self):
        self.logger.info("Running VGSC decoding...")
        s_time = time.time()
        splats_list_c = self.codec_method.decode(self.compress_dir)
        self.update_splats_list(splats_list_c)
        duration = time.time() - s_time
        self.save_info(duration, "vgsc_dec_time")
        
    def gpcc_encode(self):
        self.logger.info("Running GPCC encoding...")
        s_time = time.time()
        for f_id in tqdm(range(self.frame_num), desc="Encoding frames"):
            quant_ply_path = f"{self.quant_ply_dir}/{self.ply_filenames[f_id]}"            
            encoded_bin_path = f"{self.compress_dir}/frame{f_id:03d}.bin"
            self.codec_method.encode(quant_ply_path, encoded_bin_path)
        duration = time.time() - s_time
        self.save_info(duration, "gpcc_enc_time")
        
    def gpcc_decode(self):
        self.logger.info("Running GPCC decoding...")
        s_time = time.time()
        full_splats_list_c = []
        for f_id in tqdm(range(self.frame_num), desc="Decoding frames"):
            encoded_bin_path = f"{self.compress_dir}/frame{f_id:03d}.bin"
            new_splats = self.codec_method.decode(encoded_bin_path)
            full_splats_list_c.append(new_splats)
        self.update_splats_list(full_splats_list_c)
        duration = time.time() - s_time
        self.save_info(duration, "gpcc_dec_time")
          
    def run(self):
        self.logger.info("Starting encoding process...")  
        load_quantized = self.load_quant_ply_sequences()
        if not load_quantized:
            self.load_ply_sequences(self.cfg.ply_dir)
            if self.cfg.codec.split_type is not None:
                self.split_blocks()  
            self.preprocess()
            self.quantize()
            
        if isinstance(self.cfg.codec, VgscCodecConfig):
            self.vgsc_encode()
            self.vgsc_decode()
        elif isinstance(self.cfg.codec, GpccCodecConfig):
            self.gpcc_encode()
            self.gpcc_decode()
        else:
            raise NotImplementedError(f"{type(self.cfg.codec).__name__} has not been implemented.")
                
        self.dequantize()  
        self.postprocess()
        
        if self.cfg.codec.split_type is not None:
            self.merge_blocks()
        
        if self.cfg.codec.save_rec_ply:
            self.save_ply()
        else:
            self.logger.warning("Reconstructed PLY not saved. Set 'save_rec_ply' to True in the codec config to save it.")
        self.eval(render_stage="compress")
                                           

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        splats: Dict[str, Tensor],
        masks: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:

        means = splats["means"] # [N, 3]
        quats = splats["quats"] # [N, 4], rasterization does normalization internally
        scales = torch.exp(splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(splats["opacities"])  # [N,]
        sh0, shN = splats["sh0"], splats["shN"] 
    
        colors = torch.cat([sh0, shN], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info
    
    @torch.no_grad()
    def eval(self, render_stage: str = "val", splats_list: Optional[List[Dict]] = None):
        """Entry for evaluation."""
        self.logger.info("Running evaluation...")
        self.set_up_datasets(self.cfg.data_dir, self.frame_num, self.cfg)
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank     

        # dict to save metrics of each frame
        seq_stats = defaultdict(dict) 

        # if splats_list is not provided, use the default splats_list
        if splats_list is None:
            splats_list_to_render = self.splats_list
        else:
            splats_list_to_render = splats_list
        
        splats_list_to_render = [
            format_splats(splats, return_type='torch', device=device) for splats in splats_list_to_render
        ]
        

        # loop on frame
        for f_id, (splats, val_dataset) in enumerate(zip(splats_list_to_render, self.valset_list)):
            valloader = torch.utils.data.DataLoader(
                val_dataset, batch_size=1, shuffle=False, num_workers=1
            )
            ellipse_time = 0
            metrics = defaultdict(list)  

            # loop on view
            for v_id, data in enumerate(valloader):
                camtoworlds = data["camtoworld"].to(device)
                Ks = data["K"].to(device)
                pixels = data["image"].to(device) / 255.0
                masks = data["mask"].to(device) if "mask" in data else None
                height, width = pixels.shape[1:3]
                
                torch.cuda.synchronize()
                tic = time.time()
                colors, _, _ = self.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=cfg.sh_degree,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    masks=masks,
                    splats=splats,
                )  
                torch.cuda.synchronize()
                ellipse_time += time.time() - tic

                colors = torch.clamp(colors, 0.0, 1.0) # [1, H, W, 3]
                canvas_list = [pixels, colors]

                if world_rank == 0:
                    # write images 
                    # canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy() # side by side
                    canvas = canvas_list[1].squeeze(0).cpu().numpy() # signle image
                    canvas = (canvas * 255).astype(np.uint8)

                    png = imagecodecs.png_encode(canvas) 
                    with open(f"{self.render_dir}/{render_stage}_frame{f_id:03d}_testv{v_id:03d}.png", "wb") as f:
                        f.write(png)

                    pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                    metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                    metrics["lpips"].append(self.lpips(colors_p, pixels_p))            
        
            if world_rank == 0:
                ellipse_time /= len(valloader)

                stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
                stats.update(
                    {
                        "ellipse_time": ellipse_time,
                        "num_GS": len(splats["means"]),
                    }
                )
                self.logger.info(
                    f"Metrics on frame{f_id}:"
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
                # write into dict
                seq_stats[f"frame{f_id:03d}"] = stats

        # calculate average
        total_metrics = {k: 0 for k, v in stats.items()}
        frame_count = len(seq_stats)
        
        for frame, metrics in seq_stats.items():
            for metric, value in metrics.items():
                total_metrics[metric] += value
        
        avg_metrics = {metric: total/frame_count for metric, total in total_metrics.items()}

        seq_stats[f"average"] = avg_metrics
        self.logger.info(
            f"Average Metrics:"
            f"PSNR: {avg_metrics['psnr']:.3f}, SSIM: {avg_metrics['ssim']:.4f}, LPIPS: {avg_metrics['lpips']:.3f} "
            f"Time: {avg_metrics['ellipse_time']:.3f}s/image "
            f"Number of GS: {avg_metrics['num_GS']}"
        )
        # save metrics
        with open(f"{self.stats_dir}/{render_stage}.json", "w") as f:
            json.dump(seq_stats, f, indent=4)        

        return seq_stats

    def eval_pngs_with_gsc_ctc_metrics(self, ref_prefix: str = "val", test_prefix: str = "compress"):
        from utils.mpeg.gsc_metric import run_QMIV_metric_for_pngs, run_LPIPS_for_pngs
        from pathlib import Path
        self.set_up_datasets(self.cfg.data_dir, self.frame_num, self.cfg)
        height, width = self.valset_list[0][0]["image"].shape[0:2]
        resolution = f"{width}x{height}"

        os.makedirs(f"{self.cfg.result_dir}/log", exist_ok=True)

        gsc_metrics_across_test_views = defaultdict(dict)
        
        # Create progress bar
        pbar = tqdm(range(len(self.cfg.test_view_id)), desc="Calculating quality metrics")
        
        for i, test_view_id in enumerate(self.cfg.test_view_id):
            if self.cfg.ori_render_dir is not None and os.path.exists(self.cfg.ori_render_dir):
                ref_png_filename = Path(f"{self.cfg.ori_render_dir}/renders/{ref_prefix}_frame{{:03d}}_testv{test_view_id:03d}.png")
            else:
                ref_png_filename = Path(f"{self.cfg.result_dir}/renders/{ref_prefix}_frame{{:03d}}_testv{test_view_id:03d}.png")
            
            render_png_filename = Path(f"{self.cfg.result_dir}/renders/{test_prefix}_frame{{:03d}}_testv{test_view_id:03d}.png")
            saved_log_file = Path(f"{self.cfg.result_dir}/log/QMIV_testv{test_view_id:03d}.txt")
            
            # Record QMIV timing
            start_time = time.time()
            gsc_metrics = run_QMIV_metric_for_pngs(render_png_filename,
                                                ref_png_filename,
                                                resolution=resolution,
                                                saved_log_file=saved_log_file)
            qmiv_time = time.time() - start_time
            
            # Record LPIPS timing
            start_time = time.time()
            lpips_dict = run_LPIPS_for_pngs(render_png_filename,
                                        ref_png_filename,
                                        lpips_calculator=self.lpips)
            lpips_time = time.time() - start_time
            
            gsc_metrics.update(lpips_dict)
            gsc_metrics_across_test_views[f"testv{test_view_id:03d}"] = gsc_metrics
            
            # Update progress bar with timing info
            pbar.set_postfix({
                'QMIV': f'{qmiv_time:.1f}s',
                'LPIPS': f'{lpips_time:.1f}s',
                'Total': f'{qmiv_time + lpips_time:.1f}s'
            })
            pbar.update(1)
        
        pbar.close()

        metric_names = gsc_metrics_across_test_views[f"testv{0:03d}"].keys()
        for metric in metric_names:
            total = sum(gsc_metrics_across_test_views[f"testv{i:03d}"][metric] 
                    for i in range(len(self.cfg.test_view_id)))
            gsc_metrics_across_test_views["average"][metric] = total / len(self.cfg.test_view_id)
        
        # save quality metrics from each views and average metrics
        with open(os.path.join(self.cfg.result_dir, "stats", "gsc_metrics.json"), "w") as fp:
            json.dump(gsc_metrics_across_test_views, fp, indent=4)

    def summary(self,):
        import pandas as pd
        def format_size(size_bytes):
            """Convert byte size to readable format (KB, MB, GB, etc.)"""
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024**2:
                return f"{size_bytes/1024:.2f} KB"
            elif size_bytes < 1024**3:
                return f"{size_bytes/(1024**2):.2f} MB"
            else:
                return f"{size_bytes/(1024**3):.2f} GB"
            
        ### rate summary
     
        # Check if directory exists
        if not os.path.exists(self.compress_dir):
            self.logger.info(f"Error: Directory '{self.compress_dir}' does not exist")
            return
        
        # Store file and size information
        file_sizes = {}
        total_size = 0
        
        for item in os.listdir(self.compress_dir):
            item_path = os.path.join(self.compress_dir, item)
            if os.path.isfile(item_path):
                size = os.path.getsize(item_path)
                file_sizes[item] = size
                total_size += size

        file_sizes_grouped = {}
        for key, value in ATTRIBUTE_MAP.items():
            size = sum(size for fname, size in file_sizes.items() if fname.startswith(value))
            file_sizes_grouped[key] = size

        # save to json
        with open(os.path.join(self.cfg.result_dir, "stats", "storage.json"), "w") as fp:
            json.dump(file_sizes_grouped, fp, indent=4)
                # plot pie chart
        plot_pie(file_sizes_grouped, f"{self.cfg.result_dir}/stats/storage_pie_chart.png", title="Storage")

            
        # Get bitrate
        Byte_to_Kbps = lambda filesize, n_frame: filesize / 1024 / n_frame * 8 * 30
        bitrate_Kbps = Byte_to_Kbps(total_size, self.frame_num)
        
        Byte_to_Mbps = lambda filesize, n_frame: filesize / (1024 ** 2) / n_frame * 8 * 30
        bitrate_Mbps = Byte_to_Mbps(total_size, self.frame_num)


        # Calculate percentage
        percentages = {name: (size / total_size) * 100 for name, size in file_sizes.items()}
    
        # Storage breakdown table
        table_data = []
        for name, size in sorted(file_sizes.items(), key=lambda x: x[1], reverse=True):
            size_formatted = format_size(size)
            percentage = percentages[name]
            table_data.append([name, size_formatted, f"{percentage:.2f}%"])
        
        # Create pandas DataFrame for table
        df = pd.DataFrame(table_data, columns=["Filename", "Size", "Percentage"])
        csv_path = os.path.join(self.cfg.result_dir, "stats", "storage_detail.csv")
        df.to_csv(csv_path, index=False)
        self.logger.info(f"CSV file saved to: {csv_path}")

        ### distortion summary
        # compressed vs GT
        with open(os.path.join(self.cfg.result_dir, "stats", "compress.json"), "r") as fp:
            quality_metrics = json.load(fp)
            avg_quality_metrics = quality_metrics["average"]
        
        # compressed vs val (before compression vs after compression)
        with open(os.path.join(self.cfg.result_dir, "stats", "gsc_metrics.json"), "r") as fp:
            gsc_metrics = json.load(fp)
            avg_gsc_metrics = gsc_metrics["average"]
        
        # save summary into a json file
        rd_summary_rendered = {key: value for key, value in avg_gsc_metrics.items()}
        rd_summary_GT = {key: value for key, value in avg_quality_metrics.items() if key != "ellipse_time"}
        rd_summary_GT["bytes"] = total_size
        rd_summary_GT["bitrate"] = bitrate_Mbps
        rd_summary_GT["bitrate_Kbps"] = bitrate_Kbps
        
        rd_summary_rendered["bytes"] = total_size
        rd_summary_rendered["bitrate"] = bitrate_Mbps
        rd_summary_rendered["bitrate_Kbps"] = bitrate_Kbps
        
        
        with open(os.path.join(self.cfg.result_dir, "rd_summary_GT.json"), "w") as fp:
            json.dump(rd_summary_GT, fp, indent=4)
        with open(os.path.join(self.cfg.result_dir, "rd_summary_rendered.json"), "w") as fp:
            json.dump(rd_summary_rendered, fp, indent=4)

    def stack_render_img_to_vid(self, render_stage_list: List[str] = ["compress", "val"]):
        # remove existing video files
        for ext in ["*.mp4", "*.yuv"]:
            for file in glob.glob(os.path.join(self.cfg.result_dir, "renders", ext)):
                os.remove(file)

        for render_stage in render_stage_list:
            for test_view_id in range(len(self.cfg.test_view_id)):
                # png sequence to mp4 for visualization
                cmd = (f'ffmpeg -framerate 30 -i "{self.cfg.result_dir}/renders/{render_stage}_frame%03d_testv{test_view_id:03d}.png" '
                    f'-c:v libx264 -pix_fmt yuv420p -crf 20 -preset medium '
                    f'-profile:v high -level 4.1 -movflags +faststart "{self.cfg.result_dir}/renders/{render_stage}_testv{test_view_id:03d}.mp4"')
                
                self.logger.info(f"Running: {cmd}")
                os.system(cmd)
                self.logger.info(f"Video created for {render_stage}, test view {test_view_id}")

                # png sequence to yuv for MPEG GSC metrics (not used for now)
                cmd = (f'ffmpeg -framerate 30 -i "{self.cfg.result_dir}/renders/{render_stage}_frame%03d_testv{test_view_id:03d}.png" '
                    f'-c:v rawvideo -pix_fmt yuv420p '
                    f'"{self.cfg.result_dir}/renders/{render_stage}_testv{test_view_id:03d}.yuv"')
                
                self.logger.info(f"Running: {cmd}")
                os.system(cmd)
                self.logger.info(f"YUV Video created for {render_stage}, test view {test_view_id}")
    
    def compare_render_stats(self, stats1: Dict, stats2: Dict, name1: str = "Original", name2: str = "Modified") -> None:
        """
        Compare rendering statistics between two sets of results.
        
        Args:
            stats1 (Dict): First set of rendering statistics
            stats2 (Dict): Second set of rendering statistics
            name1 (str): Name/label for the first set of statistics (default: "Original")
            name2 (str): Name/label for the second set of statistics (default: "Modified")
        
        The function prints a comparison of PSNR, SSIM, and LPIPS metrics for each frame
        and the average across all frames, showing the difference between the two sets.
        """
        self.logger.info(f"\n=== Rendering Comparison: {name1} vs {name2} ===")
        # frame_ids are common in both stats1 and stats2
        frame_ids1 = set(stats1.keys())
        frame_ids2 = set(stats2.keys())
        common_frame_ids = frame_ids1.intersection(frame_ids2)
        if len(common_frame_ids) < len(frame_ids1) or len(common_frame_ids) < len(frame_ids2):
            self.logger.warning("Warning: The two stats have different frame IDs. Only common frames will be compared.")
            common_frame_ids.remove("average")  # Because frame_ids are not matched, so 'average' cannot be compared.
            common_frame_ids = sorted(list(common_frame_ids))
        
        for frame_id in common_frame_ids:
            if frame_id == "average":
                self.logger.info("\n=== Average Metrics Comparison ===")
            else:
                self.logger.info(f"\n=== Frame {frame_id} Comparison ===")
            
            metrics1 = stats1[frame_id]
            metrics2 = stats2[frame_id]
            
            for metric in ["psnr", "ssim", "lpips"]:
                value1 = metrics1[metric]
                value2 = metrics2[metric]
                diff = value2 - value1
                diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
                
                self.logger.info(f"{metric.upper():<8}: {value1:.4f} -> {value2:.4f} ({diff_str})")
