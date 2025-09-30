'''
Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
Date: 2025-07-06 18:28:46
LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
LastEditTime: 2025-08-12 11:45:11
FilePath: /VGSC/vgsc/codec/video_codec.py
Description: A unified wrapper for various video codecs (vvenc, VTM, HM, ffmpeg).

Copyright (c) 2025 by Zhiwei Zhu (zhuzhiwei21@zju.edu.cn), All Rights Reserved. 
'''

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from gsc import get_logger
logger = get_logger("VideoCodec")


class VideoCodec:
    """
    A generic command-line wrapper for various video codecs.

    This class provides a unified interface for different codecs (e.g., vvenc,
    VTM, HM, ffmpeg) by automatically generating the correct command-line
    arguments based on the input configuration dictionary.
    """

    def __init__(self,
                 video_codec_type: Literal['vvenc', 'vtm', 'hm', 'ffmpeg'],
                 encoder_path: str,
                 decoder_path: str,
                 encode_config_path: Optional[str] = None,
                 decode_config_path: Optional[str] = None):
        """
        Initializes the video codec wrapper.
 
        Args:
            video_codec_type: The type of video codec to use.
            encoder_path: Path to the encoder executable.
            decoder_path: Path to the decoder executable.
            encode_config_path: Path to the encoding configuration file.
            decode_config_path: Path to the decoding configuration file (optional).
        """
        self.video_codec_type = video_codec_type.lower()
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.encode_config_path = encode_config_path
        self.decode_config_path = decode_config_path
        self._check_executables()

    def _check_executables(self):
        """Ensures encoder and decoder executables exist and are accessible."""
        if self.video_codec_type == 'ffmpeg':
            # FFmpeg is assumed to be in the system PATH.
            return

        if not os.path.exists(self.encoder_path):
            raise FileNotFoundError(f"Encoder executable not found: {self.encoder_path}")
        if not os.path.exists(self.decoder_path):
            raise FileNotFoundError(f"Decoder executable not found: {self.decoder_path}")
        if not self.encode_config_path:
            raise ValueError("Encoding configuration path must be provided for HM or VTM codecs.")


    def _run_command(self, cmd: List[str], process_name: str):
        """
        Executes a subprocess command and handles logger and errors.

        Args:
            cmd: The command to execute as a list of strings.
            process_name: A descriptive name for the process (e.g., "Encoding").
        """
        logger.info(f"{process_name} with {self.video_codec_type.upper()}...")
        logger.debug(f"Executing command: {shlex.join(cmd)}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
            logger.info(f"Process stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"Process stderr: {result.stderr}")
            logger.info(f"{process_name} completed successfully.")
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"{process_name} failed.")
            logger.error(f"Return code: {e.returncode}")
            logger.error(f"Command: {shlex.join(e.cmd)}")
            logger.error(f"Stderr: {e.stderr}")
            logger.error(f"Stdout: {e.stdout}")
            raise

    # --- Command Builders: Encoder ---

    def _build_encode_cmd(self, input_path: str, output_path: str, params: Dict[str, Any]) -> List[str]:
        """Dispatches to the appropriate command builder based on codec type."""
        builders = {
            "hm": self._build_hm_encode_cmd,
            "ffmpeg": self._build_ffmpeg_encode_cmd,
        }
        return builders[self.video_codec_type](input_path, output_path, params)
    
    def _build_hm_encode_cmd(self, i_path: str, o_path: str, p: Dict[str, Any]) -> List[str]:
        """Builds the encoding command for HM (TAppEncoder)."""
        for key in ['width', 'height', 'qp', 'pix_fmt', 'frame_num', 'bit_depth']:
            if key not in p:
                raise ValueError(f"Missing required parameter for HM: '{key}'")

        pix_fmt_map = {'yuv420p': '420', 'yuv422p': '422', 'yuv444p': '444', 'yuv400p': '400'}
        chroma_format = pix_fmt_map.get(p['pix_fmt'])
        if not chroma_format:
            raise ValueError(f"Unsupported pix_fmt for HM: {p['pix_fmt']}")
        
        lossless_mode = False
        qp = p.get('qp', -1)
        if qp <= 0: 
            qp = 0  # Set qp to 0 for lossless mode
            lossless_mode = True
            logger.warning(f"qp={qp} <= 0, will use lossless mode for HM encoding.")
        if qp > 51:
            qp = 51  # Cap qp to maximum value for HM
            logger.warning(f"qp={qp} > 51, will cap qp to 51 for HM encoding.")

        config_path = self.encode_config_path
        bit_depth = p['bit_depth']
        if bit_depth < 8: 
            bit_depth = 8  # HM requires at least 8-bit depth
            logger.warning(f"bit_depth={bit_depth} < 8, will set to 8 for HM encoding.")
                        
        cmd = [
            self.encoder_path,
            '-c', config_path,
            '-i', str(i_path),
            '-b', str(o_path),
            '-fr', '30',
            '-f', str(p['frame_num']),
            '-wdt', str(p['width']),
            '-hgt', str(p['height']),
            '-q', str(qp),
            f'--InputBitDepth={bit_depth}',
            f'--InternalBitDepth={bit_depth}',
            f'--InputChromaFormat={chroma_format}', 
            f'--ChromaFormatIDC={chroma_format}',
        ]
        if lossless_mode:
            # HM lossless encoding parameters
            lossless_cmd = [
                '--CostMode=lossless',
                '--TransquantBypassEnableFlag=1',   
                '--CUTransquantBypassFlagForce=1',
                '--IntraReferenceSmoothing=0',
            ]
            cmd.extend(lossless_cmd)
        
        if 'extra_params' in p:
            cmd.extend(p['extra_params'])
        return cmd

    def _build_ffmpeg_encode_cmd(self, i_path: str, o_path: str, p: Dict[str, Any]) -> List[str]:
        """Builds the encoding command for FFmpeg (e.g., using libx265)."""
        for key in ['width', 'height', 'qp', 'pix_fmt', 'gop_size', 'use_all_intra', 'bit_depth']:
            if key not in p:
                raise ValueError(f"Missing required parameter for ffmpeg: '{key}'")

        pix_fmt_map = {'yuv420p': 'yuv420p', 'yuv422p': 'yuv422p', 'yuv444p': 'yuv444p', 'yuv400p': 'gray'}
        chroma_format = pix_fmt_map.get(p['pix_fmt'])             
        if not chroma_format:
            raise ValueError(f"Unsupported pix_fmt for ffmpeg: {p['pix_fmt']}")
        
        bit_depth = p.get('bit_depth', 8)  # Default to 8-bit if not specified
        if bit_depth in [10, 12, 14, 16]:
            chroma_format += f'{bit_depth}le'
        elif bit_depth != 8:
            raise ValueError(f"Unsupported bit depth for ffmpeg: {bit_depth}")

        # Use qp or lossless mode
        qp_params = f"qp={p['qp']}" if p['qp'] > 0 else 'lossless=1'
        cmd = [
            self.encoder_path,
            '-y',  # Overwrite output file if it exists
            '-f', 'rawvideo',
            '-pix_fmt', chroma_format,
            '-s:v', f"{p['width']}x{p['height']}",
            '-r', '30',  # Frame rate
            '-i', str(i_path),
            '-c:v', 'libx265',
            '-x265-params', qp_params,
        ]
        if p.get('use_all_intra', False):
            cmd.extend(['-g', '1'])  # GOP size of 1 means all-intra
        else:
            cmd.extend(['-g', str(p.get('gop_size', 16))])
        
        cmd.append(str(o_path))
        return cmd

    # --- Command Builders: Decoder ---

    def _build_decode_cmd(self, input_path: str, output_path: str, params: Dict[str, Any]) -> List[str]:
        """Dispatches to the appropriate decoder command builder."""
        if self.video_codec_type == 'ffmpeg':
            return self._build_ffmpeg_decode_cmd(input_path, output_path, params)
        else:
            cmd = [
                str(self.decoder_path),
                '-b', str(input_path),
                '-o', str(output_path),
            ]
            if params and 'extra_params' in params:
                cmd.extend(params['extra_params'])
            return cmd

    def _build_ffmpeg_decode_cmd(self, i_path: str, o_path: str, p: Dict[str, Any]) -> List[str]:
        """Builds the decoding command for FFmpeg."""
        pix_fmt_map = {'yuv420p': 'yuv420p', 'yuv422p': 'yuv422p', 'yuv444p': 'yuv444p', 'yuv400p': 'gray'}
        # Default to a common format if not specified
        chroma_format = pix_fmt_map.get(p.get('pix_fmt', 'yuv420p'))
        bit_depth = p.get('bit_depth', 8)  # Default to 8-bit if not specified
        if bit_depth in [10, 12, 14, 16]:
            chroma_format += f'{bit_depth}le'
        elif bit_depth != 8:
            raise ValueError(f"Unsupported bit depth for ffmpeg: {bit_depth}")

        cmd = [
            str(self.decoder_path),
            '-y',
            '-i', str(i_path),
            '-pix_fmt', chroma_format,
            str(o_path),
        ]
        return cmd

    # --- Public API ---

    def encode(self,
               input_yuv: str,
               output_bitstream: str,
               config_params: Dict[str, Any]):
        """
        Encodes a YUV file to a bitstream using the specified configuration.

        Args:
            input_yuv: Path to the input YUV file.
            output_bitstream: Path for the output bitstream file.
            config_params: A dictionary of encoding parameters. Common keys:
                'width' (int), 'height' (int), 'qp' (int),
                'pix_fmt' (str: 'yuv420p', etc.), 'frame_num' (int),
                'all_intra' (bool), 'extra_params' (List[str], optional).
        """
        cmd = self._build_encode_cmd(input_yuv, output_bitstream, config_params)
        self._run_command(cmd, f"Encoding {input_yuv.name}")

    def decode(self,
               input_bitstream: str,
               output_yuv: str,
               config_params: Dict[str, Any] = None):
        """
        Decodes a bitstream file to a YUV file.

        Args:
            input_bitstream: Path to the input bitstream file.
            output_yuv: Path for the output YUV file.
            config_params (optional): A dictionary of decoding parameters.
                Mainly used by ffmpeg to specify output format.
        """
        if config_params is None:
            config_params = {}
        cmd = self._build_decode_cmd(input_bitstream, output_yuv, config_params)
        self._run_command(cmd, f"Decoding {input_bitstream.name}")