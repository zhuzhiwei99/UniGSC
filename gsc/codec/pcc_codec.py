''':
Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
Date: 2025-07-06 18:28:46
LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
LastEdi:Time: 2025-07-09 14:56:55
FilePath: /VGSC/vgsc/codec/pcc_codec.py
Description: A unified wrapper for Point Cloud Compression (PCC) codecs like GPCC.

Copyright (c) 2025 by Zhiwei Zhu (zhuzhiwei21@zju.edu.cn), All Rights Reserved. 
'''
import os
from pathlib import Path
import subprocess
from typing import List, Dict, Optional
from gsc.utils.file import force_make_dirs, safe_make_dirs
from gsc.utils.gs_io import load_ply
from gsc.config import QUANT_CONFIG_KEYS, DEC_SUFFIX
from gsc import get_logger
logger = get_logger("PccCodec")



class PccCodec:
    """
    An abstract class for Point Cloud Compression (PCC) codecs like Ges-TM and GPCC.
    
    This class encapsulates the encoding and decoding process, including
    pre-processing (quantization), calling the external codec, and 
    post-processing (de-quantization).
    """
    def __init__(self, 
                 encoder_path: str,
                 decoder_path: str, 
                 encode_config_path: str,
                 decode_config_path: str,
                 quant_type: Optional[QUANT_CONFIG_KEYS] = 'N00677') -> None:
        """N
        Initializes the PccCodec instance.

        Args:
            encoder_path (str): Path to the encoder executable.
            decoder_path (str): Path to the decoder executable.
            result_dir (str): Directory where results will be stored.
            encode_config_path (str): Path to the encoding configuration file.
            decode_config_path (str): Path to the decoding configuration file.
            quant_type (QUANT_CONFIG_KEYS, optional): Type of quantization to use.
            
        """
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path    
        self.encode_config_path = encode_config_path
        self.decode_config_path = decode_config_path
        self._check_file_exist()

        self.quant_type = quant_type
        

    def _check_file_exist(self):
        """Checks if the required files exist."""
        if not os.path.exists(self.encoder_path):
            raise FileNotFoundError(f"Encoder executable not found: {self.encoder_path}")
        if not os.path.exists(self.decoder_path):
            raise FileNotFoundError(f"Decoder executable not found: {self.decoder_path}")
        if not os.path.exists(self.encode_config_path):
            raise FileNotFoundError(f"Encoder configuration file not found: {self.encode_config_path}")
        if not os.path.exists(self.decode_config_path):
            raise FileNotFoundError(f"Decoder configuration file not found: {self.decode_config_path}")
        

    def _setup_directories(self, result_dir: Path) -> None:
        """Creates the working directories."""
        logger.info("Setting up working directories...")
        self.result_dir = Path(result_dir)
        self.compress_dir = self.result_dir / "compression"
        self.intermediate_dir = self.result_dir / "intermediate"
        self.log_dir = self.result_dir / "log"
        
        safe_make_dirs(self.compress_dir)
        safe_make_dirs(self.intermediate_dir)
        safe_make_dirs(self.log_dir)
        logger.info("Working directories are ready.")


    def _run_command(self, command: list, log_path: str) -> subprocess.CompletedProcess:
        """Runs a subprocess command and logs its output to a file."""
        with open(log_path, 'w') as log_file:
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True  # Raises CalledProcessError for non-zero exit codes
                )
                log_file.write("--- STDOUT ---\n")
                log_file.write(result.stdout)
                log_file.write("\n--- STDERR ---\n")
                log_file.write(result.stderr)
                return result
            except FileNotFoundError:
                logger.error(f"Command not found: {command[0]}. Ensure the path is correct and the file is executable.")
                log_file.write(f"ERROR: Command not found at {command[0]}")
                raise
            except subprocess.CalledProcessError as e:
                logger.error(f"Command '{' '.join(command)}' failed with exit code {e.returncode}")
                log_file.write("--- STDOUT ---\n")
                log_file.write(e.stdout)
                log_file.write("\n--- STDERR ---\n")
                log_file.write(e.stderr)
                raise

    def encode(self, 
               quant_ply_path: str, 
               encoded_bin_path: Path) -> None:
        """
        Encodes a single PLY file into a compressed binary format.
        
        Args:
            quant_ply_path (str): Path to the quantized PLY file.
            ply_path (str): Path to the original PLY file if quantized PLY does not exist.
            encoded_bin_path (Path): Path to save the encoded binary file.
        Raises:
            FileNotFoundError: If neither the quantized PLY nor the original PLY exists
        """
        result_dir = Path(encoded_bin_path).parent.parent
        self._setup_directories(result_dir)
        if not os.path.exists(quant_ply_path):
            raise FileNotFoundError(f"Quantized PLY file not found: {quant_ply_path}")  

        filename = os.path.basename(encoded_bin_path)
        filename = filename.split('.')[0]  # Extract frame name without extension

        # 2. Encoding
        encode_cmd = [
            self.encoder_path,
            '-c', self.encode_config_path,
            f'--uncompressedDataPath={quant_ply_path}',
            f'--compressedStreamPath={encoded_bin_path}'
        ]
        
        log_file = os.path.join(self.log_dir, f"{filename}_encode.log")
        logger.debug(f"[{filename}] Running encode command: {' '.join(encode_cmd)}")
        self._run_command(encode_cmd, log_file)
        
        logger.info(f"{filename} encoded successfully. Output: {encoded_bin_path}")

    def decode(self, encoded_bin_path: str) -> Dict:
        """
        Decodes a compressed binary file into a PLY file and de-quantizes it.
        
        Args:
            encoded_bin_path (str): Path to the compressed binary file to decode.
        Raises:
            FileNotFoundError: If the encoded binary file does not exist.
        """
        if not os.path.exists(encoded_bin_path):
            raise FileNotFoundError(f"Encoded binary file not found: {encoded_bin_path}")
        filename = os.path.basename(encoded_bin_path)
        filename = filename.split('.')[0]  # Extract frame name without extension
        # 1. Decoding
        decoded_ply_file = os.path.join(self.intermediate_dir, f"{filename}{DEC_SUFFIX}.ply")
        decode_cmd = [
            self.decoder_path,
            '-c', self.decode_config_path,
            f'--compressedStreamPath={encoded_bin_path}',
            f'--reconstructedDataPath={decoded_ply_file}'
        ]

        log_file = os.path.join(self.log_dir, f"{filename}_decode.log")
        logger.debug(f"[{filename}] Running decode command: {' '.join(decode_cmd)}")

        self._run_command(decode_cmd, log_file)
        logger.info(f"{filename} decoded successfully. Output: {decoded_ply_file}")

        # 2. De-quantization
        return load_ply(decoded_ply_file)