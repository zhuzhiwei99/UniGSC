'''
Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
Date: 2025-09-30 23:56:15
LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
LastEditTime: 2025-10-05 21:40:24
FilePath: /UniGSC/gsc/codec/__init__.py
Description: 

Copyright (c) 2025 by Zhiwei Zhu, All Rights Reserved. 
'''

import subprocess
import logging
from typing import List, Optional
import shlex

def run_command(cmd: List[str], logger: logging.Logger, log_path: Optional[str]=None) -> None:
    """
    Run a command as a subprocess and log its output.
    Args:
        cmd (List[str]): The command and its arguments to execute.
        logger (logging.Logger): Logger for logging messages.
        log_path (Optional[str]): Path to save the log file. If None, no log file is created.
    Raises:
        subprocess.CalledProcessError: If the command returns a non-zero exit status.
    """
    cmd_str = shlex.join(cmd)
    logger.info(f"Executing command: {cmd_str}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        logger.info(f"Process stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"Process stderr: {result.stderr}")
        logger.info(f"Completed successfully: {cmd_str}.")
        if log_path:
            with open(log_path, 'w') as log_file:
                log_file.write(f"--- STDOUT ---\n{result.stdout}\n")
                if result.stderr:
                    log_file.write(f"--- STDERR ---\n{result.stderr}\n")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {cmd_str}.")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"Command: {shlex.join(e.cmd)}")
        logger.error(f"Stderr: {e.stderr}")
        logger.error(f"Stdout: {e.stdout}")
        if log_path:
            with open(log_path, 'w') as log_file:
                log_file.write(f"--- STDOUT ---\n{e.stdout}\n")
                log_file.write(f"--- STDERR ---\n{e.stderr}\n")       
        raise
