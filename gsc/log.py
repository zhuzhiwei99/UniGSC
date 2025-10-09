'''
Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
Date: 2025-07-15 10:07:21
LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
LastEditTime: 2025-07-15 10:45:51
FilePath: /VGSC/vgsc/log.py
Description: 

Copyright (c) 2025 by Zhiwei Zhu (zhuzhiwei21@zju.edu.cn), All Rights Reserved. 
'''
# logger_config.py

import logging

_initialized = False

def init_logging(log_file: str, level=logging.INFO):
    global _initialized
    if _initialized:
        return
    _initialized = True

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()  # optional
        ]
    )

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
