#!/bin/bash
###
 # @Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
 # @Date: 2025-07-04 00:41:56
 # @LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
 # @LastEditTime: 2025-10-03 20:05:59
 # @FilePath: /UniGSC/scripts/render.sh
 # @Description: 
 # 
 # Copyright (c) 2025 by Zhiwei Zhu (zhuzhiwei21@zju.edu.cn), All Rights Reserved. 
### 

# check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <frame_num> <scene_type> <ply_dir> <data_dir>"
    echo "  frame_num: number of frames to process (eg. 1, 2, 3, etc.)"
    echo "  scene_type: type of scene (e.g., 'gsc_dynamic' for an MPEG GSC dynamic scene, 'gsc_static' for an MPEG GSC static scene)"
    echo "  ply_dir: path to the directory containing Gaussian splats ply files"
    echo "  data_dir: path to the directory containing COLMAP data"
    echo "Example: bash $0 1 gsc_dynamic data/GSC_splats/m71763_bartender_stable/track data/GSC_splats/m71763_bartender_stable/colmap_data"
    exit 1
fi

FRAME_NUM=$1
SCENE_TYPE=$2
PLY_DIR=$3
DATA_DIR=$4

RESULT_DIR=$(echo "$PLY_DIR" | sed 's|^data|renders/gsplat|')/frame$FRAME_NUM

# Default all views as test views, if needed, specify test views via --test_view_id in run_experiment function, e.g., --test_view_id 0 1 2 3 4 5 6 7

# Function to run a single experiment
run_experiment() {
    local gpu_id=$1
    
    echo "Starting experiment on GPU ${gpu_id}"
    
    CUDA_VISIBLE_DEVICES=${gpu_id} python gs_pipeline.py \
        vgsc \
        --pipe_stage render \
        --data_factor 1 \
        --data_dir $DATA_DIR \
        --ply_dir $PLY_DIR \
        --result_dir $RESULT_DIR \
        --lpips_net vgg \
        --no-normalize_world_space \
        --scene_type $SCENE_TYPE \
        --frame_num ${FRAME_NUM} 
}


# Function to automatically detect the best GPU based on available memory
get_best_gpu() {
    local best_gpu=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | awk '{print $1}' \
        | nl -v 0 \
        | sort -k2 -nr \
        | head -n1 \
        | awk '{print $1}')
    echo "$best_gpu"
}

run_experiment $(get_best_gpu)
