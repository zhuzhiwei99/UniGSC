#!/bin/bash
###
 # @Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
 # @Date: 2025-07-04 00:41:56
 # @LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
 # @LastEditTime: 2025-08-14 10:24:29
 # @FilePath: /VGSC/scripts/render.sh
 # @Description: 
 # 
 # Copyright (c) 2025 by Zhiwei Zhu (zhuzhiwei21@zju.edu.cn), All Rights Reserved. 
### 

# 检查参数数量
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <frame_num> <dataset> <ply_dir>"
    echo "  frame_num: number of frames to process (eg. 1, 2, 3, etc.)"
    echo "  dataset: dataset name (eg. \"bartender, breakfast, cinema, bust\")"
    echo "  ply_dir: path to the directory containing Gaussian splats files"
    echo "Example: $0 1 bartender track"
    exit 1
fi

frame_num=$1
dataset=$2
ply_dir=$3

DATA_DIR=data/GSC_splats/m71763_${dataset}_stable/colmap_data 
PLY_DIR=data/GSC_splats/m71763_${dataset}_stable/${ply_dir}
RESULT_DIR=$(echo "$PLY_DIR" | sed 's|^data|renders/gsplat|')

if [ "$dataset" == "bartender" ]; then
    test_view_id="$(echo {0..20})"  
elif [ "$dataset" == "breakfast" ]; then
    test_view_id="$(echo {0..14})"  
elif [ "$dataset" == "cinema" ]; then
    test_view_id="$(echo {0..20})"
elif [ "$dataset" == "fruit" ]; then
    test_view_id="$(echo {0..23})"
else
    echo "Unknown dataset: $dataset"
    exit 1
fi


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
        --scene_type GSC \
        --frame_num ${frame_num} \
        --test_view_id ${test_view_id} \

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
