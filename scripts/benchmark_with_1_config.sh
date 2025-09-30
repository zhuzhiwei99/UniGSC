#!/bin/bash
###
 # @Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
 # @Date: 2025-07-04 00:41:56
 # @LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
 # @LastEditTime: 2025-10-01 00:47:07
 # @FilePath: /VGSC/scripts/benchmark_with_1_config.sh
 # @Description: 
 # 
 # Copyright (c) 2025 by Zhiwei Zhu (zhuzhiwei21@zju.edu.cn), All Rights Reserved. 
### 

# 检查参数数量
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <frame_num> <dataset> <dataset_type> <codec_type> <config_file>"
    echo "  frame_num: number of frames to process (eg. 1, 2, 3, etc.)"
    echo "  dataset: dataset name (eg. 'bartender', 'breakfast', 'cinema', 'bust')"
    echo "  dataset_type: type of dataset (eg. 'track' or 'partially-track', or 'track_pup_pruned_80_50_ply')"
    echo "  codec_type: type of codec to use (eg. 'vgsc' or 'gpcc')"
    echo "  config_file: path to the configuration file (eg. 'configs/video_ffmpeg/anchor_0.0/rp04.yaml')"
    echo "Example: $0 1 bartender configs/video_ffmpeg/anchor_0.0"
    exit 1
fi

frame_num=$1
dataset=$2
dataset_type=$3
codec_type=$4
config_file=$5

DATA_DIR=data/GSC_splats/m71763_${dataset}_stable/colmap_data 
PLY_DIR=data/GSC_splats/m71763_${dataset}_stable/${dataset_type} 
RESULT_DIR=$(echo "$PLY_DIR" | sed 's|^data|results|')/frame${frame_num}/${config_file%.*}

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

gpu_id=$(get_best_gpu)
    
echo "Starting experiment on GPU ${gpu_id}"

CUDA_VISIBLE_DEVICES=${gpu_id} python gs_pipeline.py \
    ${codec_type} \
    --config ${config_file} \
    --pipe_stage benchmark \
    --data_factor 1 \
    --data_dir $DATA_DIR \
    --ply_dir $PLY_DIR \
    --result_dir $RESULT_DIR \
    --lpips_net vgg \
    --no-normalize_world_space \
    --scene_type GSC \
    --frame_num ${frame_num} \
    --test_view_id ${test_view_id} \


python utils/summary/summarize_stats.py --results_dir $RESULT_DIR 