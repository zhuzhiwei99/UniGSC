#!/bin/bash
###
 # @Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
 # @Date: 2025-07-04 00:41:56
 # @LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
 # @LastEditTime: 2025-09-28 23:42:42
 # @FilePath: /VGSC/scripts/codec_with_1_ply_dir_configs.sh
 # @Description: 
 # 
 # Copyright (c) 2025 by Zhiwei Zhu (zhuzhiwei21@zju.edu.cn), All Rights Reserved. 
### 

# 检查参数数量
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <codec_type> <config_dir> <ply_dir>"
    echo "  codec_type: type of codec to use (eg. \"vgsc\" or \"gpcc\")"
    echo "  config_dir: path to the directory containing configuration files"
    echo "  ply_dir: path to the directory containing configuration files"
    echo "Example: $0 1 bartender configs/video_ffmpeg/anchor_0.0"
    exit 1
fi

codec_type=$1
config_dir=$2
ply_dir=$3

PLY_DIR=$ply_dir
RESULT_DIR=$(echo "$PLY_DIR" | sed 's|^data|results|')/${config_dir}
DATA_DIR=null

# Function to run a single experiment
run_experiment() {
    local gpu_id=$1
    local rp_id=$2
    
    echo "Starting experiment ${rp_id} on GPU ${gpu_id}"
    
    CUDA_VISIBLE_DEVICES=${gpu_id} python gs_pipeline.py \
        $codec_type \
        --config ${config_dir}/${rp_id}.yaml \
        --pipe_stage codec \
        --data_factor 1 \
        --data_dir $DATA_DIR \
        --ply_dir $PLY_DIR \
        --result_dir ${RESULT_DIR}/${rp_id} \
        --lpips_net vgg \
        --no-normalize_world_space \
        --scene_type GSC \
        --frame_num 1 \
        --codec.gop_size 1 \
        --codec.save_rec_ply 

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

## Launch experiments in parallel
for i in {1..4}; do
    gpu_id=$(get_best_gpu)
    rp_id=$(printf "rp%02d" $i)  # Format rp_id as rp01, rp02, etc.
    if [ "$gpu_id" -eq -1 ]; then
        echo "No available GPU found, skipping experiment ${rp_id}"
        continue
    fi
    run_experiment ${gpu_id} $rp_id &
    sleep 10  # sleep to avoid overwhelming the GPU scheduler
done

# Wait for all background processes to complete
wait

echo "All experiments completed"