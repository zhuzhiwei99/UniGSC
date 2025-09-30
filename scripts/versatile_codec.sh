#!/bin/bash
###
 # @Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
 # @Date: 2025-07-04 00:41:56
 # @LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
 # @LastEditTime: 2025-09-26 20:25:45
 # @FilePath: /VGSC/scripts/versatile_codec.sh
 # @Description: 
 # 
 # Copyright (c) 2025 by Zhiwei Zhu (zhuzhiwei21@zju.edu.cn), All Rights Reserved. 
### 

# 检查参数数量
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <frame_num> <data_dir> <ply_dir> <codec_type> <config_dir>"
    echo "  frame_num: number of frames to process (eg. 1, 2, 3, etc.)"
    echo "  data_dir:  directory containing images and colmap data (eg. 'data/GSC_splats/m71763_bartender_stable/colmap_data')"
    echo "  ply_dir: directory containing PLY files (eg. 'data/GSC_splats/m71763_bartender_stable/track')"
    echo "  codec_type: type of codec to use (eg. 'vgsc' or 'gpcc')"
    echo "  config_dir: path to the directory containing configuration files"
    echo "Example: $0 1 bartender configs/video_ffmpeg/anchor_0.0"
    exit 1
fi

frame_num=$1
data_dir=$2
ply_dir=$3
codec_type=$4
config_dir=$5

DATA_DIR=$data_dir
PLY_DIR=$ply_dir 
RESULT_DIR=$(echo "$PLY_DIR" | sed 's|^data|results|')/frame${frame_num}/${config_dir}

if [[ "$DATA_DIR" == *"bartender"* ]]; then
    test_view_id="$(echo {0..20})"
elif [[ "$DATA_DIR" == *"breakfast"* ]]; then
    test_view_id="$(echo {0..14})"
elif [[ "$DATA_DIR" == *"cinema"* ]]; then
    test_view_id="$(echo {0..20})"
elif [[ "$DATA_DIR" == *"fruit"* ]]; then
    test_view_id="$(echo {0..23})"
else
    echo "Unknown dataset in path: $DATA_DIR"
    exit 1
fi



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
        --frame_num ${frame_num} \
        --test_view_id ${test_view_id} \
        --codec.gop_size 16 

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

# Run the Python script to generate CSV after all experiments
python utils/summary/RD_stats_to_csv.py --results_dir ${RESULT_DIR}
python utils/summary/summarize_stats.py --results_dir $RESULT_DIR 