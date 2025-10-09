#!/bin/bash
###
 # @Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
 # @Date: 2025-07-04 00:41:56
 # @LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
 # @LastEditTime: 2025-10-02 19:17:06
 # @FilePath: /UniGSC/scripts/codec.sh
 # @Description: 
 # 
 # Copyright (c) 2025 by Zhiwei Zhu (zhuzhiwei21@zju.edu.cn), All Rights Reserved. 
### 

# check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <frame_num> <ply_dir> <codec_type> <config_dir>"
    echo "  frame_num: number of frames to process (eg. 1, 2, 3, etc.)"
    echo "  ply_dir: directory containing PLY files (eg. 'data/GSC_splats/m71763_bartender_stable/track')"
    echo "  codec_type: type of codec to use (eg. 'vgsc' or 'gpcc')"
    echo "  config_dir: path to the directory containing configuration files"
    echo "Example: bash $0 1 data/GSC_splats/m71763_bartender_stable/track vgsc configs/ffmpeg/anchor_0.0"
    exit 1
fi

FRAME_NUM=$1
PLY_DIR=$2
codec_type=$3
config_dir=$4

RESULT_DIR=$(echo "$PLY_DIR" | sed 's|^data|results|')/frame${FRAME_NUM}/${config_dir}


# Function to run a single experiment
run_experiment() {
    local gpu_id=$1
    local rp_id=$2
    
    echo "Starting experiment ${rp_id} on GPU ${gpu_id}"
    
    CUDA_VISIBLE_DEVICES=${gpu_id} python gs_pipeline.py \
        $codec_type \
        --config ${config_dir}/${rp_id}.yaml \
        --pipe_stage codec \
        --ply_dir $PLY_DIR \
        --result_dir ${RESULT_DIR}/${rp_id} \
        --frame_num ${FRAME_NUM} \
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