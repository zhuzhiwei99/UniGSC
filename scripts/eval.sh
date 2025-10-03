#!/bin/bash
###
 # @Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
 # @Date: 2025-07-04 00:41:56
 # @LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
 # @LastEditTime: 2025-10-03 01:33:19
 # @FilePath: /UniGSC/scripts/eval.sh
 # @Description: 
 # 
 # Copyright (c) 2025 by Zhiwei Zhu (zhuzhiwei21@zju.edu.cn), All Rights Reserved. 
### 

# check if the correct number of arguments is provided
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <frame_num> <scene_type> <data_dir> <ply_dir> <result_dir>"
    echo "  frame_num: number of frames to process (eg. 1, 2, 3, etc.)"
    echo "  scene_type: type of scene (e.g., 'gsc_dynamic' for an MPEG GSC dynamic scene, 'gsc_static' for an MPEG GSC static scene)"
    echo "  data_dir:  directory containing images and colmap data (eg. 'data/GSC_splats/m71763_bartender_stable/colmap_data')"
    echo "  ply_dir: directory containing PLY files (eg. 'data/GSC_splats/m71763_bartender_stable/track')"
    echo "  result_dir: path to the directory containing compressed results"
    echo "Example: bash $0 1 gsc_dynamic data/GSC_splats/m71763_bartender_stable/colmap_data data/GSC_splats/m71763_bartender_stable/track results/GSC_splats/m71763_bartender_stable/track/frame1/configs/ffmpeg/anchor_0.0"
    exit 1
fi

FRAME_NUM=$1
SCENE_TYPE=$2
DATA_DIR=$3
PLY_DIR=$4
RESULT_DIR=$5

ORI_RENDER_DIR=$(echo "$PLY_DIR" | sed 's|^data|renders/gsplat|')/frame${FRAME_NUM}

# Default all views as test views, if needed, specify test views via --test_view_id in run_experiment function, e.g., --test_view_id 0 1 2 3 4 5 6 7

# Function to run a single experiment
run_experiment() {
    local gpu_id=$1
    local rp_id=$2
    
    echo "Starting experiment ${rp_id} on GPU ${gpu_id}"
    
    CUDA_VISIBLE_DEVICES=${gpu_id} python gs_pipeline.py \
        vgsc \
        --pipe_stage eval \
        --data_factor 1 \
        --data_dir $DATA_DIR \
        --ply_dir $PLY_DIR \
        --ori_render_dir ${ORI_RENDER_DIR} \
        --result_dir ${RESULT_DIR}/${rp_id} \
        --lpips_net vgg \
        --no-normalize_world_space \
        --scene_type $SCENE_TYPE \
        --frame_num ${FRAME_NUM} \

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
python utils/summary/summarize_stats.py --results_dir $RESULT_DIR 