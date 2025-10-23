#!/bin/bash
###
 # @Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
 # @Date: 2025-09-30 23:56:15
 # @LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
 # @LastEditTime: 2025-10-11 12:31:21
 # @FilePath: /VGSC/examples/scripts/mpeg/1f_mandatory_render.sh
 # @Description: 
 # 
 # Copyright (c) 2025 by Zhiwei Zhu, All Rights Reserved. 
### 

FRAME_NUM=1
RENDER=gsplat  # Currently only supports "gsplat", #TODO: add "mpeg-3d-renderer" or "mpeg-gsc-metrics"
forward_facing_seq="bartender breakfast cinema"       
 

set -e

max_jobs=8  # Max concurrent jobs
sleep_interval=10  # Delay between launches to avoid GPU overload

# Function: wait until there is a free slot (less than max_jobs running)
wait_for_free_slot() {
    while true; do
        running_jobs=$(jobs -rp | wc -l)
        if (( running_jobs < max_jobs )); then
            break
        fi
        sleep 5
    done
}

# --- Utility: find GPU with max free memory ---
get_best_gpu() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
    | nl -v 0 \
    | sort -k2 -nr \
    | head -n1 \
    | awk '{print $1}'
}

# --- Run one experiment ---
run_experiment() {
    local gpu_id=$1 scene_type=$2 ply_dir=$3 data_dir=$4
    local result_dir="renders/${RENDER}${ply_dir#data}/frame${FRAME_NUM}"

    echo "[INFO] Rendering $ply_dir on GPU $gpu_id"

    CUDA_VISIBLE_DEVICES=$gpu_id python gs_pipeline.py \
        vgsc \
        --pipe_stage render \
        --data_factor 1 \
        --data_dir $data_dir \
        --ply_dir $ply_dir \
        --result_dir $result_dir \
        --lpips_net vgg \
        --no-normalize_world_space \
        --scene_type $scene_type \
        --frame_num $FRAME_NUM
}


# --- Main loop ---
for seq in $forward_facing_seq; do
    wait_for_free_slot
    run_experiment "$(get_best_gpu)" "gsc_dynamic" \
        "data/GSC_splats/m71763_${seq}_stable/track" \
        "data/GSC_splats/m71763_${seq}_stable/colmap_data" &
    sleep $sleep_interval

    wait_for_free_slot
    run_experiment "$(get_best_gpu)" "gsc_dynamic" \
        "data/GSC_splats/m71763_${seq}_stable/partially-track" \
        "data/GSC_splats/m71763_${seq}_stable/colmap_data" &
    sleep $sleep_interval
done


# Wait for all jobs to finish
wait
echo "[INFO] All forward-facing render experiments are completed."