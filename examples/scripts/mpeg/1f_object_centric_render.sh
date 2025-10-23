#!/bin/bash
###
 # @Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
 # @Date: 2025-09-30 23:56:15
 # @LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
 # @LastEditTime: 2025-10-23 10:45:25
 # @FilePath: /UniGSC/examples/scripts/mpeg/1f_object_centric_render.sh
 # @Description: 
 # 
 # Copyright (c) 2025 by Zhiwei Zhu, All Rights Reserved. 
### 

FRAME_NUM=1
RENDER=gsplat  # Currently only supports "gsplat", #TODO: add "mpeg-3d-renderer" or "mpeg-gsc-metrics"
dynamic_object_centric_seq="fruit" 
static_object_centric_seq="Cricket_player LEGO_Bugatti LEGO_Ferrari Plant Solo_Tango_Female Solo_Tango_Male Tango_duo Tennis_player"

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
    local gpu_id=$1 scene_type=$2 ply_dir=$3 data_dir=$4 lpips_net=$5
    local result_dir="renders/${RENDER}${ply_dir#data}/frame${FRAME_NUM}"

    echo "[INFO] Rendering $ply_dir on GPU $gpu_id"

    CUDA_VISIBLE_DEVICES=$gpu_id python gs_pipeline.py \
        vgsc \
        --pipe_stage render \
        --data_factor 1 \
        --data_dir $data_dir \
        --ply_dir $ply_dir \
        --result_dir $result_dir \
        --lpips_net $lpips_net \
        --no-normalize_world_space \
        --scene_type $scene_type \
        --frame_num $FRAME_NUM
}


# --- Main loop ---
for seq in $dynamic_object_centric_seq; do
    wait_for_free_slot
    run_experiment "$(get_best_gpu)" "gsc_dynamic" \
        "data/GSC_splats/m71903_bust_dataset/trained_models/${seq}" \
        "data/GSC_splats/m71903_bust_dataset/colmap_data/${seq}" \
        "vgg" &
    sleep $sleep_interval
done
wait
for seq in $static_object_centric_seq; do 
    wait_for_free_slot
    run_experiment "$(get_best_gpu)" "gsc_static" \
        "data/GSC_splats/humans_and_objects_1f/${seq}" \
        "data/GSC_splats/humans_and_objects_1f/${seq}/colmap_SFM" \
        "alex" &
    sleep $sleep_interval
done
wait
echo "[INFO] All object-centric render experiments are completed."