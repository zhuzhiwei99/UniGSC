#!/bin/bash
###
 # @Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
 # @Date: 2025-09-30 23:56:15
 # @LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
 # @LastEditTime: 2025-10-03 22:02:16
 # @FilePath: /UniGSC/scripts/mpeg/main_render.sh
 # @Description: 
 # 
 # Copyright (c) 2025 by Zhiwei Zhu, All Rights Reserved. 
### 

RENDER=gsplat  # Currently only supports "gsplat", #TODO: add "mpeg-3d-renderer" or "mpeg-gsc-metrics"
forward_facing_seq="bartender breakfast cinema"       
object_centric_seq="fruit"  

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
    local gpu_id=$1 scene_type=$2 ply_dir=$3 data_dir=$4 frame_num=$5
    local result_dir="renders/${RENDER}${ply_dir#data}/frame${frame_num}"

    echo "[INFO] Running $rp_id on GPU $gpu_id"

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
        --frame_num $frame_num
}


# --- Main loop ---
for seq in $forward_facing_seq; do
    run_experiment "$(get_best_gpu)" "gsc_dynamic" \
        "data/GSC_splats/m71763_${seq}_stable/track" \
        "data/GSC_splats/m71763_${seq}_stable/colmap_data" \
        32 &
    sleep 10  # prevent GPU scheduler overload 
    run_experiment "$(get_best_gpu)" "gsc_dynamic" \
        "data/GSC_splats/m71763_${seq}_stable/partially-track" \
        "data/GSC_splats/m71763_${seq}_stable/colmap_data" \
        32 &
    sleep 10  # prevent GPU scheduler overload 
done

for seq in $object_centric_seq; do
    run_experiment "$(get_best_gpu)" "gsc_dynamic" \
        "data/GSC_splats/m71903_bust_dataset/trained_models/${seq}" \
        "data/GSC_splats/m71903_bust_dataset/colmap_data/${seq}" \
        300 &
    sleep 10  # prevent GPU scheduler overload
done
