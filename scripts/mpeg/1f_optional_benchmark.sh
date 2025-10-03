#!/bin/bash
###
 # @Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
 # @Date: 2025-09-30 23:56:15
 # @LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
 # @LastEditTime: 2025-10-03 23:48:43
 # @FilePath: /UniGSC/scripts/mpeg/1f_optional_benchmark.sh
 # @Description: 
 # 
 # Copyright (c) 2025 by Zhiwei Zhu, All Rights Reserved. 
### 


# --- Argument check ---
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <CODEC_TYPE> <CONFIG_DIR>"
    echo "  CODEC_TYPE: codec type (e.g., 'vgsc' or 'gpcc')"
    echo "  CONFIG_DIR: path to configuration directory (e.g., 'configs/mpeg/151/video//video_anchor_ctc')"
    echo "Example: $0 vgsc configs/mpeg/151/video_anchor_ctc"
    exit 1
fi

CODEC_TYPE=$1
CONFIG_DIR=$2

FRAME_NUM=1
RENDER=gsplat  # Currently only supports "gsplat", #TODO: add "mpeg-3d-renderer" or "mpeg-gsc-metrics"
forward_facing_seq=""  # e.g., "bartender cinema" in m73341_pruned_sequences    
object_centric_seq="Cricket_player LEGO_Bugatti LEGO_Ferrari Plant Solo_Tango_Female Solo_Tango_Male Tango_duo Tennis_player"

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
    local gpu_id=$1 rp_id=$2 scene_type=$3 ply_dir=$4 data_dir=$5 lpips_net=$6
    local result_dir="results/${ply_dir#data}/frame${FRAME_NUM}/${CONFIG_DIR}"
    local render_dir="renders/${RENDER}${ply_dir#data}/frame${FRAME_NUM}"

    echo "[INFO] Running $rp_id on GPU $gpu_id"

    CUDA_VISIBLE_DEVICES=$gpu_id python gs_pipeline.py \
        $CODEC_TYPE \
        --config ${CONFIG_DIR}/${rp_id}.yaml \
        --pipe_stage benchmark \
        --data_factor 1 \
        --data_dir $data_dir \
        --ply_dir $ply_dir \
        --result_dir ${result_dir}/${rp_id} \
        --lpips_net $lpips_net \
        --no-normalize_world_space \
        --scene_type $scene_type \
        --frame_num $FRAME_NUM
}

# --- Launch all experiments for a sequence ---
run_sequence() {
    local seq=$1 scene_type=$2 ply_dir=$3 data_dir=$4 lpips_net=$5

    echo "[INFO] Starting sequence: $seq"

    for i in {1..4}; do
        gpu_id=$(get_best_gpu)
        rp_id=$(printf "rp%02d" $i)

        if [ -z "$gpu_id" ]; then
            echo "[WARN] No GPU available, skipping $rp_id"
            continue
        fi

        run_experiment $gpu_id $rp_id $scene_type $ply_dir $data_dir $lpips_net &
        sleep 10  # prevent GPU scheduler overload
    done

    wait
    echo "[INFO] Sequence $seq completed"
    
    python utils/summary/summarize_stats.py --results_dir $(echo "$ply_dir" | sed 's|^data|results|')/${CONFIG_DIR}
}

# --- Main loop ---
for seq in $forward_facing_seq; do
    run_sequence "$seq" "gsc_dynamic" \
        "data/GSC_splats/m73341_pruned_sequences/${seq}/track_pruned_90/" \
        "data/GSC_splats/m71763_${seq}_stable/colmap_data" \
        "vgg"
done

for seq in $object_centric_seq; do 
    run_sequence "$seq" "gsc_static" \
        "data/GSC_splats/humans_and_objects_1f/${seq}" \
        "data/GSC_splats/humans_and_objects_1f/${seq}/colmap_SFM" \
        "alex"
done