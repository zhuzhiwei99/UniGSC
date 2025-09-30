#!/bin/bash
###
 # @Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
 # @Date: 2025-07-04 00:41:56
 # @LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
 # @LastEditTime: 2025-10-01 00:49:31
 # @FilePath: /VGSC/scripts/eval_with_result_dirs.sh
 # @Description: 
 # 
 # Copyright (c) 2025 by Zhiwei Zhu (zhuzhiwei21@zju.edu.cn), All Rights Reserved. 
### 

# 检查参数数量
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <frame_num> <dataset> <result_dir>"
    echo "  frame_num: number of frames to process (eg. 1, 2, 3, etc.)"
    echo "  dataset: dataset name (eg. \"bartender, breakfast, cinema, bust\")"
    echo "  result_dir: path to the directory containing compressed results"
    echo "Example: $0 1 bartender results/m71763_bartender_stable/track/frame1/configs/hm/anchor_0.0"
    exit 1
fi

frame_num=$1
dataset=$2
result_dir=$3

DATA_DIR=data/GSC_splats/m71763_${dataset}_stable/colmap_data 
PLY_DIR=data/GSC_splats/m71763_${dataset}_stable/track 
RESULT_DIR=$result_dir

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
    local rp_id=$2
    
    echo "Starting experiment ${rp_id} on GPU ${gpu_id}"
    
    CUDA_VISIBLE_DEVICES=${gpu_id} python gs_pipeline.py \
        vgsc \
        --pipe_stage eval \
        --data_factor 1 \
        --data_dir $DATA_DIR \
        --ply_dir $PLY_DIR \
        --result_dir ${RESULT_DIR}/${rp_id} \
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