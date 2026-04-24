#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
python task1.py --input_path validation_folder/images --output ./result_task1_val.json
python ComputeFBeta/ComputeFBeta.py --preds result_task1_val.json --groundtruth validation_folder/ground-truth.json
python task1.py --input_path test_folder/images --output ./result_task1.json
python task2.py --input_path faceCluster_5 --num_cluster 5
python utils.py --ubit $1

# For visualization
python visualize.py --task1_val result_task1.json --task2 result_task2.json --img_dir test_folder/images --cluster_dir faceCluster_5
python utils.py --ubit $1