#!/usr/bin/env bash

# DA360_small
CUDA_VISIBLE_DEVICES=0 python evaluate.py --val_datasets matterport3d stanford2d3d metropolis --model_path checkpoints/DA360_small.pth --batch_size 1  --alignment 1 --model_name DA360_small --save_samples 

# DA360_base
CUDA_VISIBLE_DEVICES=0 python evaluate.py --val_datasets matterport3d stanford2d3d metropolis  --model_path checkpoints/DA360_base.pth --batch_size 1  --alignment 1 --model_name DA360_base --save_samples 

# DA360_large
CUDA_VISIBLE_DEVICES=0 python evaluate.py --val_datasets matterport3d stanford2d3d metropolis --model_path checkpoints/DA360_large.pth --batch_size 1  --alignment 1 --model_name DA360_large --save_samples 


