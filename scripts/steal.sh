#!/bin/bash

#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --mem=10g
#SBATCH -c 4
#SBATCH --partition=t4v2

source ~/envnew
CUDA_VISIBLE_DEVICES=0
python steal.py --losstype 'mse' 
python linear_eval.py --dataset-test 'cifar10'
python linear_eval.py --dataset-test 'svhn'
python linear_eval.py --dataset-test 'stl10'