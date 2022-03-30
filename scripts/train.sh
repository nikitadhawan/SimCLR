#!/bin/bash

#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --mem=10g
#SBATCH -c 4
#SBATCH --partition=t4v2

source ~/envnew
CUDA_VISIBLE_DEVICES=0
python run.py --dataset 'cifar10'
python linear_eval.py --dataset 'cifar10' --dataset-test 'cifar10' --modeltype 'victim'