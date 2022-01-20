#!/bin/bash

#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --mem=10g
#SBATCH -c 4
#SBATCH --partition=t4v2

source ~/envnew
CUDA_VISIBLE_DEVICES=0

python steal.py --dataset 'cifar10' --watermark 'True' --arch 'resnet18' --archstolen 'resnet18' --epochstrain 100 --epochs 100 --force 'True' --num_queries 50000 --losstype 'softnn'
python linear_eval.py --dataset 'cifar10' --dataset-test 'cifar10' --arch 'resnet18' --epochstrain 100 --watermark 'True' --num_queries 50000 --losstype 'softnn'
