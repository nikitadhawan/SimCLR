#!/bin/bash

#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --mem=10g
#SBATCH -c 4
#SBATCH --partition=t4v2

source ~/envnew
CUDA_VISIBLE_DEVICES=0
for NUMQUERIES in 10000 20000 30000 40000 ; do  # 50000
  python steal.py --losstype 'softnn' --dataset 'imagenet' --datasetsteal 'cifar10' --num_queries $NUMQUERIES --temperaturesn 1000
  python linear_eval.py --losstype 'softnn' --arch 'resnet50' --dataset 'imagenet' --dataset-test 'cifar10' --datasetsteal 'cifar10' --num_queries $NUMQUERIES
  python linear_eval.py --losstype 'softnn' --arch 'resnet50' --dataset 'imagenet' --dataset-test 'svhn' --datasetsteal 'cifar10' --num_queries $NUMQUERIES
  python linear_eval.py --losstype 'softnn' --arch 'resnet50' --dataset 'imagenet' --dataset-test 'stl10' --datasetsteal 'cifar10' --num_queries $NUMQUERIES
done
