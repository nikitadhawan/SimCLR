#!/bin/bash

#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --mem=20g
#SBATCH -c 4
#SBATCH --partition=t4v2

source ~/envnew
CUDA_VISIBLE_DEVICES=0
python linsimsiam.py --world-size -1 --rank 0 --batch-size 256 --lars --lr 1.0 --datasetsteal 'imagenet' --num_queries 100000 --losstype 'infonce' --dataset 'cifar10' --modeltype 'stolen' --worker 2
#python linsimsiam.py --world-size -1 --rank 0 --batch-size 256 --lars --lr 1.0 --datasetsteal 'cifar10' --num_queries 50000 --losstype 'infonce' --dataset 'svhn' --modeltype 'stolen' --worker 2