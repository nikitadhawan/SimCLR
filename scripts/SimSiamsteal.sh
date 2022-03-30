#!/bin/bash

#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --mem=60g
#SBATCH -c 8
#SBATCH --partition=t4v2

source ~/envnew
CUDA_VISIBLE_DEVICES=0,1,2,3
#python stealsimsiam.py --world-size -1 --rank 0  --pretrained models/checkpoint_0099-batch256.pth.tar --data /home/nicolas/data/imagenet --batch-size 64 --lars --losstype 'infonce'
python stealsimsiam.py --world-size -1 --rank 0  --pretrained models/checkpoint_0099-batch256.pth.tar --data /home/nicolas/data/imagenet --batch-size 512 --losstype 'softnn' --datasetsteal 'cifar10' --temperaturesn 10000 --workers 8 --num_queries 50000 --useaug 'True'
