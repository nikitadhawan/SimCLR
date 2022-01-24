#!/bin/bash

#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --mem=40g
#SBATCH -c 4
#SBATCH --partition=t4v2

source ~/envnew
CUDA_VISIBLE_DEVICES=0
#python stealsimsiam.py --world-size -1 --rank 0  --pretrained models/checkpoint_0099-batch256.pth.tar --data /home/nicolas/data/imagenet --batch-size 64 --lars --losstype 'infonce'
python stealsimsiam.py --world-size -1 --rank 0  --pretrained models/checkpoint_0099-batch256.pth.tar --data /home/nicolas/data/imagenet --batch-size 64 --losstype 'softnn' --datasetsteal 'imagenet' --temperaturesn 1000 --workers 4 --num_queries 50000 --useval 'True'
