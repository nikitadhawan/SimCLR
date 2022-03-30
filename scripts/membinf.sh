#!/bin/bash

#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
#SBATCH -c 2
#SBATCH --partition=t4v2
#SBATCH --job-name=membinf
#SBATCH --output=membinf.out

source ~/envnew
CUDA_VISIBLE_DEVICES=0
python membershipinference.py --maxlim 1000 --similarity 'mse' --n-augmentations 10 --dataset 'imagenet'
python membershipinference.py --maxlim 1001 --similarity 'cosine' --n-augmentations 10 --dataset 'imagenet'
