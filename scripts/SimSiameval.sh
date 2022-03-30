#!/bin/bash

#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --mem=20g
#SBATCH -c 4
#SBATCH --partition=t4v2
#SBATCH --job-name=evalinfonce
#SBATCH --output=evalinfonce.out

source ~/envnew
CUDA_VISIBLE_DEVICES=0,1
#(while true; do nvidia-smi; top -b -n 1 | head -20; sleep 10; done) & python linsimsiam.py --world-size -1 --rank 0 --batch-size 256 --lars --lr 1.0 --datasetsteal 'cifar10' --num_queries 50000 --losstype 'mse' --dataset 'stl10' --modeltype 'stolen' --worker 4
python linsimsiam.py --world-size -1 --rank 0 --batch-size 256 --lars --lr 1.0 --datasetsteal 'cifar10' --num_queries 50000 --losstype 'infonce' --dataset 'cifar100' --modeltype 'stolen' --worker 4
python linsimsiam.py --world-size -1 --rank 0 --batch-size 256 --lars --lr 1.0 --datasetsteal 'imagenet' --num_queries 50000 --losstype 'mse' --dataset 'cifar100' --modeltype 'stolen' --worker 4
python linsimsiam.py --world-size -1 --rank 0 --batch-size 256 --lars --lr 1.0 --datasetsteal 'imagenet' --num_queries 50000 --losstype 'mse' --dataset 'cifar10' --modeltype 'stolen' --worker 4
python linsimsiam.py --world-size -1 --rank 0 --batch-size 256 --lars --lr 1.0 --datasetsteal 'imagenet' --num_queries 50000 --losstype 'mse' --dataset 'stl10' --modeltype 'stolen' --worker 4
#python linsimsiam.py --world-size -1 --rank 0 --batch-size 256 --lars --lr 1.0 --datasetsteal 'imagenet' --num_queries 100000 --losstype 'infonce' --dataset 'cifar10' --modeltype 'stolen' --worker 4
#python linsimsiam.py --world-size -1 --rank 0 --batch-size 256 --lars --lr 1.0 --datasetsteal 'imagenet' --num_queries 100000 --losstype 'infonce' --dataset 'cifar100' --modeltype 'stolen' --worker 4
#python linsimsiam.py --world-size -1 --rank 0 --batch-size 256 --lars --lr 1.0 --datasetsteal 'cifar10' --num_queries 50000 --losstype 'infonce' --dataset 'svhn' --modeltype 'stolen' --worker 2