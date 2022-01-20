#!/bin/bash

#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --mem=10g
#SBATCH -c 4
#SBATCH --partition=t4v2

source ~/envnew
CUDA_VISIBLE_DEVICES=0
<<com
python steal.py --dataset 'cifar10' --watermark 'True' --arch 'resnet18' --archstolen 'resnet18' --epochstrain 100 --epochs 100 --losstype 'mse'
python linear_eval.py --dataset 'cifar10' --dataset-test 'cifar10' --arch 'resnet18' --epochstrain 100 --watermark 'True' --losstype 'mse' --clear 'False'

python linear_eval.py --dataset 'cifar10' --dataset-test 'svhn' --arch 'resnet18' --epochstrain 100 --watermark 'True'
python linear_eval.py --dataset 'cifar10' --dataset-test 'stl10' --arch 'resnet18' --epochstrain 100 --watermark 'True'
com

<<com3
python steal.py --dataset 'cifar10' --watermark 'True' --arch 'resnet18' --archstolen 'resnet18' --epochstrain 100 --epochs 100 --force 'True' --num_queries 50000 --losstype 'mse'
python linear_eval.py --dataset 'cifar10' --dataset-test 'cifar10' --arch 'resnet18' --epochstrain 100 --watermark 'True' --num_queries 50000 --losstype 'mse'
python steal.py --dataset 'cifar10' --watermark 'True' --arch 'resnet18' --archstolen 'resnet18' --epochstrain 100 --epochs 100 --force 'True' --num_queries 50000 --losstype 'infonce'
python linear_eval.py --dataset 'cifar10' --dataset-test 'cifar10' --arch 'resnet18' --epochstrain 100 --watermark 'True' --num_queries 50000 --losstype 'infonce'
com3


python steal.py --dataset 'cifar10' --watermark 'True' --arch 'resnet18' --archstolen 'resnet18' --epochstrain 100 --epochs 100 --force 'True' --num_queries 50000 --losstype 'softnn'
python linear_eval.py --dataset 'cifar10' --dataset-test 'cifar10' --arch 'resnet18' --epochstrain 100 --watermark 'True' --num_queries 50000 --losstype 'softnn'
