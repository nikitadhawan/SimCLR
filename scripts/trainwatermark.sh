#!/bin/bash

#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --mem=10g
#SBATCH -c 4
#SBATCH --partition=t4v2

source ~/envnew
CUDA_VISIBLE_DEVICES=0
python run.py --dataset 'cifar10' --arch 'resnet18' --epochs 50 # used as a random model for Dataset inference
python linear_eval.py --dataset 'cifar10' --dataset-test 'cifar10' --modeltype 'victim' --arch 'resnet18' --epochstrain 50
<<com
python run.py --dataset 'cifar10' --watermark 'True' --arch 'resnet18' --epochs 100
python linear_eval.py --dataset 'cifar10' --dataset-test 'cifar10' --modeltype 'victim' --arch 'resnet18' --epochstrain 100 --watermark 'True'
python linear_eval.py --dataset 'cifar10' --dataset-test 'svhn' --modeltype 'victim' --arch 'resnet18' --epochstrain 100 --watermark 'True'
python linear_eval.py --dataset 'cifar10' --dataset-test 'stl10' --modeltype 'victim' --arch 'resnet18' --epochstrain 100 --watermark 'True'

python run.py --dataset 'cifar10' --entropy 'True' --arch 'resnet18' --epochs 100 --clear 'False'
python linear_eval.py --dataset 'cifar10' --dataset-test 'cifar10' --modeltype 'victim' --arch 'resnet18' --epochstrain 100 --entropy 'True'
python linear_eval.py --dataset 'cifar10' --dataset-test 'svhn' --modeltype 'victim' --arch 'resnet18' --epochstrain 100 --entropy 'True'
python linear_eval.py --dataset 'cifar10' --dataset-test 'stl10' --modeltype 'victim' --arch 'resnet18' --epochstrain 100 --entropy 'True'
com