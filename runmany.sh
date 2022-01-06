#!/bin/bash

#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --mem=10g
#SBATCH -c 4
#SBATCH --partition=t4v2

source ~/envnew
CUDA_VISIBLE_DEVICES=0
for NUMQUERIES in 100 500 1000 5000 10000 20000 30000 40000 50000 ; do
  python steal.py --losstype 'softnn' --num_queries $NUMQUERIES
  python linear_eval.py --losstype 'softnn'--dataset-test 'cifar10' --num_queries $NUMQUERIES
  python linear_eval.py --losstype 'softnn' --dataset-test 'svhn' --num_queries $NUMQUERIES
done
