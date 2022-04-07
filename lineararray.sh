#!/bin/bash

#SBATCH --job-name=lineval
#SBATCH --array=0-3
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH -c 3
#SBATCH --partition=t4v2
#SBATCH --output=/checkpoint/$USER/%A/%a.out

source ~/envnew
CUDA_VISIBLE_DEVICES=0
python linear_eval_batch.py --modeltype 'victim' --jobid $SLURM_ARRAY_JOB_ID --arrayid $SLURM_ARRAY_TASK_ID