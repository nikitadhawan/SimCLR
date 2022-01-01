# Steals the head when only given access to the representations.
# This file first load the stolen model trained without the projection
# head. It adds in the projection head to the architecture. Then
# the model is trained in a standard way with the SimCLR method
# with every layer but the last fc (which includes the head)
# frozen.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import sys
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torchvision
import argparse
from torch.utils.data import DataLoader
from models.resnet_simclr import ResNetSimCLR
import torchvision.transforms as transforms
import logging
from torchvision import datasets
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset, \
    RegularDataset
from utils import save_config_file, accuracy, save_checkpoint





parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='/ssd003/home/akaleem/data',
                    help='path to dataset')
parser.add_argument('-dataset', default='cifar10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=200, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--num_queries', default=9000, type=int, metavar='N',
                    help='Number of queries to steal the model.')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--losstype', default='infonce', type=str,
                    help='Loss function to use')


def info_nce_loss(features):
    n = int(features.size()[0] / args.batch_size)
    labels = torch.cat(
        [torch.arange(args.batch_size) for i in range(n)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(
        similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape
    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(
        similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(
        device)
    logits = logits / args.temperature
    # print("labels", torch.sum(labels))
    # print("logits",logits)
    return logits, labels



if __name__ == "__main__":
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    dataset = RegularDataset(args.data)
    query_dataset = dataset.get_test_dataset(args.dataset, args.n_views)
    indxs = list(range(0, len(query_dataset) - 1000 * args.n_views - (9000-args.num_queries) * args.n_views))
    query_dataset = torch.utils.data.Subset(query_dataset,
                                            indxs)  # query set (without last 1000 samples in the test set)

    query_loader = torch.utils.data.DataLoader(
        query_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)


    print("Loading stolen model: ")

    stolen_model = ResNetSimCLR(base_model=args.arch,
                                out_dim=args.out_dim, include_mlp = True).to(device)
    checkpoint = torch.load(
    f'/ssd003/home/akaleem/SimCLR/runs/test/stolen_checkpoint_{args.epochs}_{args.losstype}.pth.tar', map_location=device)
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('backbone.'):
            if k.startswith('backbone') and not k.startswith('backbone.fc'):
                # remove prefix
                state_dict[k[len("backbone."):]] = state_dict[k]
        del state_dict[k]

    stolen_model.load_state_dict(state_dict, strict=False)
    # freeze all layers but the last two i.e. the head and the final layer
    for name, param in stolen_model.named_parameters():
        if name not in ['backbone.fc.0.weight', 'backbone.fc.2.weight', 'backbone.fc.0.bias', 'backbone.fc.2.bias']:
            param.requires_grad = False
    print("Loaded model")
    log_dir = f"/checkpoint/{os.getenv('USER')}/SimCLR/{args.epochs}{args.arch}STEALHEAD/"
    if os.path.exists(os.path.join(log_dir, 'training.log')):
        os.remove(os.path.join(log_dir, 'training.log'))
    else:
        try:
            os.mkdir(log_dir)
        except:
            raise Exception(f"Error creating directory at {log_dir}")
    logging.basicConfig(
        filename=os.path.join(log_dir, 'training.log'),
        level=logging.DEBUG)
    scaler = GradScaler(enabled=args.fp16_precision)

    save_config_file(log_dir,args)
    n_iter = 0
    logging.info(f"Start SimCLR training for {args.epochs} epochs.")
    logging.info(f"Training with gpu: {torch.cuda.is_available()}.")
    optimizer = torch.optim.Adam(stolen_model.parameters(), args.lr,
                                 weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(
        query_loader), eta_min=0,last_epoch=-1)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for epoch_counter in range(args.epochs):
        for images, _ in tqdm(query_loader):
            images = torch.cat(images, dim=0)

            images = images.to(device)

            features = stolen_model(images)
            logits, labels = info_nce_loss(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            n_iter += 1

        # warmup for the first 10 epochs
        if epoch_counter >= 10:
            scheduler.step()
        logging.debug(
            f"Epoch: {epoch_counter}\tLoss: {loss}\t")

    logging.info("Training has finished.")
    #save model checkpoints
    checkpoint_name = f'{args.dataset}_checkpoint_{args.epochs}_head.pth.tar'
    save_checkpoint({
        'epoch': args.epochs,
        'arch': args.arch,
        'state_dict': stolen_model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, is_best=False,
        filename=os.path.join(log_dir, checkpoint_name))
    logging.info(
        f"Model checkpoint and metadata has been saved at {log_dir}")