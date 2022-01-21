# Uses downloaded and converted Imagenet Resnet 50 model https://github.com/tonylins/simclr-converter https://github.com/google-research/simclr
# Evaluate the performance of the trained representation model on various downstream tasks.

import torch
import torch.nn as nn
import sys
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torchvision
import argparse
from torch.utils.data import DataLoader
from models.resnet_wider import resnet50rep, resnet50rep2, resnet50x1
from models.resnet_simclr import SimSiam
import torchvision.transforms as transforms
from torchvision import models
import logging
from torchvision import datasets
import random


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='/ssd003/home/akaleem/data',
                    help='path to dataset')
parser.add_argument('--dataset', default='cifar10',
                    help='dataset name', choices=['stl10', 'cifar10', 'svhn', 'imagenet'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
parser.add_argument('--archstolen', default='resnet18')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochstrain', default=200, type=int, metavar='N',
                    help='number of epochs victim was trained with')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=512, type=int,
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
parser.add_argument('--folder_name', default='resnet18_100-epochs_cifar10',
                    type=str, help='Pretrained SimCLR model to steal.')
parser.add_argument('--logdir', default='test', type=str,
                    help='Log directory to save output to.')
parser.add_argument('--losstype', default='infonce', type=str,
                    help='Loss function to use (softce or infonce)', choices=['softce', 'infonce'])

args = parser.parse_args()


class ResNet50(nn.Module):
    def __init__(self, pretrained, num_classes=10):
        super(ResNet50, self).__init__()
        self.pretrained = pretrained
        self.fc = nn.Sequential(nn.Linear(512 * 4 * 1, 512*4), nn.ReLU(), nn.Linear(512 * 4* 1, num_classes))

    def forward(self, x):
        x = self.pretrained(x)
        x = self.fc(x)
        return x

# class ResNet50v2(nn.Module):
#     def __init__(self, pretrained, num_classes=10):
#         super(ResNet50v2, self).__init__()
#         self.pretrained = pretrained
#
#     def forward(self, x):
#         x = self.pretrained(x)
#         return x
#
# class ResNet50v3(nn.Module): # For SimSiam
#     # with SimSiam
#     def __init__(self, pretrained, num_classes=10):
#         super(ResNet50v3, self).__init__()
#         self.pretrained = pretrained
#         self.num_classes = num_classes
#         self.fc = nn.Sequential(nn.Linear(512 * 4 * 1, num_classes))
#
#     def forward(self, x):
#         x = self.pretrained.encoder(x)
#         x = self.fc(x)
#         return x


#victim_model = resnet50x1().to(device)
#####
# victim_model = resnet50rep().to(device)
# checkpoint = torch.load(
#         f'/ssd003/home/akaleem/SimCLR/models/resnet50-1x.pth', map_location=device)
# state_dict = checkpoint['state_dict']
# victim_model.load_state_dict(state_dict, strict=False)
# victim_model = ResNet50(pretrained=victim_model).to(device)
#####
#victim_model = resnet50rep2().to(device)
# victim_model = SimSiam(models.__dict__["resnet50"], args.out_dim, args.out_dim).to(device)
# victim_model.encoder.fc = nn.Identity()

# checkpoint = torch.load(
#         f'/ssd003/home/akaleem/SimCLR/models/resnet50SimSiam.pth.tar', map_location=device)
# state_dict = checkpoint['state_dict']
# print("state dict", state_dict.keys())
# new_state_dict = state_dict.copy()
# # for k in state_dict.keys():
# #     if k.startswith('fc.'):
# #         del new_state_dict[k]
#
# victim_model.load_state_dict(new_state_dict, strict=False)

# if args.dataset == "imagenet":
#     victim_model = ResNet50(pretrained=victim_model, num_classes=1000).to(device)
# else:
#     victim_model = ResNet50v2(pretrained=victim_model).to(device)
#victim_model = ResNet50v3(pretrained=victim_model).to(device)

###
victim_model = models.resnet50(pretrained=True).to(device) # pretrained torch model on imagenet
victim_model.fc = nn.Sequential(nn.Linear(2048, 10)).to(device)
###

print("Loaded victim")

def get_stl10_data_loaders(download, shuffle=False, batch_size=64):
    train_dataset = datasets.STL10(f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10", split='unlabeled', download=download,
                                  transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
    test_dataset = datasets.STL10(f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10", split='test', download=download,
                                  transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def get_cifar10_data_loaders(download, shuffle=False, batch_size=64):
    train_dataset = datasets.CIFAR10('/ssd003/home/akaleem/data/', train=True, download=download,
                                  transform=transforms.ToTensor())#transforms.Compose([
    #     transforms.RandomResizedCrop(32),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.49139969, 0.48215842, 0.44653093], [0.24703223, 0.24348513, 0.26158784])
    # ]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
    test_dataset = datasets.CIFAR10('/ssd003/home/akaleem/data/', train=False, download=download,
                                  transform=transforms.ToTensor())
    # transforms.transforms.Compose([
    #     transforms.Resize(32),
    #     transforms.CenterCrop(32),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.49139969, 0.48215842, 0.44653093], [0.24703223, 0.24348513, 0.26158784])
    # ]))
    test_loader = DataLoader(test_dataset, batch_size=64,
                            num_workers=2, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def get_svhn_data_loaders(download, shuffle=False, batch_size=args.batch_size):
    train_dataset = datasets.SVHN('/ssd003/home/akaleem/data/SVHN', split='train', download=download,
                                  transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
    test_dataset = datasets.SVHN('/ssd003/home/akaleem/data/SVHN', split='test', download=download,
                                  transform=transforms.ToTensor())
    indxs = list(range(len(test_dataset) - 1000, len(test_dataset)))
    test_dataset = torch.utils.data.Subset(test_dataset,
                                           indxs)  # only select last 1000 samples to prevent overlap with queried samples.
    test_loader = DataLoader(test_dataset, batch_size=64,
                            num_workers=2, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def get_imagenet_data_loaders(download, shuffle=False, batch_size=args.batch_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocessing = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]
    imagenet_dataset = datasets.ImageNet(
        root="/scratch/ssd002/datasets/imagenet256/",   #Path for imagenet on Vector cluster
        split = "val",
        transform=transforms.Compose(preprocessing)
    )
    indxstest = random.sample(range(0, len(imagenet_dataset)), 10000)
    indxstrain = []
    for i in range(len(imagenet_dataset)):
        if i not in indxstest:
            indxstrain.append(i)
    test_dataset = torch.utils.data.Subset(imagenet_dataset,
                                           indxstest)
    train_dataset = torch.utils.data.Subset(imagenet_dataset,
                                           indxstrain)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                           drop_last=False, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            drop_last=False, shuffle=shuffle)
    return train_loader, test_loader


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if args.dataset == "cifar10":
    train_loader, test_loader = get_cifar10_data_loaders(download=False)
elif args.dataset == "svhn":
    train_loader, test_loader = get_svhn_data_loaders(download=False)
elif args.dataset == "stl10":
    train_loader, test_loader = get_stl10_data_loaders(download=False)
elif args.dataset == "imagenet":
    _, test_loader = get_imagenet_data_loaders(download=False)
    train_loader = []
    # train, test both from eval for faster runs

for name, param in victim_model.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False
parameters = list(filter(lambda p: p.requires_grad, victim_model.parameters()))
print(f"Retraining {len(parameters)} parameters.")

# optimizer = torch.optim.Adam(victim_model.parameters(), lr=args.lr,
#                                  weight_decay=args.weight_decay)
optimizer = torch.optim.SGD(victim_model.parameters(), lr=args.lr,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)
criterion = torch.nn.CrossEntropyLoss().to(device)

# Train last layer for the specific downstream task

for epoch in range(args.epochs):
    top1_train_accuracy = 0# torch.tensor(0.0)
    counter = 0
    for counter, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = victim_model(x_batch)
        loss = criterion(logits, y_batch)
        top1 = accuracy(logits, y_batch, topk=(1,))
        top1_train_accuracy += top1[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    top1_train_accuracy /= (counter + 1)
    top1_accuracy = 0
    top5_accuracy = 0
    for counter, (x_batch, y_batch) in enumerate(test_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = victim_model(x_batch)

        top1, top5 = accuracy(logits, y_batch, topk=(1,5))
        top1_accuracy += top1[0]
        top5_accuracy += top5[0]

    top1_accuracy /= (counter + 1)
    top5_accuracy /= (counter + 1)
    print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")