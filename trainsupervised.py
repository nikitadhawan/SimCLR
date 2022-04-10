import torch
import sys
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torchvision
import argparse
from torch.utils.data import DataLoader
#from models.resnet_simclr import ResNetSimCLR
from models.resnet import ResNetSimCLR, ResNet18, ResNet34 , ResNet50 # from other file
from models.resnet_wider import resnet50rep, resnet50rep2, resnet50x1
import torchvision.transforms as transforms
import logging
from torchvision import datasets


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-folder-name', metavar='DIR', default='test',
                    help='path to dataset')
parser.add_argument('--dataset', default='cifar10',
                    help='dataset name', choices=['stl10', 'cifar10', 'svhn', 'imagenet'])
parser.add_argument('--dataset-test', default='cifar10',
                    help='dataset to run downstream task on', choices=['stl10', 'cifar10', 'svhn'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
        choices=['resnet18', 'resnet34', 'resnet50'], help='model architecture')
parser.add_argument('-n', '--num-labeled', default=50000,type=int,
                     help='Number of labeled examples to train on')
parser.add_argument('--epochstrain', default=200, type=int, metavar='N',
                    help='number of epochs victim was trained with')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of epochs stolen model was trained with')
parser.add_argument('--num_queries', default=9000, type=int, metavar='N',
                    help='Number of queries to steal the model.')
parser.add_argument('--lr', default=3e-4, type=float, # maybe try other lrs
                    help='learning rate to train the model with.')
parser.add_argument('--save', default='True', type=str,
                    help='Save final model', choices=['True', 'False'])
parser.add_argument('--clear', default='False', type=str,
                    help='Clear previous logs', choices=['True', 'False'])
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
args = parser.parse_args()




def get_stl10_data_loaders(download, shuffle=False, batch_size=args.batch_size):
    train_dataset = datasets.STL10(f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10", split='train', download=download,
                                  transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
    test_dataset = datasets.STL10(f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10", split='test', download=download,
                                  transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))
    test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=2, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def get_cifar10_data_loaders(download, shuffle=False, batch_size=args.batch_size):
    train_dataset = datasets.CIFAR10(f"/ssd003/home/{os.getenv('USER')}/data/", train=True, download=download,
                                  transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
    test_dataset = datasets.CIFAR10(f"/ssd003/home/{os.getenv('USER')}/data/", train=False, download=download,
                                  transform=transforms.ToTensor())
    indxs = list(range(len(test_dataset) - 1000, len(test_dataset)))
    test_dataset = torch.utils.data.Subset(test_dataset,
                                           indxs)  # only select last 1000 samples to prevent overlap with queried samples.
    test_loader = DataLoader(test_dataset, batch_size=64,
                            num_workers=2, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def get_svhn_data_loaders(download, shuffle=False, batch_size=args.batch_size):
    train_dataset = datasets.SVHN(f"/ssd003/home/{os.getenv('USER')}/data/SVHN", split='train', download=download,
                                  transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
    test_dataset = datasets.SVHN(f"/ssd003/home/{os.getenv('USER')}/data/SVHN", split='test', download=download,
                                  transform=transforms.ToTensor())
    indxs = list(range(len(test_dataset) - 1000, len(test_dataset)))
    test_dataset = torch.utils.data.Subset(test_dataset,
                                           indxs)  # only select last 1000 samples to prevent overlap with queried samples.
    test_loader = DataLoader(test_dataset, batch_size=64,
                            num_workers=2, drop_last=False, shuffle=shuffle)
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



log_dir = f"/checkpoint/{os.getenv('USER')}/SimCLRsupervised/"
logname = f'trainingvictim{args.dataset}.log'
if args.clear == "True":
    if os.path.exists(os.path.join(log_dir, logname)):
        os.remove(os.path.join(log_dir, logname))
logging.basicConfig(
    filename=os.path.join(log_dir, logname),
    level=logging.DEBUG)

if args.arch == 'resnet18':
    model = ResNet18(num_classes=10).to(device)
elif args.arch == 'resnet34':
    model = ResNet34( num_classes=10).to(device)
elif args.arch == 'resnet50':
    model = ResNet50(num_classes=10).to(device)

if args.dataset_test == 'cifar10':
    train_loader, test_loader = get_cifar10_data_loaders(download=False)
elif args.dataset_test == 'stl10':
    train_loader, test_loader = get_stl10_data_loaders(download=False)
elif args.dataset_test == "svhn":
    train_loader, test_loader = get_svhn_data_loaders(download=False)


optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0008)
criterion = torch.nn.CrossEntropyLoss().to(device)

epochs = 100

## Trains the representation model with a linear classifier to measure the accuracy on the test set labels of the victim/stolen model

logging.info(f"Training supervised victim")
logging.info(f"Args: {args}")
for epoch in range(epochs):
    top1_train_accuracy = 0
    for counter, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        top1 = accuracy(logits, y_batch, topk=(1,))
        top1_train_accuracy += top1[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (counter+1) * x_batch.shape[0] >= args.num_labeled:
            break

    top1_train_accuracy /= (counter + 1)
    top1_accuracy = 0
    top5_accuracy = 0
    for counter, (x_batch, y_batch) in enumerate(test_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)

        top1, top5 = accuracy(logits, y_batch, topk=(1,5))
        top1_accuracy += top1[0]
        top5_accuracy += top5[0]

    top1_accuracy /= (counter + 1)
    top5_accuracy /= (counter + 1)
    print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
    logging.debug(
        f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")

torch.save(model.state_dict(), f"/checkpoint/{os.getenv('USER')}/SimCLRsupervised/victim_supervised_{args.dataset_test}.pth.tar")
