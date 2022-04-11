# Steal linear SimCLR model (saved after lin evaluation) using standard stealing method.
# Measure the accuracy of the resulting stolen model and then compare with downstream accuracy of model that was stolen with embeddings.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset
import torch.optim as optim
import sys
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torchvision
import argparse
from torch.utils.data import DataLoader
from models.resnet import ResNetSimCLR, ResNet18, ResNet34 , ResNet50
import torchvision.transforms as transforms
import logging
from torchvision import datasets
from knockoff import train_model as trainknockoff
from knockoff import soft_cross_entropy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)


parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-folder-name', metavar='DIR', default='test',
                    help='path to dataset')
parser.add_argument('--dataset', default='cifar10',
                    help='dataset name', choices=['stl10', 'cifar10', 'svhn', 'imagenet'])
parser.add_argument('--dataset-test', default='cifar10',
                    help='dataset to run downstream task on', choices=['stl10', 'cifar10', 'svhn'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',
        choices=['resnet18', 'resnet34', 'resnet50'], help='model architecture')
parser.add_argument('-n', '--num-labeled', default=500,
                     help='Number of labeled batches to train on')
parser.add_argument('--epochstrain', default=200, type=int, metavar='N',
                    help='number of epochs victim was trained with')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of epochs stolen model was trained with')
parser.add_argument('--modeloutput', default='logits', type=str,
                    help='Type of victim model access.', choices=['logits', 'labels'])
parser.add_argument('--num_queries', default=9000, type=int, metavar='N',
                    help='Number of queries to steal the model.')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.5, type=float,
                    help='momentum')
parser.add_argument('--mode', default='standard', type=str,
                    help='Type of stealing (knockoff / standard)', choices=['knockoff', 'standard'])
parser.add_argument('--victimtype', default='simclr', type=str,
                    help='Type of victim (simclr / supervised)', choices=['simclr', 'supervised'])
args = parser.parse_args()

if args.victimtype == "supervised":
    args.arch = "resnet18"
if args.mode == "knockoff":
    args.lr = 0.01

if args.dataset_test == "cifar10":
    unlabeled_dataset = datasets.CIFAR10(f"/ssd003/home/{os.getenv('USER')}/data/", train=False, download=False,
                                      transform=transforms.transforms.Compose([
                                      transforms.ToTensor(),
                                      ]))
    indxs = list(range(len(unlabeled_dataset) - 1000, len(unlabeled_dataset)))
    test_dataset = torch.utils.data.Subset(unlabeled_dataset, indxs)
    test_loader = DataLoader(test_dataset, batch_size=64,
                            num_workers=2, drop_last=False, shuffle=False)
elif args.dataset_test == "stl10":
    test_dataset = datasets.STL10(
        f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10", split='test',
        download=False,
        transform=transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor()]))
    test_loader = DataLoader(test_dataset, batch_size=2 * 64,
                             num_workers=2, drop_last=False, shuffle=False)
    unlabeled_dataset = datasets.STL10(f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10", split='unlabeled', download=False,
                                       transform=transforms.Compose(
                                           [transforms.Resize(32),
                                            transforms.ToTensor()]))
elif args.dataset_test == "svhn":
    unlabeled_dataset = datasets.SVHN(f"/ssd003/home/{os.getenv('USER')}/data/SVHN", split='test', download=False,
                                      transform=transforms.transforms.Compose([
                                      transforms.ToTensor(),
                                      ]))
    indxs = list(range(len(unlabeled_dataset) - 1000, len(unlabeled_dataset)))
    test_dataset = torch.utils.data.Subset(unlabeled_dataset, indxs)
    test_loader = DataLoader(test_dataset, batch_size=64,
                            num_workers=2, drop_last=False, shuffle=False)

# Helper functions and classes
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

def get_prediction(model, unlabeled_dataloader):
    initialized = False
    with torch.no_grad():
        for data, _ in unlabeled_dataloader:
            data = data.cuda()
            output = model(data)
            if not initialized:
                result = output
                initialized = True
            else:
                result = torch.cat((result, output), 0)
    return result

class DatasetLabels(Dataset):
    r"""
    Subset of a dataset at specified indices and with specific labels.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels
        self.correct = 0
        self.total = 0

    def __getitem__(self, idx):
        data, raw_label = self.dataset[idx]
        label = self.labels[idx]
        # print('labels: ', label, raw_label)
        if raw_label == label:
            self.correct += 1
        self.total += 1
        return data, label

    def __len__(self):
        return len(self.labels)

class DatasetProbs(Dataset):
    r"""
    Subset of a dataset at specified indices and with specific labels.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, probs):
        self.dataset = dataset
        self.probs = probs
        self.correct = 0
        self.total = 0

    def __getitem__(self, idx):
        data, raw_prob = self.dataset[idx]
        prob = self.probs[idx]
        # print('labels: ', label, raw_label)
        # if raw_prob == prob:
        #     self.correct += 1
        self.total += 1
        return data, prob

    def __len__(self):
        # print(self.probs)
        # print(len(self.probs))
        return len(self.probs)

if args.arch == 'resnet18':
    stolen_model = ResNet18(num_classes=10)

    victim_model = ResNet18(num_classes=10)
elif args.arch == 'resnet34':
    stolen_model = ResNet34(num_classes=10).to(device)
    victim_model = ResNet34(num_classes=10).to(device)
elif args.arch == 'resnet50':
    stolen_model = ResNet50(num_classes=10)
    victim_model = ResNet50(num_classes=10)

if args.victimtype == "simclr":
    checkpoint2 = torch.load(
        f"/checkpoint/{os.getenv('USER')}/SimCLR/downstream/victim_linear_{args.dataset_test}.pth.tar")
    victim_model.load_state_dict(checkpoint2)  # load victim model
else:
    checkpoint2 = torch.load(
        f"/checkpoint/{os.getenv('USER')}/SimCLRsupervised/victim_supervised_{args.dataset_test}.pth.tar")["state_dict"]
    victim_model.load_state_dict(checkpoint2)  # load victim model
    # checkpoint2 = torch.load(
    #     f"/checkpoint/{os.getenv('USER')}/SimCLRsupervised/supervised_resnet18_cifar10_ckpt.pth")["net"]
    # new_state_dict = {}
    # for k in list(checkpoint2.keys()):
    #     if k in ["module.linear.weight", "module.linear.bias"]:
    #         new_state_dict["fc." + k[len("module.linear."):]] = checkpoint2[k]
    #     else:
    #         new_state_dict[k[len("module."):]] = checkpoint2[k]
    # victim_model.load_state_dict(new_state_dict)



victim_model.to(device)
victim_model.eval()
stolen_model.to(device)

## Evaluating victim model

with torch.no_grad():
    top1_accuracy = 0
    for counter, (x_batch, y_batch) in enumerate(test_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = victim_model(x_batch)

        top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
        top1_accuracy += top1[0]
    top1_accuracy /= (counter + 1)
    correct = 0
    total = 0
    # for batch_idx, (inputs, targets) in enumerate(test_loader):
    #     inputs, targets = inputs.to(device), targets.to(device)
    #     outputs = victim_model(inputs)
    #
    #     _, predicted = outputs.max(1)
    #     total += targets.size(0)
    #     correct += predicted.eq(targets).sum().item()
    #
    # print(f"Accuracy: {100. * correct/total}")


print(f"Victim accuracy: {top1_accuracy} %")

# stolen dataset formed by querying the victim model
if args.dataset_test == "stl10":
    unlabeled_subset = Subset(unlabeled_dataset,
                              list(range(0, 9000)))
    unlabeled_subloader = DataLoader(
        unlabeled_subset,
        batch_size=64,
        shuffle=False)
else:
    unlabeled_subset = Subset(unlabeled_dataset, list(range(0, len(unlabeled_dataset) - 1000)))
    unlabeled_subloader = DataLoader(
                    unlabeled_subset,
                    batch_size=64,
                    shuffle=False)
predicted_logits = get_prediction(victim_model, unlabeled_subloader)
all_labels = predicted_logits.argmax(axis=1).cpu()
all_probs = F.softmax(predicted_logits, dim=1).cpu().detach()
adaptive_dataset = DatasetProbs(unlabeled_subset, all_probs)
adaptive_dataset2 = DatasetLabels(unlabeled_subset, all_labels)

# Dataloaders to train the stolen model
adaptive_loader = DataLoader(
    adaptive_dataset,
    batch_size=64,
    shuffle=False)

adaptive_loader2 = DataLoader(
    adaptive_dataset2,
    batch_size=64,
    shuffle=False)

if args.mode == "knockoff":
    if args.modeloutput == "logits":
        optimizer = optim.SGD(stolen_model.parameters(), 0.1)
        trainknockoff(stolen_model, adaptive_dataset,
                      batch_size=64,
                      criterion_train=soft_cross_entropy,
                      criterion_test=soft_cross_entropy,
                      testloader=test_loader,
                      device=device, num_workers=2, lr=args.lr,
                      momentum=args.momentum, lr_step=30, lr_gamma=0.1,
                      epochs=100, log_interval=100,
                      checkpoint_suffix='', optimizer=optimizer,
                      scheduler=None,
                      writer=None, victimmodel=victim_model)

    else:
        optimizer = optim.SGD(stolen_model.parameters(), 0.1)
        trainknockoff(stolen_model, adaptive_dataset,
                      batch_size=64,
                      testloader=test_loader,
                      device=device, num_workers=2, lr=args.lr,
                      momentum=args.momentum, lr_step=30, lr_gamma=0.1,
                      epochs=100, log_interval=100,
                      checkpoint_suffix='', optimizer=optimizer,
                      scheduler=None,
                      writer=None, victimmodel=victim_model)
else:
    optimizer = torch.optim.Adam(stolen_model.parameters(), lr=args.lr,
                                 weight_decay=0.0008)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    for epoch in range(args.epochs):
        top1_train_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(adaptive_loader2):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = stolen_model(x_batch)
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

            logits = stolen_model(x_batch)

            top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]

        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)
        print(
            f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")




