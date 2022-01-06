import torch
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


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-folder-name', metavar='DIR', default='test',
                    help='path to dataset')
parser.add_argument('-dataset', default='cifar10',
                    help='dataset name', choices=['stl10', 'cifar10', 'svhn'])
parser.add_argument('--dataset-test', default='svhn',
                    help='dataset to run downstream task on', choices=['stl10', 'cifar10', 'svhn'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
        choices=['resnet18', 'resnet34', 'resnet50'], help='model architecture')
parser.add_argument('-n', '--num-labeled', default=500,
                     help='Number of labeled batches to train on')  # Right now this is equivalent to the full training set because of the batch size being 512
parser.add_argument('--epochstrain', default=200, type=int, metavar='N',
                    help='number of epochs victim was trained with')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of epochs stolen model was trained with')
parser.add_argument('--num_queries', default=9000, type=int, metavar='N',
                    help='Number of queries to steal the model.')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='learning rate to train the model with.')
parser.add_argument('--modeltype', default='stolen', type=str,
                    help='Type of model to evaluate', choices=['victim', 'stolen'])
parser.add_argument('--save', default='False', type=str,
                    help='Save final model', choices=['True', 'False'])
parser.add_argument('--losstype', default='infonce', type=str,
                    help='Loss function to use.')
parser.add_argument('--head', default='False', type=str,
                    help='stolen model was trained using recreated head.', choices=['True', 'False'])
parser.add_argument('--clear', default='True', type=str,
                    help='Clear previous logs', choices=['True', 'False'])

args = parser.parse_args()
if args.modeltype == "stolen":
    if args.head == "False":
        log_dir = f"/checkpoint/{os.getenv('USER')}/SimCLR/{args.epochs}{args.arch}{args.losstype}STEAL/"  # save logs here.
    else:
        log_dir = f"/checkpoint/{os.getenv('USER')}/SimCLR/{args.epochs}{args.arch}STEALHEAD/"  
else:
    args.arch = "resnet34"
    log_dir = f"/checkpoint/{os.getenv('USER')}/SimCLR/{args.epochstrain}{args.arch}{args.losstype}TRAIN/"
logname = f'testing{args.modeltype}{args.dataset_test}{args.num_queries}.log'
if args.clear == "True":
    if os.path.exists(os.path.join(log_dir, logname)):
        os.remove(os.path.join(log_dir, logname))
logging.basicConfig(
    filename=os.path.join(log_dir, logname),
    level=logging.DEBUG)


def load_victim(epochs, dataset, model, loss, device):

    print("Loading victim model: ")
    checkpoint = torch.load(
        f"/checkpoint/{os.getenv('USER')}/SimCLR/{epochs}{args.arch}{loss}TRAIN/{dataset}_checkpoint_{epochs}_{loss}.pth.tar",
        map_location=device)
    # checkpoint = torch.load(
    #     f'/ssd003/home/akaleem/SimCLR/runs/{dataset}_checkpoint_{epochs}.pth.tar', map_location=device)
    state_dict = checkpoint['state_dict']

    # Remove head.
    for k in list(state_dict.keys()):
        if k.startswith('backbone.'):
            if k.startswith('backbone') and not k.startswith('backbone.fc'):
                # remove prefix
                state_dict[k[len("backbone."):]] = state_dict[k]
        del state_dict[k] # delete the head (i.e. layers that do not include backbone)

    log = model.load_state_dict(state_dict, strict=False)
    assert log.missing_keys == ['fc.weight', 'fc.bias']
    return model

def load_stolen(epochs, loss, model, device):

    print("Loading stolen model: ")

    if args.head == "False":
        checkpoint = torch.load(
            f"/checkpoint/{os.getenv('USER')}/SimCLR/{epochs}{args.arch}{loss}STEAL/stolen_checkpoint_{epochs}_{loss}.pth.tar", map_location=device)
        # checkpoint = torch.load(
        # f'/ssd003/home/akaleem/SimCLR/runs/test/stolen_checkpoint_{epochs}_{loss}.pth.tar', map_location=device)
    else:
        checkpoint = torch.load(
        f"/checkpoint/{os.getenv('USER')}/SimCLR/{epochs}{args.arch}STEALHEAD/stolen_checkpoint_{epochs}_{loss}.pth.tar", map_location=device)
    state_dict = checkpoint['state_dict']
    # Remove head.
    if loss == "symmetrized":
        for k in list(state_dict.keys()):
            if k.startswith('encoder.'):
                if k.startswith('encoder') and not k.startswith('encoder.fc'):
                    # remove prefix
                    state_dict[k[len("encoder."):]] = state_dict[k]
            del state_dict[k]
    else:
        for k in list(state_dict.keys()):
            if k.startswith('backbone.'):
                if k.startswith('backbone') and not k.startswith('backbone.fc'):
                    # remove prefix
                    state_dict[k[len("backbone."):]] = state_dict[k]
            del state_dict[k]

    log = model.load_state_dict(state_dict, strict=False)
    assert log.missing_keys == ['fc.weight', 'fc.bias']
    return model

def get_stl10_data_loaders(download, shuffle=False, batch_size=256):
    train_dataset = datasets.STL10('/ssd003/home/akaleem/data/', split='train', download=download,
                                  transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
    test_dataset = datasets.STL10('/ssd003/home/akaleem/data/', split='test', download=download,
                                  transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def get_cifar10_data_loaders(download, shuffle=False, batch_size=256):
    train_dataset = datasets.CIFAR10('/ssd003/home/akaleem/data/', train=True, download=download,
                                  transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
    test_dataset = datasets.CIFAR10('/ssd003/home/akaleem/data/', train=False, download=download,
                                  transform=transforms.ToTensor())
    indxs = list(range(len(test_dataset) - 1000, len(test_dataset)))
    test_dataset = torch.utils.data.Subset(test_dataset,
                                           indxs)  # only select last 1000 samples to prevent overlap with queried samples.
    test_loader = DataLoader(test_dataset, batch_size=64,
                            num_workers=2, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def get_svhn_data_loaders(download, shuffle=False, batch_size=256):
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



if args.arch == 'resnet18':
    model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(device)
elif args.arch == 'resnet34':
    model = torchvision.models.resnet34(pretrained=False,
                                        num_classes=10).to(device)
elif args.arch == 'resnet50':
    model = torchvision.models.resnet50(pretrained=False, num_classes=10).to(device)

if args.modeltype == "victim":
    model = load_victim(args.epochstrain, args.dataset, model, args.losstype,
                                         device=device)
    print("Evaluating victim")
else:
    model = load_stolen(args.epochs, args.losstype, model,
                        device=device)
    print("Evaluating stolen model")

if args.dataset_test == 'cifar10':
    train_loader, test_loader = get_cifar10_data_loaders(download=True)
elif args.dataset_test == 'stl10':
    train_loader, test_loader = get_stl10_data_loaders(download=True)
elif args.dataset_test == "svhn":
    train_loader, test_loader = get_svhn_data_loaders(download=False)

# freeze all layers but the last fc
for name, param in model.named_parameters():
    if name not in ['fc.weight', 'fc.bias']:
        param.requires_grad = False

parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
assert len(parameters) == 2  # fc.weight, fc.bias

if args.modeltype == "victim":
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0008)
    criterion = torch.nn.CrossEntropyLoss().to(device)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=0.0008)
    criterion = torch.nn.CrossEntropyLoss().to(device)
epochs = 100

## Trains the representation model with a linear classifier to measure the accuracy on the test set labels of the victim/stolen model

logging.info(f"Evaluating {args.modeltype} model on {args.dataset_test} dataset. Model trained using {args.losstype}.")
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

        if counter+1 == args.num_labeled:
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

if args.save == "True":
    if args.modeltype == "stolen":
        torch.save(model.state_dict(), 'runs/eval/stolen_linear.pth.tar')
    else:
        torch.save(model.state_dict(), 'runs/eval/victim_linear.pth.tar')
