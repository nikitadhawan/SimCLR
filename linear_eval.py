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
from models.convnet import ConvNet, ConvNetSimCLR
from models.resnet_wider import resnet50rep, resnet50rep2, resnet50x1
import torchvision.transforms as transforms
import logging
from torchvision import datasets
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-folder-name', metavar='DIR', default='test',
                    help='path to dataset')
parser.add_argument('--dataset', default='mixed',
                    help='dataset name', choices=['stl10', 'cifar10', 'svhn', 'imagenet', 'mixed'])
parser.add_argument('--dataset-test', default='bias',
                    help='dataset to run downstream task on', choices=['stl10', 'cifar10', 'svhn', 'emnist', 'mnist','bias'])
parser.add_argument('--datasetsteal', default='emnist',
                    help='dataset used for querying the victim', choices=['stl10', 'cifar10', 'svhn', 'imagenet', 'mixed','emnist','mnist'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='convnet',
        choices=['resnet18', 'resnet34', 'resnet50', 'convnet'], help='model architecture')
parser.add_argument('-n', '--num-labeled', default=50000,type=int,
                     help='Number of labeled examples to train on')
parser.add_argument('--epochstrain', default=200, type=int, metavar='N',
                    help='number of epochs victim was trained with')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of epochs stolen model was trained with')
parser.add_argument('--num_queries', default=9000, type=int, metavar='N',
                    help='Number of queries to steal the model.')
parser.add_argument('--lr', default=1e-4, type=float, # maybe try other lrs
                    help='learning rate to train the model with.')
parser.add_argument('--modeltype', default='victim', type=str,
                    help='Type of model to evaluate', choices=['victim', 'stolen'])
parser.add_argument('--save', default='False', type=str,
                    help='Save final model', choices=['True', 'False'])
parser.add_argument('--losstype', default='infonce', type=str,
                    help='Loss function to use.')
parser.add_argument('--head', default='False', type=str,
                    help='stolen model was trained using recreated head.', choices=['True', 'False'])
parser.add_argument('--defence', default='False', type=str,
                    help='Use defence on the victim side by perturbing outputs', choices=['True', 'False'])
parser.add_argument('--sigma', default=0.5, type=float,
                    help='standard deviation used for perturbations')
parser.add_argument('--mu', default=5, type=float,
                    help='mean noise used for perturbations')
parser.add_argument('--clear', default='True', type=str,
                    help='Clear previous logs', choices=['True', 'False'])
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--bias_samples', default=100, type=int, metavar='N',
                    help='Number of samples to train downstream model to see effect of bias')
args = parser.parse_args()
if args.bias_samples < 1000:
    args.batch_size = 64
if args.bias_samples <= 100:
    args.batch_size = 32

def load_victim(epochs, dataset, model, loss, args, device):

    print("Loading victim model: ")
    if dataset == "imagenet":
        # model = torchvision.models.resnet50(pretrained=True).to(device) # Pytorch pretrained resnet
        # model.fc = torch.nn.Linear(2048, 10).to(device)

        class ResNet50(torch.nn.Module):   # model from SimCLR
            def __init__(self, pretrained, num_classes=10):
                super(ResNet50, self).__init__()
                self.pretrained = pretrained
                self.fc = torch.nn.Sequential(torch.nn.Linear(512 * 4 * 1, num_classes))

            def forward(self, x):
                x = self.pretrained(x)
                x = self.fc(x)
                return x

        model = resnet50rep().to(device)
        checkpoint = torch.load(   # change path
                f'/ssd003/home/akaleem/SimCLR/models/resnet50-1x.pth', map_location=device)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict, strict=False)
        model = ResNet50(pretrained=model).to(device)


        return model
    if args.arch == "convnet":
        checkpoint = torch.load(
            f"/checkpoint/{os.getenv('USER')}/SimCLRBias/{epochs}{args.arch}{loss}TRAIN/{dataset}_checkpoint_{epochs}_{loss}.pth.tar",
            map_location=device)
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        # Remove head.
        for k in list(state_dict.keys()):
            if k.startswith('backbone.'):
                if k.startswith('backbone') and not k.startswith('backbone.out2'):
                    # remove prefix
                    new_state_dict[k[len("backbone."):]] = state_dict[k]
            else:
                new_state_dict[k] = state_dict[k]

        log = model.load_state_dict(new_state_dict, strict=False)
        assert log.missing_keys == ['out2.weight', 'out2.bias']
    else:
        checkpoint = torch.load(
            f"/checkpoint/{os.getenv('USER')}/SimCLRBias/{epochs}{args.arch}{loss}TRAIN/{dataset}_checkpoint_{epochs}_{loss}.pth.tar",
            map_location=device)
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        # Remove head.
        for k in list(state_dict.keys()):
            if k.startswith('backbone.'):
                if k.startswith('backbone') and not k.startswith('backbone.fc'):
                    # remove prefix
                    new_state_dict[k[len("backbone."):]] = state_dict[k]
            else:
                new_state_dict[k] = state_dict[k]

        log = model.load_state_dict(new_state_dict, strict=False)
        assert log.missing_keys == ['fc.weight', 'fc.bias']
    return model

def load_stolen(epochs, loss, model, dataset, queries, device):

    print("Loading stolen model: ")

    if args.head == "False":
        checkpoint = torch.load(
            f"/checkpoint/{os.getenv('USER')}/SimCLRBias/{epochs}{args.arch}{loss}STEAL/stolen_checkpoint_{queries}_{loss}_{dataset}.pth.tar",
            map_location=device)
    else:
        checkpoint = torch.load(
        f"/checkpoint/{os.getenv('USER')}/SimCLRBias/{epochs}{args.arch}STEALHEAD/stolen_checkpoint_{epochs}_{loss}.pth.tar", map_location=device)

    if args.defence == "True":
        checkpoint = torch.load(
            f"/checkpoint/{os.getenv('USER')}/SimCLRBias/{epochs}{args.arch}{loss}DEFENCE/stolen_checkpoint_{queries}_{loss}_{dataset}.pth.tar",
            map_location=device)
        print("Used defence")

    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    if args.arch == "convnet":
        # Remove head.
        for k in list(state_dict.keys()):
            if k.startswith('backbone.'):
                if k.startswith('backbone') and not k.startswith('backbone.out2'):
                    # remove prefix
                    new_state_dict[k[len("backbone."):]] = state_dict[k]
            else:
                new_state_dict[k] = state_dict[k]

        log = model.load_state_dict(new_state_dict, strict=False)
        assert log.missing_keys == ['out2.weight', 'out2.bias']
    else:
        if loss == "symmetrized":
            for k in list(state_dict.keys()):
                if k.startswith('encoder.'):
                    if k.startswith('encoder') and not k.startswith(
                            'encoder.fc'):
                        # remove prefix
                        new_state_dict[k[len("encoder."):]] = state_dict[k]
                else:
                    new_state_dict[k] = state_dict[k]
        else:
            for k in list(state_dict.keys()):
                if k.startswith('backbone.'):
                    if k.startswith('backbone') and not k.startswith('backbone.fc'):
                        # remove prefix
                        new_state_dict[k[len("backbone."):]] = state_dict[k]
                else:
                    new_state_dict[k] = state_dict[k]

        log = model.load_state_dict(new_state_dict, strict=False)
        assert log.missing_keys == ['fc.weight', 'fc.bias']
    return model

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

def get_mnist_data_loaders(download, shuffle=False, batch_size=args.batch_size):
    train_dataset = datasets.MNIST(f"/ssd003/home/{os.getenv('USER')}/data", train=True, download=download,
                                  transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
    test_dataset = datasets.MNIST(f"/ssd003/home/{os.getenv('USER')}/data", train=False, download=download,
                                  transform=transforms.ToTensor())
    # indxs = list(range(len(test_dataset) - 1000, len(test_dataset)))
    # test_dataset = torch.utils.data.Subset(test_dataset,
    #                                        indxs)  # only select last 1000 samples to prevent overlap with queried samples.
    test_loader = DataLoader(test_dataset, batch_size=64,
                            num_workers=2, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def get_emnist_data_loaders(download, shuffle=False, batch_size=args.batch_size):
    train_dataset = datasets.EMNIST(f"/ssd003/home/{os.getenv('USER')}/data", train=True, download=download,
                                  transform=transforms.ToTensor(), split="letters")
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
    test_dataset = datasets.EMNIST(f"/ssd003/home/{os.getenv('USER')}/data", train=False, download=download,
                                  transform=transforms.ToTensor(),split="letters")
    train_dataset.targets -= 1
    test_dataset.targets -= 1  # labels go from 0 to 25 instead of 1 to 26
    # indxs = list(range(len(test_dataset) - 1000, len(test_dataset)))
    # test_dataset = torch.utils.data.Subset(test_dataset,
    #                                        indxs)  # only select last 1000 samples to prevent overlap with queried samples.
    test_loader = DataLoader(test_dataset, batch_size=64,
                            num_workers=2, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def get_bias_data_loaders(download, shuffle=False, batch_size=args.batch_size, num_train=args.bias_samples):
    # only include 1000 samples from the training set.
    train_dataset = datasets.MNIST(f"/ssd003/home/{os.getenv('USER')}/data", train=True, download=download,
                                  transform=transforms.ToTensor())
    # select fixed number of samples per class
    nums = [0 for i in range(10)]
    targets = train_dataset.targets
    indxs = []
    total = num_train # number of samples to select
    l = [i for i in range(len(train_dataset))]
    random.shuffle(l)  # randomly shuffle samples
    for i in l:
        if nums[targets[i]] < total/10:
            indxs.append(i)
            nums[targets[i]] += 1
        if len(indxs) == total:
            break

    # print("Numbers from each class", nums)
    # print("indxs", indxs)

    train_dataset = torch.utils.data.Subset(train_dataset, indxs)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=True, shuffle=shuffle)
    test_dataset = datasets.MNIST(f"/ssd003/home/{os.getenv('USER')}/data", train=False, download=download,
                                  transform=transforms.ToTensor())
    #indxs =  list(range(len(test_dataset) - 1000, len(test_dataset)))
    # test_dataset = torch.utils.data.Subset(test_dataset,
    #                                        indxs)  # only select last 1000 samples to prevent overlap with queried samples.
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





if args.modeltype == "stolen":
    if args.head == "False":
        log_dir = f"/checkpoint/{os.getenv('USER')}/SimCLRBias/{args.epochs}{args.arch}{args.losstype}STEAL/"  # save logs here.
        logname = f'testing{args.modeltype}{args.dataset_test}{args.num_queries}.log'
    else:
        log_dir = f"/checkpoint/{os.getenv('USER')}/SimCLRBias/{args.epochs}{args.arch}STEALHEAD/"
        logname = f'testing{args.modeltype}{args.dataset_test}{args.num_queries}{args.losstype}.log'
    if args.datasetsteal == "imagenet":
        log_dir = f"/checkpoint/{os.getenv('USER')}/SimCLRBias/{args.epochs}{args.arch}{args.losstype}STEAL/"  # save logs here.
        logname = f'testing{args.modeltype}{args.dataset_test}{args.num_queries}IMAGENET.log'
else:
    if args.dataset == "imagenet":
        args.arch = "resnet50"
    # else:
    #     args.arch = "resnet34"
    log_dir = f"/checkpoint/{os.getenv('USER')}/SimCLRBias/{args.epochstrain}{args.arch}{args.losstype}TRAIN/"
    logname = f'testing{args.modeltype}{args.dataset_test}.log'
if args.clear == "True":
    if os.path.exists(os.path.join(log_dir, logname)):
        os.remove(os.path.join(log_dir, logname))
logging.basicConfig(
    filename=os.path.join(log_dir, logname),
    level=logging.DEBUG)

if args.dataset_test == "cifar100":
    args.num_classes = 100
elif args.dataset_test == "emnist":
    args.num_classes = 26
else:
    args.num_classes = 10

if args.arch == 'resnet18':
    model = ResNet18(num_classes=args.num_classes).to(device)
elif args.arch == 'resnet34':
    model = ResNet34( num_classes=args.num_classes).to(device)
elif args.arch == 'resnet50':
    model = ResNet50(num_classes=args.num_classes).to(device)
elif args.arch == 'convnet':
    model = ConvNet(num_classes=args.num_classes).to(device)

if args.modeltype == "victim":
    model = load_victim(args.epochstrain, args.dataset, model, args.losstype, args,
                                         device=device)
    print("Evaluating victim")
else:
    model = load_stolen(args.epochs, args.losstype, model, args.datasetsteal, args.num_queries,
                        device=device)
    print("Evaluating stolen model")

if args.dataset_test == 'cifar10':
    train_loader, test_loader = get_cifar10_data_loaders(download=False)
elif args.dataset_test == 'stl10':
    train_loader, test_loader = get_stl10_data_loaders(download=False)
elif args.dataset_test == "svhn":
    train_loader, test_loader = get_svhn_data_loaders(download=False)
elif args.dataset_test == "mnist":
    train_loader, test_loader = get_mnist_data_loaders(download=False)
elif args.dataset_test == "emnist":
    train_loader, test_loader = get_emnist_data_loaders(download=True)
elif args.dataset_test == "bias":
    train_loader, test_loader = get_bias_data_loaders(download=True, num_train=args.bias_samples)

# freeze all layers but the last fc (can try by training all layers)
if args.arch == "convnet":
    for name, param in model.named_parameters():
        if name not in ['out2.weight', 'out2.bias']:
            param.requires_grad = False
else:
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias', 'fc.0.weight', 'fc.0.bias']: # the imagenet model has fc.0 for the last layer
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

# Per class accuracy:
model.eval()
losses = []
correct = 0
total = len(test_loader.dataset)
correct_detailed = np.zeros(args.num_classes, dtype=np.int64)
wrong_detailed = np.zeros(args.num_classes, dtype=np.int64)
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()

        output = model(data)

        preds = output.data.argmax(axis=1)
        labels = target.data.view_as(preds)
        correct += preds.eq(labels).cpu().sum().item()
        outputd = output.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy().astype(int)
        preds_np = preds.detach().cpu().numpy().astype(int)


        for label, pred in zip(target, preds):
            if label == pred:
                correct_detailed[label] += 1
            else:
                wrong_detailed[label] += 1

acc = 100.0 * correct / total

assert correct_detailed.sum() + wrong_detailed.sum() == total
acc_detailed = 100.0 * correct_detailed / (
        correct_detailed + wrong_detailed)
print("acc detailed", acc_detailed)
logging.debug(
        f"detailed acc: {acc_detailed}")

if args.save == "True":
    if args.modeltype == "stolen":
        torch.save(model.state_dict(), f"/checkpoint/{os.getenv('USER')}/SimCLRBias/downstream/stolen_linear_{args.dataset_test}.pth.tar")
    else:
        torch.save(model.state_dict(), f"/checkpoint/{os.getenv('USER')}/SimCLRBias/downstream/victim_linear_{args.dataset_test}.pth.tar")
