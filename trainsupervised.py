'''Train CIFAR10 with PyTorch.'''
import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models.resnet import ResNet18

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', default=100, type=int, help='num of epochs')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root=f"/ssd003/home/{os.getenv('USER')}/data/", train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root=f"/ssd003/home/{os.getenv('USER')}/data/", train=False, download=False, transform=transform_test)
indxs = list(range(len(testset) - 1000, len(testset)))
testset = torch.utils.data.Subset(testset,
                                       indxs)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


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

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # train_loss += loss.item()
        # _, predicted = outputs.max(1)
        # total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()

        #print(f"Loss: {train_loss/(batch_idx+1)}, Accuracy: {100. * correct/total}")


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        top1_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(testloader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = net(x_batch)

            top1, top5 = accuracy(logits, y_batch, topk=(1,5))
            top1_accuracy += top1[0]

        top1_accuracy /= (counter + 1)
        # for batch_idx, (inputs, targets) in enumerate(testloader):
        #     inputs, targets = inputs.to(device), targets.to(device)
        #     outputs = net(inputs)
        #     loss = criterion(outputs, targets)
        #
        #     test_loss += loss.item()
        #     _, predicted = outputs.max(1)
        #     total += targets.size(0)
        #     correct += predicted.eq(targets).sum().item()
        #
        #     print(f"Loss: {test_loss/(batch_idx+1)}, Accuracy: {100. * correct/total}")

        print(f"Test accuracy: {top1_accuracy}")

    # Save checkpoint.
    acc = top1_accuracy
    if acc > best_acc:
        print('Saving..')
        model = net.module
        state = {
            'state_dict': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, f"/checkpoint/{os.getenv('USER')}/SimCLRsupervised/victim_supervised_cifar10.pth.tar")
        best_acc = acc


for epoch in range(start_epoch, start_epoch + args.epochs):
    train(epoch)
    test(epoch)
    scheduler.step()



#
#
# import torch
# import sys
# import numpy as np
# import os
# import yaml
# import matplotlib.pyplot as plt
# import torchvision
# import argparse
# from torch.utils.data import DataLoader
# from models.resnet import ResNetSimCLR, ResNet18, ResNet34 , ResNet50 # from other file
# from models.resnet_wider import resnet50rep, resnet50rep2, resnet50x1
# import torchvision.transforms as transforms
# import logging
# from torchvision import datasets
#
# #https://github.com/kuangliu/pytorch-cifar/blob/49b7aa97b0c12fe0d4054e670403a16b6b834ddd/main.py
#
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print("Using device:", device)
#
# parser = argparse.ArgumentParser(description='PyTorch SimCLR')
# parser.add_argument('-folder-name', metavar='DIR', default='test',
#                     help='path to dataset')
# parser.add_argument('--dataset', default='cifar10',
#                     help='dataset name', choices=['stl10', 'cifar10', 'svhn', 'imagenet'])
# parser.add_argument('--dataset-test', default='cifar10',
#                     help='dataset to run downstream task on', choices=['stl10', 'cifar10', 'svhn'])
# parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
#         choices=['resnet18', 'resnet34', 'resnet50'], help='model architecture')
# parser.add_argument('-n', '--num-labeled', default=50000,type=int,
#                      help='Number of labeled examples to train on')
# parser.add_argument('--epochstrain', default=200, type=int, metavar='N',
#                     help='number of epochs victim was trained with')
# parser.add_argument('--epochs', default=100, type=int, metavar='N',
#                     help='number of epochs stolen model was trained with')
# parser.add_argument('--lr', default=0.1, type=float,
#                     help='learning rate to train the model with.')
# parser.add_argument('--save', default='True', type=str,
#                     help='Save final model', choices=['True', 'False'])
# parser.add_argument('--clear', default='False', type=str,
#                     help='Clear previous logs', choices=['True', 'False'])
# parser.add_argument('-b', '--batch-size', default=128, type=int,
#                     metavar='N',
#                     help='mini-batch size (default: 256), this is the total '
#                          'batch size of all GPUs on the current node when '
#                          'using Data Parallel or Distributed Data Parallel')
# args = parser.parse_args()
#
#
#
#
# def get_stl10_data_loaders(download, shuffle=False, batch_size=args.batch_size):
#     train_dataset = datasets.STL10(f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10", split='train', download=download,
#                                   transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))
#     train_loader = DataLoader(train_dataset, batch_size=batch_size,
#                             num_workers=0, drop_last=False, shuffle=shuffle)
#     test_dataset = datasets.STL10(f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10", split='test', download=download,
#                                   transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))
#     test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
#                             num_workers=2, drop_last=False, shuffle=shuffle)
#     return train_loader, test_loader
#
# def get_cifar10_data_loaders(download, shuffle=False, batch_size=args.batch_size):
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465),
#                              (0.2023, 0.1994, 0.2010)),
#     ])
#
#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465),
#                              (0.2023, 0.1994, 0.2010)),
#     ])
#     train_dataset = datasets.CIFAR10(f"/ssd003/home/{os.getenv('USER')}/data/", train=True, download=download,
#                                   transform=transform_train)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size,
#                             num_workers=0, drop_last=False, shuffle=shuffle)
#     test_dataset = datasets.CIFAR10(f"/ssd003/home/{os.getenv('USER')}/data/", train=False, download=download,
#                                   transform=transform_test)
#     indxs = list(range(len(test_dataset) - 1000, len(test_dataset)))
#     test_dataset = torch.utils.data.Subset(test_dataset,
#                                            indxs)  # only select last 1000 samples to prevent overlap with queried samples.
#     test_loader = DataLoader(test_dataset, batch_size=64,
#                             num_workers=2, drop_last=False, shuffle=shuffle)
#     return train_loader, test_loader
#
# def get_svhn_data_loaders(download, shuffle=False, batch_size=args.batch_size):
#     train_dataset = datasets.SVHN(f"/ssd003/home/{os.getenv('USER')}/data/SVHN", split='train', download=download,
#                                   transform=transforms.ToTensor())
#     train_loader = DataLoader(train_dataset, batch_size=batch_size,
#                             num_workers=0, drop_last=False, shuffle=shuffle)
#     test_dataset = datasets.SVHN(f"/ssd003/home/{os.getenv('USER')}/data/SVHN", split='test', download=download,
#                                   transform=transforms.ToTensor())
#     indxs = list(range(len(test_dataset) - 1000, len(test_dataset)))
#     test_dataset = torch.utils.data.Subset(test_dataset,
#                                            indxs)  # only select last 1000 samples to prevent overlap with queried samples.
#     test_loader = DataLoader(test_dataset, batch_size=64,
#                             num_workers=2, drop_last=False, shuffle=shuffle)
#     return train_loader, test_loader

#
#
#
# log_dir = f"/checkpoint/{os.getenv('USER')}/SimCLRsupervised/"
# logname = f'trainingvictim{args.dataset}.log'
# if args.clear == "True":
#     if os.path.exists(os.path.join(log_dir, logname)):
#         os.remove(os.path.join(log_dir, logname))
# logging.basicConfig(
#     filename=os.path.join(log_dir, logname),
#     level=logging.DEBUG)
#
# if args.arch == 'resnet18':
#     model = ResNet18(num_classes=10).to(device)
# elif args.arch == 'resnet34':
#     model = ResNet34( num_classes=10).to(device)
# elif args.arch == 'resnet50':
#     model = ResNet50(num_classes=10).to(device)
#
# if args.dataset_test == 'cifar10':
#     train_loader, test_loader = get_cifar10_data_loaders(download=False)
# elif args.dataset_test == 'stl10':
#     train_loader, test_loader = get_stl10_data_loaders(download=False)
# elif args.dataset_test == "svhn":
#     train_loader, test_loader = get_svhn_data_loaders(download=False)
#
#
# #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0008)
# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4)
# criterion = torch.nn.CrossEntropyLoss().to(device)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200,eta_min=0)
#
# epochs = 100
#
# ## Trains the representation model with a linear classifier to measure the accuracy on the test set labels of the victim/stolen model
#
# logging.info(f"Training supervised victim")
# logging.info(f"Args: {args}")
# for epoch in range(epochs):
#     top1_train_accuracy = 0
#     for counter, (x_batch, y_batch) in enumerate(train_loader):
#         x_batch = x_batch.to(device)
#         y_batch = y_batch.to(device)
#
#         logits = model(x_batch)
#         loss = criterion(logits, y_batch)
#         top1 = accuracy(logits, y_batch, topk=(1,))
#         top1_train_accuracy += top1[0]
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if (counter+1) * x_batch.shape[0] >= args.num_labeled:
#             break
#
#     top1_train_accuracy /= (counter + 1)
#     top1_accuracy = 0
#     top5_accuracy = 0
#     for counter, (x_batch, y_batch) in enumerate(test_loader):
#         x_batch = x_batch.to(device)
#         y_batch = y_batch.to(device)
#
#         logits = model(x_batch)
#
#         top1, top5 = accuracy(logits, y_batch, topk=(1,5))
#         top1_accuracy += top1[0]
#         top5_accuracy += top5[0]
#
#     top1_accuracy /= (counter + 1)
#     top5_accuracy /= (counter + 1)
#     print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
#     logging.debug(
#         f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
#     scheduler.step()
#
# torch.save(model.state_dict(), f"/checkpoint/{os.getenv('USER')}/SimCLRsupervised/victim_supervised_{args.dataset_test}.pth.tar")
#

