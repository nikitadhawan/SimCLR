# FGSM attack on linear classifier for trained stolen model.
# Also tests the performance of the victim model based on the generated adversarial examples.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
test_dataset = datasets.CIFAR10('/ssd003/home/akaleem/data/', train=False, download=False,
                                  transform=transforms.transforms.Compose([
                                  transforms.ToTensor(),
                                  ]))
indxs = list(range(len(test_dataset) - 1000, len(test_dataset)))
test_dataset = torch.utils.data.Subset(test_dataset, indxs)
test_loader = DataLoader(test_dataset, batch_size=1,
                        num_workers=2, drop_last=False, shuffle=False)

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-folder-name', metavar='DIR', default='test',
                    help='path to dataset')
parser.add_argument('-dataset', default='cifar10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
        choices=['resnet18', 'resnet34', 'resnet50'], help='model architecture')
parser.add_argument('-n', '--num-labeled', default=500,
                     help='Number of labeled batches to train on')
parser.add_argument('--epochstrain', default=200, type=int, metavar='N',
                    help='number of epochs victim was trained with')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of epochs stolen model was trained with')
parser.add_argument('--modeltype', default='victim', type=str,
                    help='Type of model to evaluate', choices=['victim', 'stolen'])
args = parser.parse_args()


# Code adapted from https://pytorch.org/tutorials/beginner/fgsm_tutorial.html

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def test( model, device, test_loader, epsilon, victim_model):

    # Accuracy counter
    correct = 0
    correct2 = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the stolen model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)
        output2 = victim_model(data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        final_pred2 = output2.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        if final_pred2.item() == target.item():
            correct2 += 1

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    final_acc_vic = correct2 / float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy (Stolen Model) = {} / {} = {}".format(
        epsilon, correct, len(test_loader), final_acc))
    print("Epsilon: {}\tTest Accuracy (Victim Model) = {} / {} = {}".format(
        epsilon, correct2, len(test_loader), final_acc_vic))
    # Return the accuracy and an adversarial example
    return final_acc, final_acc_vic, adv_examples



if args.arch == 'resnet18':
    stolen_model = torchvision.models.resnet18(pretrained=False, num_classes=10)

    victim_model = torchvision.models.resnet18(pretrained=False, num_classes=10)
elif args.arch == 'resnet34':
    stolen_model = torchvision.models.resnet34(pretrained=False,
                                        num_classes=10)
    victim_model = torchvision.models.resnet34(pretrained=False,
                                        num_classes=10)
elif args.arch == 'resnet50':
    stolen_model = torchvision.models.resnet50(pretrained=False, num_classes=10)
    victim_model = torchvision.models.resnet50(pretrained=False, num_classes=10)

checkpoint = torch.load(
        '/ssd003/home/akaleem/SimCLR/runs/eval/stolen_linear.pth.tar')
stolen_model.load_state_dict(checkpoint)  # load stolen model

checkpoint2 = torch.load(
    '/ssd003/home/akaleem/SimCLR/runs/eval/victim_linear.pth.tar')
victim_model.load_state_dict(checkpoint2)  # load victim model

victim_model.to(device)
stolen_model.to(device)
stolen_model.eval()
victim_model.eval()

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

top1_accuracy = 0
top5_accuracy = 0
for counter, (x_batch, y_batch) in enumerate(test_loader):
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    logits = stolen_model(x_batch)

    top1, top5 = accuracy(logits, y_batch, topk=(1,5))
    top1_accuracy += top1[0]
    top5_accuracy += top5[0]

top1_accuracy /= (counter + 1)
top5_accuracy /= (counter + 1)
print(f"Top1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")


epsilons = [0, .05, .1, .15, .2, .25, .3]

for eps in epsilons:
    test(stolen_model, device, test_loader, eps, victim_model)
