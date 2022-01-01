# Other loss functions to try for the stealing process.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision
import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import logging
from torchvision import datasets



# Wasserstein:
# https://github.com/tensorflow/gan/blob/master/tensorflow_gan/python/losses/losses_impl.py#L71
#https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html#wasserstein-gan-wgan
# https://github.com/lilianweng/unified-gan-tensorflow/blob/317c1b6ec4d00db0d486dfce2965cb27156d334d/model.py#L169

def wasserstein_loss(pred, target):
    """
    pred is the representation output from the stolen model.
    target is the representation obtained from the victim model (h)
    """
    # Need to check if this implementation is correct as the setting here is slightly different (no seperate generator)
    # We consider f to be a function which gives the representation for the stolen model. We want to get E(f(x)) to be
    # as close to the real model

    return torch.mean(torch.softmax(target, dim=1)) -  torch.mean(torch.softmax(pred, dim=1))

# https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#contrastive-training-objectives

# Contrastive, triplet loss might not feasible as we do not know the actual labels and rather only the embedding.

# NCE (Noise contrastive estimation):
# https://github.com/demelin/Noise-Contrastive-Estimation-NCE-for-pyTorch/blob/master/nce_loss.py
# Binary cross entropy loss is used here. We also wanted to use binary cross entropy loss for a comparison with multilabel.
# https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
#nn.BCELoss

# Soft nearest neighbours loss requires a class label and so is not feasible.

