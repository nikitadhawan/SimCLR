# Other loss functions to try.

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

    return torch.abs(torch.mean(torch.sigmoid(target)) -  torch.mean(torch.sigmoid(pred)))  # loss was negative so torch.abs added. It might also work with a negative loss.

