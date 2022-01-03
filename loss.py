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

    return torch.mean(pred) -  torch.mean(target)

# https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#contrastive-training-objectives

# Contrastive, triplet loss might not feasible as we do not know the actual labels and rather only the embedding.

# NCE (Noise contrastive estimation):
# https://github.com/demelin/Noise-Contrastive-Estimation-NCE-for-pyTorch/blob/master/nce_loss.py
# Binary cross entropy loss is used here. We also wanted to use binary cross entropy loss for a comparison with multilabel.
# https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
#nn.BCELoss

# Soft nearest neighbours Loss:
# https://arxiv.org/pdf/1902.01889.pdf , https://twitter.com/nickfrosst/status/1093581702453231623
# https://github.com/tensorflow/similarity/tree/master/tensorflow_similarity/losses
#https://github.com/tensorflow/similarity/pull/203/commits/c7b5304be9c7df40297aa8382d28400ba94337c8#diff-6fb616049a9a9c0d7cc4dc686ec1746520039c9845fa7fbf9d291054b222ca18


def build_masks(labels,
                batch_size):
    """Build masks that allows to select only the positive or negatives
    embeddings.
    Args:
        labels: 1D int `Tensor` that contains the class ids.
        batch_size: size of the batch.
    Returns:
        Tuple of Tensors containing the positive_mask and negative_mask
    """
    if np.ndim(labels) == 1:
        labels = torch.reshape(labels, (-1, 1))

    # same class mask
    positive_mask = (labels == labels.T).to(torch.bool)
    # not the same class
    negative_mask = torch.logical_not(positive_mask)

    # we need to remove the diagonal from positive mask
    diag = torch.logical_not(torch.diag(torch.ones(batch_size, dtype=torch.bool)))
    positive_mask = torch.logical_and(positive_mask, diag)

    return positive_mask, negative_mask

def pairwise_euclid_distance(a, b):
    STABILITY_EPS = 0.00001
    a = a.double()
    b = b.double()

    batch_a = a.shape[0]
    batch_b = b.shape[0]

    sqr_norm_a = torch.pow(a, 2).sum(dim=1).view(1,
                                                  batch_a) + STABILITY_EPS
    sqr_norm_b = torch.pow(b, 2).sum(dim=1).view(batch_b,
                                                  1) + STABILITY_EPS

    tile_1 = sqr_norm_a.repeat([batch_a, 1])
    tile_2 = sqr_norm_b.repeat([1, batch_b])

    inner_prod = torch.matmul(b, a.T) + STABILITY_EPS
    dist = tile_1 + tile_2 - 2 * inner_prod
    return dist

def soft_nn_loss(args,
                 features,
                 distance,
                 temperature=10000):
    """Computes the soft nearest neighbors loss.
    Args:
        labels: Labels associated with features. (now calculated in code below)
        features: Embedded examples.
        temperature: Controls relative importance given
                        to the pair of points.
    Returns:
        loss: loss value for the current batch.
    """

    # Can possibly combine cross entropy with this loss (as mentioned in the paper)
    batch_size = features.size()[0] 
    n = int(features.size()[0] / args.batch_size) # I think number of augmentations. Need to update code below based on this
    labels = torch.cat(
            [torch.arange(args.batch_size) for i in range(n)], dim=0)
    #labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    eps = 1e-9
    # might need to make labels manually as is done in info_nce_loss(). 
    pairwise_dist = distance(features, features)
    pairwise_dist = pairwise_dist / temperature
    negexpd = torch.exp(-pairwise_dist)

    # Mask out diagonal entries
    diag = torch.diag(torch.ones(batch_size, dtype=torch.bool))
    diag_mask = torch.logical_not(diag).float().to(args.device)
    negexpd = torch.mul(negexpd, diag_mask)

    # creating mask to sample same class neighboorhood
    pos_mask, _ = build_masks(labels, batch_size)
    pos_mask = pos_mask.type(torch.FloatTensor)
    pos_mask = pos_mask.to(args.device)

    # all class neighborhood
    alcn = torch.sum(negexpd, dim=1)

    # same class neighborhood
    sacn = torch.sum(torch.mul(negexpd, pos_mask), dim=1)

    # exclude examples with unique class from loss calculation
    excl = torch.not_equal(torch.sum(pos_mask, dim=1),
                             torch.zeros(batch_size).to(args.device))
    excl = excl.type(torch.FloatTensor).to(args.device)

    loss = torch.divide(sacn, alcn)
    loss = torch.multiply(torch.log(eps+loss), excl)
    loss = -torch.mean(loss)
    return loss

# https://github.com/vimarshc/fastai_experiments/blob/master/Colab%20Notebooks/entanglement.ipynb might help