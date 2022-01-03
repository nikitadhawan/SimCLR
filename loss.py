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


# NCE (Noise contrastive estimation):
# https://github.com/demelin/Noise-Contrastive-Estimation-NCE-for-pyTorch/blob/master/nce_loss.py
# Binary cross entropy loss is used here. We also wanted to use binary cross entropy loss for a comparison with multilabel.
# https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html

def make_sampling_array(range_max, array_path):
    """ Creates and populates the array from which the fake labels are sampled during the NCE loss calculation."""
    # Get class probabilities
    print('Computing the Zipfian distribution probabilities for the corpus items.')
    class_probs = {class_id: get_probability(class_id, range_max) for class_id in range(range_max)}

    print('Generating and populating the sampling array. This may take a while.')
    # Generate empty array
    sampling_array = np.zeros(int(1e8))
    # Determine how frequently each index has to appear in array to match its probability
    class_counts = {class_id: int(np.round((class_probs[class_id] * 1e8))) for class_id in range(range_max)}
    assert(sum(list(class_counts.values())) == 1e8), 'Counts don\'t add up to the array size!'

    # Populate sampling array
    pos = 0
    for key, value in class_counts.items():
        while value != 0:
            sampling_array[pos] = key
            pos += 1
            value -= 1

    return sampling_array, class_probs

def sample_values(true_classes, num_sampled, unique, no_accidental_hits, sampling_array, class_probs):
    """ Samples negative items for the calculation of the NCE loss. Operates on batches of targets. """
    # Initialize output sequences
    sampled_candidates = np.zeros(num_sampled)
    true_expected_count = np.zeros(true_classes.size())
    sampled_expected_count = np.zeros(num_sampled)

    # If the true labels should not be sampled as a noise items, add them all to the rejected list
    if no_accidental_hits:
        rejected = list()
    else:
        rejected = true_classes.tolist()
    # Assign true label probabilities
    rows, cols = true_classes.size()
    for i in range(rows):
        for j in range(cols):
            true_expected_count[i][j] = class_probs[true_classes.data[i][j]]
    # Obtain sampled items and their probabilities
    print('Sampling items and their probabilities.')
    for k in range(num_sampled):
        sampled_pos = np.random.randint(int(1e8))
        sampled_idx = sampling_array[sampled_pos]
        if unique:
            while sampled_idx in rejected:
                sampled_idx = sampling_array[np.random.randint(0, int(1e8))]
        # Append sampled candidate and its probability to the output sequences for current target
        sampled_candidates[k] = sampled_idx
        sampled_expected_count[k] = class_probs[sampled_idx]
        # Re-normalize probabilities
        if unique:
            class_probs = renormalize(class_probs, sampled_idx)

    # Process outputs before they are returned
    sampled_candidates = sampled_candidates.astype(np.int64, copy=False)
    true_expected_count = true_expected_count.astype(np.float32, copy=False)
    sampled_expected_count = sampled_expected_count.astype(np.float32, copy=False)

    return Variable(torch.LongTensor(sampled_candidates)), \
           Variable(torch.FloatTensor(true_expected_count)), \
           Variable(torch.FloatTensor(sampled_expected_count))

class NCELoss(nn.Module):
    """ Class for calculating of the noise-contrasting estimation loss. """
    def __init__(self, opt, vocab_size):
        super(NCELoss, self).__init__()
        # Initialize parameters
        self.vocab_size = vocab_size
        self.opt = opt

        # Initialize the sampling array and the class probability dictionary
        if os.path.isfile(self.opt.array_path):
            print('Loading sampling array from the pickle %s.' % self.opt.array_path)
            with open(self.opt.array_path, 'rb') as f:
                self.sampling_array, self.class_probs = pickle.load(f)
        else:
            self.sampling_array, self.class_probs = make_sampling_array(self.vocab_size, self.opt.array_path)

    def forward(self, inputs, labels, weights, biases, sampled_values=None):
        """ Performs the forward pass. If sampled_values is None, a log uniform candidate sampler is used
        to obtain the required values. """

        # SHAPES:
        # inputs shape=[batch_size, dims]
        # flat_labels has shape=[batch_size * num_true]
        # sampled_candidates has shape=[num_sampled]
        # true_expected_count has shape=[batch_size, num_true]
        # sampled_expected_count has shape=[num_sampled]
        # all_ids has shape=[batch_size * num_true + num_sampled]
        # true_w has shape=[batch_size * num_true, dims]
        # true_b has shape=[batch_size * num_true]
        # sampled_w has shape=[num_sampled, dims]
        # sampled_b has shape=[num_sampled]
        # row_wise_dots has shape=[batch_size, num_true, dims]
        # dots_as_matrix as size=[batch_size * num_true, dims]
        # true_logits has shape=[batch_size, num_true]
        # sampled_logits has shape=[batch_size, num_sampled]

        flat_labels = labels.view([-1])
        num_true = labels.size()[1]
        true_per_batch = flat_labels.size()[0]
        print('Obtaining sampled values ...')
        if sampled_values is None:
	    # Indices representing the data classes have to be sorted in the order of descending frequency
	    # for the sampler to provide representative distractors and frequency counts
            sampled_values = sample_values(labels, self.opt.num_sampled, self.opt.unique,
                                           self.opt.remove_accidental_hits, self.sampling_array,
                                           self.class_probs)
        # Stop gradients for the sampled values
        sampled_candidates, true_expected_count, sampled_expected_count = (s.detach() for s in sampled_values)

        print('Calculating the NCE loss ...')
        # Concatenate true and sampled labels
        all_ids = torch.cat((flat_labels, sampled_candidates), 0)
        # Look up the embeddings of the combined labels
        all_w = torch.index_select(weights, 0, all_ids)
        all_b = torch.index_select(biases, 0, all_ids)
        # Extract true values
        true_w = all_w[:true_per_batch, :]
        true_b = all_b[:true_per_batch]
        # Extract sampled values
        sampled_w = all_w[true_per_batch:, :]
        sampled_b = all_b[true_per_batch:]
        # Obtain true logits
        tw_c = true_w.size()[1]
        true_w = true_w.view(-1, num_true, tw_c)
        row_wise_dots = inputs.unsqueeze(1) * true_w
        dots_as_matrix = row_wise_dots.view(-1, tw_c)
        true_logits = torch.sum(dots_as_matrix, 1).view(-1, num_true)
        true_b = true_b.view(-1, num_true)
        true_logits += true_b.expand_as(true_logits)
        # Obtain sampled logits; @ is the matmul operator
        sampled_logits = inputs @ sampled_w.t()
        sampled_logits += sampled_b.expand_as(sampled_logits)

        if self.opt.subtract_log_q:
            print('Subtracting log(Q(y|x)) ...')
            # Subtract the log expected count of the labels in the sample to get the logits of the true labels
            true_logits -= torch.log(true_expected_count)
            sampled_logits -= torch.log(sampled_expected_count.expand_as(sampled_logits))

        # Construct output logits and labels
        out_logits = torch.cat((true_logits, sampled_logits), 1)
        # Divide true logit labels by num_true to ensure the per-example labels sum to 1.0,
        # i.e. form a proper probability distribution.
        true_logit_labels = torch.ones(true_logits.size()) / num_true
        sampled_logit_labels = torch.zeros(sampled_logits.size())
        out_labels = torch.cat((true_logit_labels, sampled_logit_labels), 1)
        out_labels = Variable(out_labels)

        # Calculate the sampled losses (equivalent to TFs 'sigmoid_cross_entropy_with_logits')
        loss_criterion = nn.BCELoss()
        nce_loss = loss_criterion(torch.sigmoid(out_logits), out_labels)
        return nce_loss


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