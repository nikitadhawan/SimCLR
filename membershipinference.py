# Membership inference as a potential defence against model stealing attacks
# Currently trying LOSS based approach where we look at the loss of individual data samples and then plot a histogram

import argparse
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset, \
    RegularDataset, WatermarkDataset
from models.resnet_simclr import ResNetSimCLRV2, SimSiam, WatermarkMLP, HeadSimCLR
from models.resnet import ResNetSimCLRV2 as ResNetSimCLRNEW
from models.resnet_wider import resnet50rep, resnet50rep2, resnet50x1
from utils import load_victim, load_watermark, accuracy, print_args
import os
from torchvision import models, transforms
from data_aug.gaussian_blur import GaussianBlur
from tqdm import tqdm
from statistical_tests.t_test import ttest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.0, style='whitegrid', rc={"grid.linewidth": 1.})

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='/ssd003/home/akaleem/data',
                    help='path to dataset')
parser.add_argument('--dataset', default='cifar10',
                    help='dataset name(for training victim model)', choices=['stl10', 'cifar10', 'svhn', 'imagenet'])
parser.add_argument('--datasetsteal', default='cifar10',
                    help='dataset used for querying the victim', choices=['stl10', 'cifar10', 'svhn', 'imagenet'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--archstolen', default='resnet34',
                    choices=model_names,
                    help='stolen model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet34)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochstrain', default=200, type=int, metavar='N',
                    help='number of epochs victim was trained with')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=1, type=int,   # use 1 when using MSE loss
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='head feature (z) dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=200, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--temperaturesn', default=100, type=float,
                    help='temperature for soft nearest neighbors loss')
parser.add_argument('--num_queries', default=9000, type=int, metavar='N',
                    help='Number of queries to steal the model.')
parser.add_argument('--n-views', default=2, type=int, metavar='N',  # use 2 to use multiple augmentations.
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--logdir', default='test', type=str,
                    help='Log directory to save output to.')
parser.add_argument('--losstype', default='mse', type=str,
                    help='Loss function to use')
parser.add_argument('--lossvictim', default='infonce', type=str,
                    help='Loss function victim was trained with')
parser.add_argument('--victimhead', default='False', type=str,  #
                    help='Using victim head when computing loss for victim', choices=['True', 'False'])
parser.add_argument('--stolenhead', default='False', type=str,
                    help='Use an additional head while training the stolen model.', choices=['True', 'False'])
parser.add_argument('--defence', default='False', type=str,
                    help='Use defence on the victim side by perturbing outputs', choices=['True', 'False'])
parser.add_argument('--sigma', default=0.5, type=float,
                    help='standard deviation used for perturbations')
parser.add_argument('--mu', default=5, type=float,
                    help='mean noise used for perturbations')
parser.add_argument('--clear', default='True', type=str,
                    help='Clear previous logs', choices=['True', 'False'])
parser.add_argument('--watermark', default='False', type=str,
                    help='Evaluate with watermark model from victim', choices=['True', 'False'])
parser.add_argument('--entropy', default='False', type=str,
                    help='Use entropy victim model', choices=['True', 'False'])
parser.add_argument('--victim_model_type', type=str,
                    # default='supervised',
                    # default='encoder',
                    default='simsiam',
                    choices=['simclr','simsiam','encoder','supervised'],
                    help='The type of the model from supervised or '
                         'self-supervised learning.')
parser.add_argument('--dist_type', type=str,
                    # default='L2',
                    default='InfoNCE',
                    help='The distance type (metric) use to compare the '
                         'difference between the representations.')

args = parser.parse_args()
print_args(args)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

if args.dataset == "imagenet":  # manually set parameters in the case of imagenet
    args.arch = "resnet50"
    args.archstolen = "resnet50"
    args.datasetsteal = "imagenet"
    args.num_queries = 100000
    args.losstype = "infonce"
dataset = ContrastiveLearningDataset(args.data)
dataset2 = RegularDataset(args.data)
assert args.n_views == 2
train_dataset = dataset.get_dataset(args.dataset,  args.n_views)  # augmented training set
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)


val_dataset = dataset.get_dataset(args.dataset, args.n_views)
indxs = list(range(len(train_dataset) - 10000, len(train_dataset)))
val_dataset = torch.utils.data.Subset(val_dataset,
                                           indxs)
val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True, # shuffle to ensure random selection of points
        num_workers=args.workers, pin_memory=True, drop_last=True)

test_dataset = dataset.get_test_dataset(args.dataset,
                                                 args.n_views)

test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, # shuffle true to ensure random selection of points
        num_workers=args.workers, pin_memory=True, drop_last=True)

test_svhn = dataset2.get_test_dataset("svhn",1)
test_loader_svhn = torch.utils.data.DataLoader(
        test_svhn, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

# Transforms from contrastive_learning_dataset.py
color_jitter = transforms.ColorJitter(0.8, 0.8 , 0.8, 0.2)
data_transforms = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.RandomResizedCrop(size=32),
     transforms.RandomHorizontalFlip(),
     transforms.RandomApply([color_jitter], p=0.8),
     transforms.RandomGrayscale(p=0.2),
     GaussianBlur(kernel_size=int(0.1 * 32)),
     transforms.ToTensor()])


def info_nce_loss(features, args):
    n = int(features.size()[0] / args.batch_size)
    labels = torch.cat(
        [torch.arange(args.batch_size) for i in range(n)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    mask = torch.eye(labels.shape[0], dtype=torch.bool)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(
        similarity_matrix.shape[0], -1)
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(
        similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long)
    logits = logits / args.temperature
    return logits, labels
    # Need to put logits and labels on cuda.

criterion = nn.CrossEntropyLoss().to(device)
criterion2 = nn.CosineSimilarity(dim=1)
criterion3 = nn.MSELoss().to(device) # MSE

if args.dataset == "imagenet":
    if args.victim_model_type == "simsiam":
        victim_model = models.__dict__[args.arch]()
        checkpoint = torch.load(
            "models/checkpoint_0099-batch256.pth.tar",
            map_location="cpu")
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder up to before the embedding layer
            if k.startswith('module.encoder') and not k.startswith(
                    'module.encoder.fc'):
                # remove prefix
                state_dict[k[len("module.encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        # print("state dict", state_dict.keys())
        victim_model.load_state_dict(state_dict, strict=False)
        victim_model.fc = torch.nn.Identity()
        victim_model = victim_model.to(device)
    # For SimCLR pretrained model
    elif args.victim_model_type == "simclr":
        args.arch = "resnet50"
        if args.victim_model_type == 'supervised':
            victim_model = models.resnet50(pretrained=True).to(device)
            victim_model.fc = torch.nn.Identity().to(device)
        else:
            class ResNet50v2(nn.Module):
                def __init__(self, pretrained, num_classes=10):
                    super(ResNet50v2, self).__init__()
                    self.pretrained = pretrained

                def forward(self, x):
                    x = self.pretrained(x)
                    return x


            victim_model= resnet50rep().to(device)
            checkpoint = torch.load(
                    f'/ssd003/home/akaleem/SimCLR/models/resnet50-1x.pth', map_location="cpu")
            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict.copy()
            # for k in state_dict.keys():
            #     if k.startswith('fc.'):
            #         del new_state_dict[k]

            victim_model.load_state_dict(new_state_dict, strict=False)
            del state_dict
        # 2048 dimensional output
elif args.victimhead == "False":
    victim_model = ResNetSimCLRNEW(base_model=args.arch,
                                  out_dim=args.out_dim, loss=args.lossvictim,
                                  include_mlp=False).to(device)
    victim_model = load_victim(args.epochstrain, args.dataset, victim_model,
                               args.arch, args.lossvictim,
                               device=device, discard_mlp=True,
                               watermark=args.watermark, entropy=args.entropy)
else:  # We use this option when computing the loss with the head
    victim_model = ResNetSimCLRNEW(base_model=args.arch,
                                  out_dim=args.out_dim, loss=args.lossvictim,
                                  include_mlp=True).to(device)
    victim_model = load_victim(args.epochstrain, args.dataset, victim_model,
                               args.arch, args.lossvictim,
                               device=device)
    only_head = HeadSimCLR(out_dim=args.out_dim).to(device)
    checkpoint = torch.load(
            f"/checkpoint/{os.getenv('USER')}/SimCLR/{args.epochstrain}{args.arch}{args.lossvictim}TRAIN/{args.dataset}_checkpoint_{args.epochstrain}_{args.lossvictim}.pth.tar",
            map_location=device)
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k in list(state_dict.keys()):
        if k.startswith('backbone.fc'):
            new_state_dict[k[len("backbone."):]] = state_dict[k]
    only_head.load_state_dict(new_state_dict, strict=False)
    del state_dict
    print("Loaded victim head")



# Load stolen copy
if args.dataset == "imagenet":
    stolen_model = ResNetSimCLRV2(base_model=args.arch, out_dim=512, loss=None,
                   include_mlp=False).to(device)
    checkpoint = torch.load(
        f"/checkpoint/{os.getenv('USER')}/SimCLR/SimSiam/checkpoint_{args.datasetsteal}_{args.losstype}_{args.num_queries}.pth.tar",
        map_location="cpu")
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k in list(state_dict.keys()):
        if k.startswith('module.backbone'):
            # remove prefix
            new_state_dict[k[len("module."):]] = state_dict[k]
        else:
            new_state_dict[k] = state_dict[k]
    stolen_model.load_state_dict(new_state_dict)
    del state_dict

else:
    stolen_model = ResNetSimCLRV2(base_model=args.arch, out_dim=128, loss=None,
                                  include_mlp=False).to(device)
    # mse loss for victim for first tests.
    checkpoint = torch.load(
                f"/checkpoint/{os.getenv('USER')}/SimCLR/{args.epochs}{args.arch}{args.losstype}STEAL/stolen_checkpoint_{args.num_queries}_{args.losstype}_{args.datasetsteal}.pth.tar",
                map_location=device)
    state_dict = checkpoint['state_dict']
    stolen_model.load_state_dict(state_dict)


# if args.dataset == "imagenet":
    # random_model2 = models.resnet50(pretrained=True).to(device)
    # random_model2.fc = torch.nn.Identity().to(device) # This is a good model since it was trained on imagenet.
    ###
    # random_model = ResNetSimCLRV2(base_model=args.arch, out_dim=512, loss=None,
    #               include_mlp=False).to(device)

    # loss = "infonce"
    # checkpoint2 = torch.load(
    #             f"/checkpoint/{os.getenv('USER')}/SimCLR/100resnet18{loss}TRAIN/cifar10_checkpoint_9000_{loss}_cifar10.pth.tar",
    #             map_location=device)
    #
# random_dict = checkpoint2['state_dict']
random_model = ResNetSimCLRV2(base_model="resnet18", out_dim=128, loss=None, include_mlp = False).to(device) # Note: out_dim does not matter since last layer has no effect.
#random_model.load_state_dict(random_dict)
random_model = load_victim(50, "cifar10", random_model,
                               "resnet18", "infonce",
                               device=device, discard_mlp=True)

random_model2 = ResNetSimCLRV2(base_model="resnet34", out_dim=128, loss=None,
                              include_mlp=False).to(
    device)
random_model2 = load_victim(100, "cifar10", random_model2,
                           "resnet34", "infonce2",
                           device=device, discard_mlp=True)  # This is the model which was trained on the first 40000 samples from the training set.

# SVHN random model: 


random_model2 = ResNetSimCLRNEW(base_model="resnet34", out_dim=128, loss=None,
                              include_mlp=False).to(
    device)
random_model2 = load_victim(200, "svhn", random_model2,
                           "resnet34", "infonce",
                           device=device, discard_mlp=True)  # This is the model which was trained on the first 40000 samples from the training set.

                
victim_model.eval()
stolen_model.eval()
# random_model.eval()
random_model2.eval()
if args.victimhead == "True":
    only_head.eval()

print("Loaded models.")

randomtrain = [] # infonce loss across each batch
stolentrain = []
victimtrain = []
randomtest = []
stolentest = []
victimtest = []
with torch.no_grad():
    for counter, (images, truelabels) in enumerate(tqdm(train_loader)):
        images = torch.cat(images, dim=0)
        images = images.to(device)
        x1 = images[0].unsqueeze(dim=0)
        x2 = images[1].unsqueeze(dim=0) # first and second augmented views for each image

        victim_features_1 = victim_model(x1)
        stolen_features_1 = stolen_model(x1)
        random_features2_1 = random_model2(x1)
        victim_features_2 = victim_model(x2)
        stolen_features_2 = stolen_model(x2)
        random_features2_2 = random_model2(x2)   # compute representations of all 3 models on both views
        if args.victimhead == "True":
            stolen_features_1 = only_head(stolen_features_1)
            stolen_features_2 = only_head(stolen_features_2)
            #random_features2 = only_head(random_features2)
        loss = criterion3(victim_features_1, victim_features_2)
        victimtrain.append(loss.item())
        loss = criterion3(random_features2_1, random_features2_2)
        randomtrain.append(loss.item())
        loss = criterion3(stolen_features_1, stolen_features_2)
        stolentrain.append(loss.item())   # loss for all 3 models
        if counter >= 10000:
            break


    for counter, (images, truelabels) in enumerate(tqdm(test_loader)):
        images = torch.cat(images, dim=0)
        images = images.to(device)

        x1 = images[0].unsqueeze(dim=0)
        x2 = images[1].unsqueeze(dim=0)  # first and second augmented views for each image

        victim_features_1 = victim_model(x1)
        stolen_features_1 = stolen_model(x1)
        random_features2_1 = random_model2(x1)
        victim_features_2 = victim_model(x2)
        stolen_features_2 = stolen_model(x2)
        random_features2_2 = random_model2(x2)  # compute representations of all 3 models on both views
        if args.victimhead == "True":
            stolen_features_1 = only_head(stolen_features_1)
            stolen_features_2 = only_head(stolen_features_2)
            #random_features2 = only_head(random_features2)
        loss = criterion3(victim_features_1, victim_features_2)
        victimtest.append(loss.item())
        loss = criterion3(random_features2_1, random_features2_2)
        randomtest.append(loss.item())
        loss = criterion3(stolen_features_1, stolen_features_2)
        stolentest.append(loss.item())
        if counter >= 10000:
            break

plt.hist(victimtrain, density=True, bins='auto', label="train")
plt.hist(victimtest, density=True, bins='auto', label="test")
plt.title("Victim")
plt.xlabel("Loss")
plt.ylabel("Scaled frequency")
plt.legend(loc='upper right')
plt.savefig('plots/victim.jpg')
plt.close()
plt.hist(stolentrain, density=True, bins='auto', label="train")
plt.hist(stolentest, density=True, bins='auto', label="test")
plt.legend(loc='upper right')
plt.title("Stolen")
plt.xlabel("Loss")
plt.ylabel("Scaled frequency")
plt.savefig('plots/stolen.jpg')
plt.close()
plt.hist(randomtrain, density=True, bins='auto', label="train")
plt.hist(randomtest, density=True, bins='auto', label="test")
plt.legend(loc='upper right')
plt.title("Random")
plt.xlabel("Loss")
plt.ylabel("Scaled frequency")
plt.savefig('plots/random.jpg')
plt.close()


def run_ttest(train_losses, test_losses, args):
    print('t-test:')
    print('train_losses shape: ', train_losses.shape)
    print('test_losses shape: ', test_losses.shape)

    mean_train = np.mean(train_losses)
    print(f'mean_train {args.dataset} losses: ', mean_train)
    print('median_train: ', np.median(train_losses))

    mean_test = np.mean(test_losses)
    print(f'mean_test {args.dataset} losses: ', mean_test)
    print('median_test: ', np.median(test_losses))

    tval, pval = ttest(test_losses, train_losses, alternative="greater")
    print('Null hypothesis: losses test <= losses train')
    print('delta u: ', mean_test - mean_train, ' pval: ', pval, 'tval: ', tval)

    tval, pval = ttest(train_losses, test_losses, alternative="greater")
    print('Null hypothesis: losses train <= losses test')
    print('delta u: ', mean_train - mean_test, ' pval: ', pval, 'tval: ', tval)


print("Running t test for victim model")
run_ttest(np.array(victimtrain), np.array(victimtest), args)
print("Running t test for stolen model")
run_ttest(np.array(stolentrain), np.array(stolentest), args)
print("Running t test for random model")
run_ttest(np.array(randomtrain), np.array(randomtest), args)


