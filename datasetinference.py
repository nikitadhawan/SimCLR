# Dataset inference approach to compare the distance between representations over the training set samples.
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset, \
    RegularDataset, WatermarkDataset
from models.resnet_simclr import ResNetSimCLRV2, SimSiam, WatermarkMLP
from models.resnet_wider import resnet50rep, resnet50rep2, resnet50x1
from utils import load_victim, load_watermark, accuracy
import os
from torchvision import models
from tqdm import tqdm


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='/ssd003/home/akaleem/data',
                    help='path to dataset')
parser.add_argument('--dataset', default='cifar10',
                    help='dataset name', choices=['stl10', 'cifar10', 'svhn', 'imagenet'])
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
parser.add_argument('-b', '--batch-size', default=64, type=int,
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
                    help='feature dimension (default: 128)')
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
parser.add_argument('--victimhead', default='False', type=str,
                    help='Access to victim head while (g) while getting representations', choices=['True', 'False'])
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


args = parser.parse_args()
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
dataset = ContrastiveLearningDataset(args.data)
train_dataset = dataset.get_dataset(args.datasetsteal, args.n_views)
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

test_dataset = dataset.get_test_dataset(args.dataset,
                                                 1)
indxs = list(range(len(test_dataset) - 1000, len(test_dataset)))
test_dataset = torch.utils.data.Subset(test_dataset,
                                           indxs)
test_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)
criterion2 = nn.CosineSimilarity(dim=1)

if args.dataset == "imagenet":
    victim_model = models.resnet50(pretrained=True).to(device)
    victim_model.fc = torch.nn.Identity().to(device)

    # 2048 dimensional output
elif args.victimhead == "False":
    victim_model = ResNetSimCLRV2(base_model=args.arch,
                                  out_dim=args.out_dim, loss=args.lossvictim,
                                  include_mlp=False).to(device)
    victim_model = load_victim(args.epochstrain, args.dataset, victim_model,
                               args.arch, args.lossvictim,
                               device=device, discard_mlp=True,
                               watermark=args.watermark, entropy=args.entropy)
else:
    victim_model = ResNetSimCLRV2(base_model=args.arch,
                                  out_dim=args.out_dim, loss=args.lossvictim,
                                  include_mlp=True).to(device)
    victim_model = load_victim(args.epochstrain, args.dataset, victim_model,
                               args.arch, args.lossvictim,
                               device=device)



# Load stolen copy

if args.dataset == "imagenet":
    stolen_model = ResNetSimCLRV2(base_model=args.arch, out_dim=512, loss=None,
                                  include_mlp=False).to(device)
else:
    stolen_model = ResNetSimCLRV2(base_model=args.arch, out_dim=128, loss=None,
                                  include_mlp=False).to(device)
# mse loss for victim for first tests.
checkpoint = torch.load(
            f"/checkpoint/{os.getenv('USER')}/SimCLR/{args.epochs}{args.arch}{args.losstype}STEAL/stolen_checkpoint_{args.num_queries}_{args.losstype}_{args.datasetsteal}.pth.tar",
            map_location=device)
state_dict = checkpoint['state_dict']
stolen_model.load_state_dict(state_dict)


# checkpoint2 = torch.load(
#             f"/checkpoint/{os.getenv('USER')}/SimCLR/100resnet18infonceTRAIN/cifar10_checkpoint_100_infonceWATERMARK.pth.tar",
#             map_location=device)
# use out_dim=512 below for this
if args.dataset == "imagenet":
    # With SimCLR model below:
    class ResNet50v2(nn.Module):
        def __init__(self, pretrained, num_classes=10):
            super(ResNet50v2, self).__init__()
            self.pretrained = pretrained

        def forward(self, x):
            x = self.pretrained(x)
            return x


    random_model= resnet50rep().to(device)
    checkpoint = torch.load(
            f'/ssd003/home/akaleem/SimCLR/models/resnet50-1x.pth', map_location=device)
    state_dict = checkpoint['state_dict']
    new_state_dict = state_dict.copy()
    # for k in state_dict.keys():
    #     if k.startswith('fc.'):
    #         del new_state_dict[k]

    random_model.load_state_dict(new_state_dict, strict=False)
else:
    loss = "mse"
    checkpoint2 = torch.load(
                f"/checkpoint/{os.getenv('USER')}/SimCLR/100resnet18{loss}STEAL/stolen_checkpoint_9000_{loss}_cifar10.pth.tar",
                map_location=device)

    random_dict = checkpoint2['state_dict']
    random_model = ResNetSimCLRV2(base_model="resnet18", out_dim=128, loss=None, include_mlp = False).to(device) # Note: out_dim does not matter since last layer has no effect.
    #random_model.load_state_dict(random_dict)
    random_model = load_victim(50, "cifar10", random_model,
                                   "resnet18", "infonce",
                                   device=device, discard_mlp=True)
victim_model.eval()
stolen_model.eval()
random_model.eval()

for images, truelabels in tqdm(train_loader):
    images = torch.cat(images, dim=0)
    images = images.to(device)
    victim_features = victim_model(images)
    stolen_features = stolen_model(images)
    random_features = random_model(images)
    dist = (victim_features - stolen_features).pow(2).sum(1).sqrt()
    dist2 = (victim_features - random_features).pow(2).sum(1).sqrt()
    #dist = criterion2(victim_features, stolen_features)
    #dist2 = criterion2(victim_features, random_features)
    print(f"distance between stolen and victim where stolen model used loss {args.losstype}", dist.mean())
    print("distance between victim and random", dist2.mean())


