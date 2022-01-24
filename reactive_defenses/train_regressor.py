"""
Train regressor to transform the output representations to confidence values.

Training the confidence regressor. We train a two-layer linear network (with
tanh activation) gV for the task of providing confidence about a given data
point’s membership in ‘private’ and ‘public’. data.

The regressor’s loss function is L(x, y) = −y · gV(x) where the label y = 1
for a point in the (public) training set of the respective model,
and −1 if it came from victim’s private set.

"""

import argparse
import os
from getpass import getuser

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import models
from tqdm import tqdm

from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset, \
    RegularDataset
from models.resnet_simclr import ResNetSimCLRV2
from reactive_defenses.custom_dataset import CustomDataset
from reactive_defenses.net_regressor import NetRegressor
from utils import load_victim

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

user = getuser()

if user == 'ahmad':
    prefix = '/ssd003'
else:
    prefix = ''

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR',
                    # default=f'{prefix}/home/{user}/data',
                    default=r"C:\Users\adzie\code\data\cifar10",
                    help='path to dataset')
parser.add_argument('--dataset', default='cifar10',
                    help='dataset name',
                    choices=['stl10', 'cifar10', 'svhn', 'imagenet'])
parser.add_argument('--datasetsteal', default='cifar10',
                    help='dataset used for querying the victim',
                    choices=['stl10', 'cifar10', 'svhn', 'imagenet'])
parser.add_argument('-a', '--arch', metavar='ARCH',
                    default='resnet34',
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
parser.add_argument('-b', '--batch-size',
                    # default=64,
                    default=100,
                    type=int,
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
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    # use 2 to use multiple augmentations.
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--logdir', default='test', type=str,
                    help='Log directory to save output to.')
parser.add_argument('--losstype', default='mse', type=str,
                    help='Loss function to use')
parser.add_argument('--lossvictim', default='infonce', type=str,
                    help='Loss function victim was trained with')
parser.add_argument('--victimhead', default='False', type=str,
                    help='Access to victim head while (g) while getting representations',
                    choices=['True', 'False'])
parser.add_argument('--stolenhead', default='False', type=str,
                    help='Use an additional head while training the stolen model.',
                    choices=['True', 'False'])
parser.add_argument('--defence', default='False', type=str,
                    help='Use defence on the victim side by perturbing outputs',
                    choices=['True', 'False'])
parser.add_argument('--sigma', default=0.5, type=float,
                    help='standard deviation used for perturbations')
parser.add_argument('--mu', default=5, type=float,
                    help='mean noise used for perturbations')
parser.add_argument('--clear', default='True', type=str,
                    help='Clear previous logs', choices=['True', 'False'])
parser.add_argument('--watermark',
                    default='False', type=str,
                    help='Evaluate with watermark model from victim',
                    choices=['True', 'False'])
parser.add_argument('--entropy', default='False', type=str,
                    help='Use entropy victim model', choices=['True', 'False'])
parser.add_argument('--model_type', type=str,
                    default='victim',
                    help='the type of the model',
                    )
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')


def load_victim(epochs, dataset, model, arch, loss, device, discard_mlp=False,
                watermark="False", entropy="False"):
    cwd = os.getcwd()
    print('cwd: ', cwd)
    if user == 'ahmad':
        prefix = f"/checkpoint/{os.getenv('USER')}/SimCLR"
    else:
        prefix = '../../SimCLRmodels'
    if watermark == "True":
        checkpoint = torch.load(
            f"{prefix}/{epochs}{arch}{loss}TRAIN/{dataset}_checkpoint_{epochs}_{loss}WATERMARK.pth.tar",
            map_location=device)
    elif entropy == "True":
        checkpoint = torch.load(
            f"{prefix}/{epochs}{arch}{loss}TRAIN/{dataset}_checkpoint_{epochs}_{loss}ENTROPY.pth.tar",
            map_location=device)
    else:
        checkpoint = torch.load(
            f"{prefix}/{epochs}{arch}{loss}TRAIN/{dataset}_checkpoint_{epochs}_{loss}.pth.tar",
            map_location=device)
    state_dict = checkpoint['state_dict']
    new_state_dict = state_dict.copy()
    if discard_mlp:  # no longer necessary as the model architecture has no backbone.fc layers
        for k in list(state_dict.keys()):
            if k.startswith('backbone.fc'):
                del new_state_dict[k]
        model.load_state_dict(new_state_dict, strict=False)
        return model
    model.load_state_dict(state_dict, strict=False)
    return model


def load_victim_model(args):
    device = args.device

    if args.dataset == "imagenet":
        victim_model = models.resnet50(pretrained=True).to(device)
        victim_model.fc = torch.nn.Identity().to(device)

    # 2048 dimensional output
    elif args.victimhead == "False":
        victim_model = ResNetSimCLRV2(base_model=args.arch,
                                      out_dim=args.out_dim,
                                      loss=args.lossvictim,
                                      include_mlp=False).to(device)
        victim_model = load_victim(args.epochstrain, args.dataset, victim_model,
                                   args.arch, args.lossvictim,
                                   device=device, discard_mlp=True,
                                   watermark=args.watermark,
                                   entropy=args.entropy)
    else:
        victim_model = ResNetSimCLRV2(base_model=args.arch,
                                      out_dim=args.out_dim,
                                      loss=args.lossvictim,
                                      include_mlp=True).to(device)
        victim_model = load_victim(args.epochstrain, args.dataset, victim_model,
                                   args.arch, args.lossvictim,
                                   device=device)

    return victim_model


def get_data_loaders(args):
    dataset_train = ContrastiveLearningDataset(
        args.data)  # RegularDataset(args.data) #

    # this is the dataset the victim was trained on.
    train_dataset = dataset_train.get_dataset(args.dataset,
                                              args.n_views)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    dataset_test = RegularDataset(args.data)
    test_dataset = dataset_test.get_test_dataset(name=args.dataset, n_views=1)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    return train_loader, test_loader


def get_features(data_loader, model, limit=10000):
    num_train = 0
    all_features = []
    for counter, (images, truelabels) in enumerate(tqdm(data_loader)):
        images = torch.cat(images, dim=0)
        images = images.to(device)
        features = model(images)
        features = features.detach().cpu().numpy()
        all_features.append(features)
        num_train += len(images)
        if num_train >= limit:
            break
    all_features = np.concatenate(all_features, axis=0)
    return all_features


def store_features(args):
    victim_model = load_victim_model(args=args)
    victim_model.eval()

    train_loader, test_loader = get_data_loaders(args=args)

    train_features = get_features(data_loader=train_loader, model=victim_model)
    test_features = get_features(data_loader=test_loader, model=victim_model)

    np.save(f'train_features_{args.model_type}.npy', train_features)
    np.save(f'test_features_{args.model_type}.npy', test_features)


def load_features(args):
    train_features = np.load(f'train_features_{args.model_type}.npy')
    test_features = np.load(f'test_features_{args.model_type}.npy')

    print('length train_features: ', len(train_features))
    print('length test_features: ', len(test_features))

    return train_features, test_features


def get_data_loader(train_features, test_features):
    train_labels = np.ones(len(train_features)) * (-1)
    test_labels = np.ones(len(test_features))

    labels = np.concatenate([train_labels, test_labels])
    features = np.concatenate([train_features, test_features])

    dataset = CustomDataset(x=features, y=labels)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False)
    return data_loader


def prepare_dataset(args):
    train_features, test_features = load_features(args=args)

    train_set_frac = 0.9

    num_train = int(train_set_frac * len(train_features))
    train_features_train_set = train_features[:num_train]
    train_features_test_set = train_features[num_train:]

    num_train = int(train_set_frac * len(test_features))
    test_features_train_set = test_features[:num_train]
    test_features_test_set = test_features[num_train:]

    # train set
    train_loader = get_data_loader(train_features=train_features_train_set,
                                   test_features=test_features_train_set)

    # test set
    test_loader = get_data_loader(train_features=train_features_test_set,
                                  test_features=test_features_test_set)

    return train_loader, test_loader


def train_regressor(args):
    model = NetRegressor()
    model = model.to(args.device)
    train_loader, test_loader = prepare_dataset(args=args)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "regressor.pt")


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.sum((-1) * target * output)
        loss.backward()
        optimizer.step()

        pred = (output > 0).to(int)
        pred[pred == 0] = -1
        correct += pred.eq(target.view_as(pred)).sum().item()
        count += len(target)
        acc = correct / count

        if batch_idx % args.log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(),
                    acc))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss = torch.sum((-1) * target * output)
            pred = (output > 0).to(int)
            pred[pred == 0] = -1
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def main(args):
    # store_features(args=args)
    # load_features(args=args)
    train_regressor(args=args)


if __name__ == "__main__":
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    args.device = device

    main(args=args)
