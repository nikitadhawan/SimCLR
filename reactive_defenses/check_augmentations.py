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
import time
from getpass import getuser

import numpy as np
import torch
from torchvision import models
from torchvision.transforms import transforms
from tqdm import tqdm

from data_aug.contrastive_learning_dataset import RegularDataset
from data_aug.gaussian_blur import GaussianBlur
from models.resnet_simclr import ResNetSimCLRV2
from statistical_tests.t_test import ttest
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
                    default=r"C:\Users\adzie\code\data",
                    help='path to dataset')
parser.add_argument('--dataset_train',
                    # default='cifar10',
                    default='svhn',
                    help='dataset name train',
                    choices=['stl10', 'cifar10', 'svhn', 'imagenet'])
parser.add_argument('--dataset_test',
                    # default='cifar10',
                    default='svhn',
                    help='dataset name test',
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
        model_path = f"{prefix}/{epochs}{arch}{loss}TRAIN/{dataset}_checkpoint_{epochs}_{loss}.pth.tar"
        print(f"model_path: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
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

    if args.dataset_train == "imagenet":
        victim_model = models.resnet50(pretrained=True).to(device)
        victim_model.fc = torch.nn.Identity().to(device)

    # 2048 dimensional output
    elif args.victimhead == "False":
        victim_model = ResNetSimCLRV2(base_model=args.arch,
                                      out_dim=args.out_dim,
                                      loss=args.lossvictim,
                                      include_mlp=False).to(device)
        victim_model = load_victim(args.epochstrain, args.dataset_train,
                                   victim_model,
                                   args.arch, args.lossvictim,
                                   device=device, discard_mlp=True,
                                   watermark=args.watermark,
                                   entropy=args.entropy)
    else:
        victim_model = ResNetSimCLRV2(base_model=args.arch,
                                      out_dim=args.out_dim,
                                      loss=args.lossvictim,
                                      include_mlp=True).to(device)
        victim_model = load_victim(args.epochstrain, args.dataset_train,
                                   victim_model,
                                   args.arch, args.lossvictim,
                                   device=device)

    return victim_model


def get_data_loaders(args):
    # This is the raw dataset.

    # train
    dataset_train = RegularDataset(args.data)
    train_dataset = dataset_train.get_train_dataset(args.dataset_train, n_views=1)
    # train_dataset = dataset_train.get_dataset(name='cifar10', n_views=1)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    # test
    dataset_test = RegularDataset(args.data)
    test_dataset = dataset_test.get_test_dataset(name=args.dataset_test,
                                                 n_views=1)
    # test_dataset = dataset_test.get_test_dataset(name='svhn', n_views=1)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    return train_loader, test_loader


def get_diffs(data_loader, model, dataset_name, limit=100):
    num_train = 0
    all_diffs = []

    if dataset_name == 'imagenet':
        size = 224
    else:
        size = 32
    s = 1
    color_jitter = transforms.ColorJitter(
        0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    data_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=int(0.1 * size)),
        ])

    for counter, (images, truelabels) in enumerate(tqdm(data_loader)):
        images = images[0]

        raw_images = images.to(device)
        raw_features = model(raw_images)

        # Number of augmentations for a single data point.
        num_augments = 100
        for _ in range(num_augments):
            augment_images = []
            for image in images:
                aug_image = to_pil(image)
                aug_image = data_transforms(aug_image)
                aug_image = to_tensor(aug_image)
                augment_images.append(aug_image)

            augment_images = torch.stack(augment_images)
            augment_images = augment_images.to(device)
            augment_features = model(augment_images)

            diff = torch.pow(raw_features - augment_features, 2)
            diff = torch.sum(diff, dim=-1)
            diff = torch.sqrt(diff)
            diff = diff.detach().cpu().numpy()
            all_diffs.extend(diff)

        num_train += len(images)
        if num_train >= limit:
            break

    return all_diffs


def store_diffs(args):
    victim_model = load_victim_model(args=args)
    victim_model.eval()

    train_loader, test_loader = get_data_loaders(args=args)

    train_diffs = get_diffs(data_loader=train_loader, model=victim_model,
                            dataset_name=args.dataset_train)
    test_diffs = get_diffs(data_loader=test_loader, model=victim_model,
                           dataset_name=args.dataset_test)

    np.save(f'train_diffs_{args.model_type}.npy', train_diffs)
    np.save(f'test_diffs_{args.model_type}.npy', test_diffs)


def load_diffs(args):
    train_diffs = np.load(f'train_diffs_{args.model_type}.npy')
    test_diffs = np.load(f'test_diffs_{args.model_type}.npy')

    print('length train_diffs: ', len(train_diffs))
    print('length test_diffs: ', len(test_diffs))

    return train_diffs, test_diffs


def run_ttest(train_diffs, test_diffs, args):
    mean_train = np.mean(train_diffs)
    print(f'mean_train {args.dataset_train} distances: ', mean_train)
    print('median_train: ', np.median(train_diffs))
    mean_test = np.mean(test_diffs)
    print(f'mean_test {args.dataset_test} distances: ', mean_test)
    print('median_test: ', np.median(test_diffs))
    tval, pval = ttest(test_diffs, train_diffs, alternative="greater")
    print('Null hypothesis: distances test <= distances train')
    print('tval: ', tval, ' pval: ', pval, ' delta u: ', mean_test - mean_train)


def main(args):
    start = time.time()
    store_diffs(args=args)
    train_diffs, test_diffs = load_diffs(args=args)
    run_ttest(train_diffs=train_diffs, test_diffs=test_diffs, args=args)
    stop = time.time()
    print('elapsed time: ', stop - start)


if __name__ == "__main__":
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    args.device = device

    main(args=args)
