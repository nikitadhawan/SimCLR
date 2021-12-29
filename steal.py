import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset, \
    RegularDataset
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
from utils import load_victim

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='/ssd003/home/akaleem/data',
                    help='path to dataset')
parser.add_argument('-dataset', default='cifar10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--archstolen', default='resnet18',
                    choices=model_names,
                    help='stolen model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochstrain', default=200, type=int, metavar='N',
                    help='number of epochs victim was trained with')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
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
parser.add_argument('--num_queries', default=9000, type=int, metavar='N',
                    help='Number of queries to steal the model.')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--folder_name', default='resnet18_100-epochs_cifar10',
                    type=str, help='Pretrained SimCLR model to steal.')
parser.add_argument('--logdir', default='test', type=str,
                    help='Log directory to save output to.')
parser.add_argument('--losstype', default='softce', type=str,
                    help='Loss function to use (softce or infonce)', choices=['softce', 'infonce'])


def main():
    args = parser.parse_args()
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    dataset = RegularDataset(args.data)

    train_dataset = dataset.get_dataset(args.dataset, args.n_views)


    query_dataset = dataset.get_test_dataset(args.dataset, args.n_views)
    indxs = list(range(0, len(query_dataset) - 1000))
    query_dataset = torch.utils.data.Subset(query_dataset,
                                           indxs)  # query set (without last 1000 samples in the test set)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    query_loader = torch.utils.data.DataLoader(
        query_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    victim_model = ResNetSimCLR(base_model=args.arch,
                                  out_dim=args.out_dim).to(args.device)
    victim_model = load_victim(args.epochstrain, args.dataset, victim_model,
                                         device=args.device, discard_mlp=True)
    model = ResNetSimCLR(base_model=args.archstolen, out_dim=args.out_dim, include_mlp = False)

    if args.losstype == "infonce":
        args.lr = 0.0003
        args.batch_size = 256

    optimizer = torch.optim.Adam(model.parameters(), args.lr,   # Maybe change the optimizer
                                 weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(
        train_loader), eta_min=0,
                                                           last_epoch=-1)
    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(stealing=True, victim_model=victim_model,
                        model=model, optimizer=optimizer, scheduler=scheduler,
                        args=args, logdir=args.logdir, loss=args.losstype)
        simclr.steal(query_loader, args.num_queries)


if __name__ == "__main__":
    main()
