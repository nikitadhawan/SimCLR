import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset, \
    RegularDataset, WatermarkDataset
from models.resnet_simclr import ResNetSimCLRV2, SimSiam, WatermarkMLP
from simclr import SimCLR
from utils import load_victim, load_watermark
import os

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
parser.add_argument('--losstype', default='infonce', type=str,
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

def main():
    args = parser.parse_args()
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    if args.losstype in  ["infonce", "softnn", "supcon", "barlow"]:
        args.batch_size = 256
        args.weight_decay = 1e-4
        args.n_views = 2
    if args.losstype == "infonce":
        args.lr = 0.0003
    if args.losstype == "supcon":
        args.lr = 0.05
    if args.losstype == "softnn":
        args.lr = 0.001
    if args.losstype == "symmetrized":
        args.batch_size = 256
        args.lr = 0.05
        args.out_dim = 512
        args.n_views = 2
        args.stolenhead = "True"
    if args.losstype in ["mse", "softce", "wassersein"]:
        args.n_views = 1
    if args.stolenhead == "True" and args.victimhead == "False":
        args.out_dim = 512 # since representations are 512 dimensional
    if args.n_views == 1:
        dataset = RegularDataset(args.data)
    elif args.n_views == 2:
        dataset = ContrastiveLearningDataset(args.data) # using data augmentation for queries
    if args.dataset == "imagenet":
        args.arch = "resnet50"
        args.archstolen = "resnet50"
        args.out_dim = 512

    print("args", args)

    #train_dataset = dataset.get_dataset(args.dataset, args.n_views)

    if args.datasetsteal != args.dataset:
        query_dataset = dataset.get_dataset(args.datasetsteal, args.n_views)
        indxs = list(range(0, len(query_dataset)))
    else:
        query_dataset = dataset.get_test_dataset(args.datasetsteal,
                                                 args.n_views)
        indxs = list(range(0, len(query_dataset) - 1000))

    query_dataset = torch.utils.data.Subset(query_dataset,
                                           indxs)  # query set (without last 1000 samples as they are used in the test set)
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True, drop_last=True)

    query_loader = torch.utils.data.DataLoader(
        query_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    if args.dataset == "imagenet":
        victim_model = models.resnet50(pretrained=True)
        victim_model.fc = torch.nn.Identity()
        # 2048 dimensional output
    else:
        if args.victimhead == "False":
            victim_model = ResNetSimCLRV2(base_model=args.arch,
                                          out_dim=args.out_dim,loss=args.lossvictim, include_mlp = False).to(args.device)
            victim_model = load_victim(args.epochstrain, args.dataset, victim_model,
                                       args.arch, args.lossvictim,
                                       device=args.device, discard_mlp = True, watermark=args.watermark, entropy=args.entropy)
        else:
            victim_model = ResNetSimCLRV2(base_model=args.arch,
                                          out_dim=args.out_dim,loss=args.lossvictim, include_mlp = True).to(args.device)
            victim_model = load_victim(args.epochstrain, args.dataset, victim_model,
                                       args.arch, args.lossvictim,
                                       device=args.device)
        if args.defence == "True": # Use the model head as part of the defence.
            victim_head = ResNetSimCLRV2(base_model=args.arch,
                                          out_dim=args.out_dim,
                                          loss=args.lossvictim,
                                          include_mlp=True).to(args.device)
            victim_head = load_victim(args.epochstrain, args.dataset,
                                       victim_head,
                                       args.arch, args.lossvictim,
                                       device=args.device, entropy=args.entropy)
            # model to be used for entropy calculation (assumes specific downstream task being used)
            entropy_model = models.resnet50(pretrained=False,
                                                num_classes=10).to(args.device)
            entropy_model.load_state_dict(torch.load(f"/checkpoint/{os.getenv('USER')}/SimCLR/downstream/victim_linear_{args.datasetsteal}.pth.tar"))
        if args.watermark == "True":
            watermark_dataset = WatermarkDataset(args.data).get_dataset(
                args.dataset, args.n_views)
            watermark_loader = torch.utils.data.DataLoader(
                watermark_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True, drop_last=True)
            if args.stolenhead == "False":
                watermark_mlp = WatermarkMLP(512,2)
            else:
                watermark_mlp = WatermarkMLP(128, 2)
            watermark_mlp = load_watermark(args.epochstrain, args.dataset,
                                       watermark_mlp,
                                       args.arch, args.lossvictim,
                                       device=args.device)
    if args.stolenhead == "False":
        model = ResNetSimCLRV2(base_model=args.archstolen, out_dim=args.out_dim, loss=args.losstype, include_mlp = False)
    else:
        model = ResNetSimCLRV2(base_model=args.archstolen, out_dim=args.out_dim,loss=args.losstype,
                               include_mlp=True)

    if args.losstype == "symmetrized":
        model = SimSiam(models.__dict__[args.arch], args.out_dim, args.out_dim)

    optimizer = torch.optim.Adam(model.parameters(), args.lr,   # Maybe change the optimizer
                                 weight_decay=args.weight_decay)

    if args.losstype in ["supcon", "symmetrized"]:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(
        query_loader), eta_min=0,last_epoch=-1)
    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        if args.defence == "True":
            simclr = SimCLR(stealing=True, victim_model=victim_model, victim_head=victim_head,entropy_model=entropy_model,
                            model=model, optimizer=optimizer, scheduler=scheduler,
                            args=args, logdir=args.logdir, loss=args.losstype)
            simclr.steal(query_loader, args.num_queries)
        elif args.watermark == "True":
            simclr = SimCLR(stealing=True, victim_model=victim_model,
                            watermark_mlp=watermark_mlp,
                            model=model, optimizer=optimizer,
                            scheduler=scheduler,
                            args=args, logdir=args.logdir, loss=args.losstype)
            simclr.steal(query_loader, args.num_queries, watermark_loader)
        else:
            simclr = SimCLR(stealing=True, victim_model=victim_model,
                            model=model, optimizer=optimizer,
                            scheduler=scheduler,
                            args=args, logdir=args.logdir, loss=args.losstype)
            simclr.steal(query_loader, args.num_queries)


if __name__ == "__main__":
    main()
