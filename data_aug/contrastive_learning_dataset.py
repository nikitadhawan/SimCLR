from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator, WatermarkViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
import os


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms



    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10", split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True),

                          'svhn': lambda: datasets.SVHN(self.root_folder+"/SVHN",
                                                          split='train',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(
                                                                  32),
                                                              n_views),
                                                          download=True),

                          }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
    def get_test_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=False,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32), # verify if we use the transform here. also need the option for multiple augmentations possibly in the main code
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10", split='test',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True),

                          'svhn': lambda: datasets.SVHN(self.root_folder+"/SVHN",
                                                        split='test',
                                                        transform=ContrastiveLearningViewGenerator(
                                                            self.get_simclr_pipeline_transform(
                                                                32),
                                                            n_views),
                                                        download=True),
                          }


        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
        
        
class RegularDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        data_transforms = transforms.Compose([transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True, 
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32), n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10", split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True),

                          'svhn': lambda: datasets.SVHN(self.root_folder+"/SVHN",
                                                        split='train',
                                                        transform=ContrastiveLearningViewGenerator(
                                                            self.get_simclr_pipeline_transform(
                                                                32),
                                                            n_views),
                                                        download=True),

                          }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
    def get_test_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=False,
                                                              transform=ContrastiveLearningViewGenerator(  
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10", split='test',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True),

                          'svhn': lambda: datasets.SVHN(self.root_folder+"/SVHN",
                                                        split='test',
                                                        transform=ContrastiveLearningViewGenerator(
                                                            self.get_simclr_pipeline_transform(
                                                                32),
                                                            n_views),
                                                        download=True),
                          'imagenet': lambda: datasets.ImageNet(
                              root="/scratch/ssd002/datasets/imagenet256/",
                              split='val',
                              transform=ContrastiveLearningViewGenerator(
                                  self.get_imagenet_transform(
                                      32),
                                  n_views))
                          }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()

if __name__ == "__main__":
    import getpass
    user = getpass.getuser()
    from torch.utils.data import ConcatDataset, DataLoader
    import numpy as np
    # Important: color jitter does not work with 1 channel images
    def get_simclr_pipeline_transform(size, s=1):
        data_transforms = transforms.Compose([transforms.ToTensor()])
        return data_transforms
    n_views = 2
    # testing combined dataset with fmnist and mnist
    fmnist = datasets.FashionMNIST(f'/ssd003/home/{user}/data/', train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  get_simclr_pipeline_transform(28),
                                                                  n_views),
                                                              download=False)
    #print(fmnist.targets)  # from 0 to 9.
    print("length fmnist", len(fmnist))
    mnist = datasets.MNIST(f'/ssd003/home/{user}/data/', train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  get_simclr_pipeline_transform(28),
                                                                  n_views),
                                                              download=False)
    #print(mnist.targets) # from 0 to 9
    idx = (mnist.targets == 2) | (mnist.targets == 3)
    mnist.targets = mnist.targets[idx] # only select 2's and 3's from mnist
    mnist.data = mnist.data[idx]
    mnist.targets = mnist.targets + 8 # to make 2's be labeled 10, 3's as 11 so they can be combined with fmnist
    print("length mnist", len(mnist))

    combined = ConcatDataset([fmnist,mnist])
    print("length combined", len(combined))
    combined_loader = DataLoader(
        combined, batch_size=64, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True)
    print(next(iter(combined_loader)))
