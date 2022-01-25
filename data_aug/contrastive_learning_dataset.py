from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator, \
    WatermarkViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
import os


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s,
                                              0.2 * s)
        data_transforms = transforms.Compose(
            [transforms.RandomResizedCrop(size=size),
             transforms.RandomHorizontalFlip(),
             transforms.RandomApply([color_jitter], p=0.8),
             transforms.RandomGrayscale(p=0.2),
             GaussianBlur(kernel_size=int(0.1 * size)),
             transforms.ToTensor()])
        return data_transforms

    @staticmethod
    def get_imagenet_transform(size, s=1):
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views, size=224):
        valid_datasets = {
            'cifar10': lambda: datasets.CIFAR10(
                self.root_folder, train=True,
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(size), n_views),
                download=True),

            'stl10': lambda: datasets.STL10(
                f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10",
                split='unlabeled',
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(size),
                    n_views),
                download=True),

            'svhn': lambda: datasets.SVHN(
                self.root_folder + "/SVHN",
                split='train',
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(
                        size),
                    n_views),
                download=True),
            'imagenet': lambda: datasets.ImageNet(
                root="/scratch/ssd002/datasets/imagenet256/",
                split='train',
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(size),
                    n_views))
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()

    def get_test_dataset(self, name, n_views):
        valid_datasets = {
            'cifar10': lambda: datasets.CIFAR10(
                self.root_folder, train=False,
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(
                        32),
                    # verify if we use the transform here. also need the option for multiple augmentations possibly in the main code
                    n_views),
                download=True),

            'stl10': lambda: datasets.STL10(
                f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10", split='test',
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(96),
                    n_views),
                download=True),

            'svhn': lambda: datasets.SVHN(self.root_folder + "/SVHN",
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
                    self.get_simclr_pipeline_transform(
                        224),
                    n_views))
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

    @staticmethod
    def get_imagenet_transform(size, s=1):
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.ToTensor()])
        return data_transforms

    def get_train_dataset(self, name, n_views, size=224):
        valid_datasets = {
            'cifar10': lambda: datasets.CIFAR10(
                self.root_folder, train=True,
                # transform=ContrastiveLearningViewGenerator(
                #     self.get_simclr_pipeline_transform(
                #         size), n_views),
                transform=transforms.Compose(
                    [
                        transforms.Pad(4),
                        transforms.RandomCrop(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.49139969, 0.48215842, 0.44653093),
                            (0.24703223, 0.24348513, 0.26158784),
                        ),
                    ]
                ),
                download=True),

            'stl10': lambda: datasets.STL10(
                f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10",
                split='unlabeled',
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(size),
                    n_views),
                download=True),

            'svhn': lambda: datasets.SVHN(
                self.root_folder + "/SVHN",
                split='train',
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.43768212, 0.44376972, 0.47280444),
                            (0.19803013, 0.20101563, 0.19703615),
                        ),
                    ]
                ),
                download=True),

            'imagenet': lambda: datasets.ImageNet(
                # root="/scratch/ssd002/datasets/imagenet256/",
                root=os.path.join(self.root_folder, 'imagenet'),
                split='train',
                transform=ContrastiveLearningViewGenerator(
                    self.get_imagenet_transform(size), n_views))
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()

    def get_test_dataset(self, name, n_views, size=224):
        valid_datasets = {
            'cifar10': lambda: datasets.CIFAR10(
                self.root_folder, train=False,
                # transform=ContrastiveLearningViewGenerator(
                #     self.get_simclr_pipeline_transform(
                #         size),
                #     n_views),
                transform=transforms.Compose(
                    [
                        transforms.Pad(4),
                        transforms.RandomCrop(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.49139969, 0.48215842, 0.44653093),
                            (0.24703223, 0.24348513, 0.26158784),
                        ),
                    ]
                ),
                download=True),

            'stl10': lambda: datasets.STL10(
                f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10", split='test',
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(size),
                    n_views),
                download=True),

            'svhn': lambda: datasets.SVHN(
                self.root_folder + "/SVHN",
                split='test',
                # transform=ContrastiveLearningViewGenerator(
                #     self.get_simclr_pipeline_transform(
                #         size),
                #     n_views),
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.43768212, 0.44376972, 0.47280444),
                            (0.19803013, 0.20101563, 0.19703615),
                        ),
                    ]
                ),
                download=True),

            'imagenet': lambda: datasets.ImageNet(
                # root="/scratch/ssd002/datasets/imagenet256/",
                root=os.path.join(self.root_folder, 'imagenet'),
                split='val',
                transform=ContrastiveLearningViewGenerator(
                    self.get_imagenet_transform(size), n_views))
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()


class WatermarkDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_transform():
        data_transform1 = transforms.Compose(
            [transforms.RandomRotation(degrees=(0, 180)),
             transforms.ToTensor()])
        data_transform2 = transforms.Compose(
            [transforms.RandomRotation(degrees=(180, 360)),
             transforms.ToTensor()])
        return [data_transform1, data_transform2]

        # data_transform1 = transforms.Compose(
        #     [transforms.RandomRotation(degrees=(0, 90)),
        #      transforms.ToTensor()])
        # data_transform2 = transforms.Compose(
        #     [transforms.RandomRotation(degrees=(90, 180)),
        #      transforms.ToTensor()])
        # data_transform3 = transforms.Compose(
        #     [transforms.RandomRotation(degrees=(180, 270)),
        #      transforms.ToTensor()])
        # data_transform4 = transforms.Compose(
        #     [transforms.RandomRotation(degrees=(270, 360)),
        #      transforms.ToTensor()])
        # return [data_transform1, data_transform2, data_transform3, data_transform4]

    @staticmethod
    def get_imagenet_transform(size, s=1):
        data_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.ToTensor()])
        data_transform2 = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomRotation(degrees=(180, 360)),
            transforms.ToTensor()])
        return [data_transform1, data_transform2]

    def get_dataset(self, name, n_views):
        valid_datasets = {
            'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                transform=WatermarkViewGenerator(
                                                    self.get_transform(),
                                                    n_views),
                                                download=True),

            'stl10': lambda: datasets.STL10(
                f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10",
                split='unlabeled',
                transform=WatermarkViewGenerator(
                    self.get_transform(),
                    n_views),
                download=True),
            'svhn': lambda: datasets.SVHN(
                self.root_folder + "/SVHN",
                split='test',
                transform=WatermarkViewGenerator(
                    self.get_transform(),
                    n_views),
                download=True),
            'imagenet': lambda: datasets.ImageNet(
                root="/scratch/ssd002/datasets/imagenet256/",
                split='val',
                transform=WatermarkViewGenerator(
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


class ImageNetDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        data_transforms = transforms.Compose([transforms.ToTensor()])
        return data_transforms

    @staticmethod
    def get_imagenet_transform(size, s=1):
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {
            'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                transform=transforms.Compose([
                                                    transforms.RandomCrop(32,
                                                                          padding=4),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    # transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                    #                      (0.2023, 0.1994, 0.2010)),
                                                    transforms.Resize(224),
                                                ]),
                                                download=False),

            'stl10': lambda: datasets.STL10(
                f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10",
                split='unlabeled',
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(224),
                ]),
                download=True),

            'svhn': lambda: datasets.SVHN(self.root_folder + "/SVHN",
                                          split='train',
                                          transform=transforms.Compose([
                                              transforms.RandomCrop(32,
                                                                    padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              # transforms.Normalize((0.4914, 0.4822, 0.4465),
                                              #                      (0.2023, 0.1994, 0.2010)),
                                              transforms.Resize(224),
                                          ]),
                                          download=True),
            'imagenet': lambda: datasets.ImageNet(
                root="/scratch/ssd002/datasets/imagenet256/",
                split='train',
                transform=self.get_imagenet_transform(224))
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()

    def get_test_dataset(self, name, n_views):
        valid_datasets = {
            'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    # transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                    #                      (0.2023, 0.1994, 0.2010)),
                                                    transforms.Resize(224),
                                                ]),
                                                download=False),

            'stl10': lambda: datasets.STL10(
                f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10", split='test',
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Resize(
                            224),
                    ]),
                download=True),

            'svhn': lambda: datasets.SVHN(self.root_folder + "/SVHN",
                                          split='train',
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              # transforms.Normalize((0.4914, 0.4822, 0.4465),
                                              #                      (0.2023, 0.1994, 0.2010)),
                                              transforms.Resize(224),
                                          ]),
                                          download=True),
            'imagenet': lambda: datasets.ImageNet(
                root="/scratch/ssd002/datasets/imagenet256/",
                split='val',
                transform=self.get_imagenet_transform(224))
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
