import os

import torch
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler

from utils.dataset.folder2lmdb import ImageFolderLMDB


def get_cifar100(train_transform, val_transform, test_transform, CONFIG):
    train_data = datasets.CIFAR100(root=CONFIG.dataset_dir, train=True,
                                   download=True, transform=train_transform)

    test_data = datasets.CIFAR100(root=CONFIG.dataset_dir, train=False,
                                  download=True, transform=test_transform)

    val_data = None
    return train_data, val_data, test_data


def get_cifar10(train_transform, val_transform, test_transform, CONFIG):
    train_data = datasets.CIFAR10(root=CONFIG.dataset_dir, train=True,
                                  download=True, transform=train_transform)

    test_data = datasets.CIFAR10(root=CONFIG.dataset_dir, train=False,
                                 download=True, transform=test_transform)

    val_data = None
    return train_data, val_data, test_data


def get_imagenet_lmdb(train_transform, val_transform, test_transform, CONFIG):
    """
    Load lmdb imagenet dataset
    https://github.com/Fangyh09/Image2LMDB
    """
    train_path = os.path.join(CONFIG.dataset_dir, "train_lmdb", "train.lmdb")
    val_path = os.path.join(CONFIG.dataset_dir, "val_lmdb", "val.lmdb")
    test_path = os.path.join(CONFIG.dataset_dir, "test_lmdb", "test.lmdb")

    train_data = ImageFolderLMDB(train_path, train_transform, None)
    val_data = ImageFolderLMDB(val_path, val_transform, None)
    test_data = ImageFolderLMDB(test_path, test_transform, None)

    return train_data, val_data, test_data


def get_imagenet(train_transform, val_transform, test_transform, CONFIG):
    train_path = os.path.join(CONFIG.dataset_dir, "train")
    val_path = os.path.join(CONFIG.dataset_dir, "val")
    test_path = os.path.join(CONFIG.dataset_dir, "test")

    train_data = datasets.ImageFolder(train_path, train_transform)
    val_data = datasets.ImageFolder(val_path, val_transform)
    test_data = datasets.ImageFolder(test_path, test_transform)

    return train_data, val_data, test_data
