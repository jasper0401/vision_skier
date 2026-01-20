
import os
import argparse
'''Train CIFAR10 with PyTorch.'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
import yaml

from src.data_utils import tiny_imagenet_class

DATA_YAML_PATH = "config/data.yaml" 
with open(DATA_YAML_PATH, 'r') as file:
    # Use safe_load to convert the YAML data into a Python dictionary
    DATA_YAML = yaml.safe_load(file)

"""
# Data
transform_train = transforms.Compose([
    transforms.RandomCrop(image_size, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
"""

def load_cifar(data_folder: str, batch_size: int, crop_size:int=28, padding:int=4, num_workers:int=8):

    normalize = v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomCrop(size=crop_size, padding=padding),
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.1),
        v2.ToDtype(torch.float32, scale=True),
        normalize,
    ])

    val_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        normalize,
    ])

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    class_mapping = {}
    index_to_string = {}
    for i, v in enumerate(classes):
        class_mapping[v] = i
        index_to_string[i] = v

    trainset = torchvision.datasets.CIFAR10(
        root=data_folder, train=True, download=False, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(
        root=data_folder, train=False, download=False, transform=val_transform)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, index_to_string

def load_tiny_image_net(
    train_folder: str,
    val_folder: str,
    wnid_file: str, 
    word_file: str, 
    batch_size: int, 
    image_size:int=64, 
    num_workers:int=8,
    train_transform=None,
    test_transform=None):
    
    normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(size=image_size, antialias=True),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        normalize,
    ])

    val_transform = transforms.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        normalize,
    ])
    
    with open(wnid_file, "r") as reader:
        wordnet_indexes = [x.rstrip("\n") for x in reader.readlines()]

    index_to_string = {} 
    with open(word_file, "r") as reader:
        for line in reader.readlines():
            strs = line.rstrip("\n").split("\t")
            if strs[0] in wordnet_indexes:
                ind = wordnet_indexes.index(strs[0])
                index_to_string[ind] = strs[-1]

    trainset = tiny_imagenet_class.TinyImagenetTrain(
        train_folder, wordnet_indexes, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testset = tiny_imagenet_class.TinyImagenetVal(
        val_folder, wordnet_indexes, transform=val_transform)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, index_to_string

def build_dataset(data_name: str, batch_size:int):
    assert data_name in DATA_YAML, "The dataset specified is unknown!"
    data_meta = DATA_YAML[data_name]

    if data_name == "cifar10":
        train_loader, val_loader, index_to_string = load_cifar(
            data_folder=data_meta["path"], batch_size=batch_size)
    elif data_name == "tiny-imagenet":
        train_loader, val_loader, index_to_string = load_tiny_image_net(
            train_folder=data_meta["train_folder"],
            val_folder=data_meta["val_folder"],
            wnid_file=data_meta["wnid_file"],
            word_file=data_meta["word_file"],
            batch_size=batch_size,
        )
    return train_loader, val_loader, index_to_string, data_meta["image_size"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="cifar10")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    build_dataset(data_name=args.data_name, batch_size=args.batch_size)