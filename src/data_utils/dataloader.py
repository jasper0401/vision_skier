
import os
import argparse
import pathlib as pl
import random
from collections import defaultdict
'''Train CIFAR10 with PyTorch.'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
from transformers import AutoTokenizer
import yaml

from src.data_utils import tiny_imagenet_class
from src.data_utils import imagenet10_class

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

def load_imagenet10(
    train_folder: pl.Path,
    val_folder: pl.Path,
    batch_size: int, 
    image_size:int=64, 
    num_workers:int=8,
    ):
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
        v2.Resize(size=image_size, antialias=True),
        v2.CenterCrop(size=image_size),
        v2.ToDtype(torch.float32, scale=True),
        normalize,
    ])
    training_set = imagenet10_class.ImageNet10(
        data_folder=train_folder,
        transform=train_transform,
    )
    train_loader = torch.utils.data.DataLoader(
        training_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_set = imagenet10_class.ImageNet10(
        data_folder=val_folder,
        label_to_index=training_set.label_to_index,
        transform=val_transform,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    index_to_string = {}
    for k, v in training_set.label_to_index.items():
        index_to_string[v] = k
    return train_loader, test_loader, index_to_string

class Flickr8kDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_folder: pl.Path,
        caption_file: pl.Path,
        tokenizer: AutoTokenizer,
        transform=None,
    ):
        self.image_folder = image_folder
        self.transform = transform
        self.tokenizer = tokenizer

        self.image_files = []
        self.labels = defaultdict(list)

        with open(caption_file, "r") as reader:
            for line in reader.readlines()[1:]:
                strs = line.rstrip("\n").split(",")
                img_name = strs[0].strip()
                caption = strs[1].replace(" .", ".")
                self.labels[img_name].append(caption)
        self.data = list(self.labels.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = torchvision.io.read_image(img_path)
        if self.transform is not None:
            image = self.transform(image)
        label = random.choice(self.labels[img_name])
        text_tokens = self.tokenizer(label, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
        return image, text_tokens

def load_flickr8k(
    image_folder: pl.Path,
    caption_file: pl.Path,
    batch_size: int,
    tokenizer: AutoTokenizer,
    image_size:int=64,
    num_workers:int=8,
):
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
        v2.Resize(size=image_size, antialias=True),
        v2.CenterCrop(size=image_size),
        v2.ToDtype(torch.float32, scale=True),
        normalize,
    ])

    training_set = Flickr8kDataset(
        image_folder=image_folder,
        caption_file=caption_file,
        tokenizer=tokenizer,
        transform=train_transform,
    )
    data_loader = torch.utils.data.DataLoader(
        training_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return data_loader


def build_dataset(data_name: str, batch_size:int, tokenizer: AutoTokenizer=None):
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
    elif data_name.startswith("imagenet10"):
        train_loader, val_loader, index_to_string = load_imagenet10(
            train_folder=data_meta["train_folder"],
            val_folder=data_meta["val_folder"],
            batch_size=batch_size,
            image_size=data_meta["image_size"],
        ) 
    elif data_name == "flickr8k":
        data_loader = load_flickr8k(
            image_folder=data_meta["image_folder"],
            caption_file=data_meta["caption"],
            tokenizer=tokenizer,
            batch_size=batch_size,
            image_size=data_meta["image_size"],
        )
        return data_loader
    return train_loader, val_loader, index_to_string, data_meta["image_size"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="cifar10")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    build_dataset(data_name=args.data_name, batch_size=args.batch_size)