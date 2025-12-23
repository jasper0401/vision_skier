import pathlib as pl

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class TinyImagenetTrain(Dataset):
    def __init__(self, train_folder, wordnet_indexes, transform=None):
        self.train_folder = pl.Path(train_folder)
        self.wordnet_indexes = wordnet_indexes

        self._build_data(self.train_folder)
        self.transform = transform

    def __len__(self):
        return len(self.data) # Return the total number of samples
    
    def _build_data(self, train_folder):
        self.data = []
        for class_folder in train_folder.glob("*"):
            index = self.wordnet_indexes.index(class_folder.name)
            for file in (class_folder/"images").glob("*"):
                self.data.append((file, index))


    def __getitem__(self, idx):
        # Construct the full image path and load the image on the fly
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        # Return the sample and label as tensors
        return image, label

class TinyImagenetVal(Dataset):
    def __init__(self, val_folder, wordnet_indexes, transform=None):
        self.val_folder = pl.Path(val_folder)
        self.data = []
        with open(self.val_folder/"val_annotations.txt", "r") as reader:
            for line in reader:
                strs = line.rstrip("\n").split("\t")
                label = wordnet_indexes.index(strs[1])
                self.data.append((self.val_folder/"images"/strs[0], label))
        self.wordnet_indexes = wordnet_indexes
        self.transform = transform

    def __len__(self):
        return len(self.data) # Return the total number of samples
    

    def __getitem__(self, idx):
        # Construct the full image path and load the image on the fly
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        # Return the sample and label as tensors
        return image, label
