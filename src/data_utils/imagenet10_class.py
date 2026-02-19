
import pathlib as pl

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class ImageNet10(Dataset):
    def __init__(self, data_folder:pl.Path, label_to_index:dict=None, transform=None):
        self.data_folder = pl.Path(data_folder)
        if label_to_index: 
            self.label_to_index = label_to_index
        else:
            self.label_to_index = {}
        self._build_data(self.data_folder)
        self.transform = transform

    def __len__(self):
        return len(self.data) # Return the total number of samples
    
    def _build_data(self, data_folder):
        self.data = []
        for class_folder in data_folder.glob("*"):
            class_name = class_folder.name
            if class_name in self.label_to_index:
                index = self.label_to_index[class_name]
            else:
                index = len(self.label_to_index)
                self.label_to_index[class_name] = index
            #print (class_folder, self.label_to_index) 
            for file in class_folder.glob("*"):
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
