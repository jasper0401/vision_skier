
import os
import argparse
import time
'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import numpy as np

from src.data_utils import dataloader
from src.models import model_builder


# Training
def train(model, train_loader, criterion, optimizer, scaler, device, dtype):
    model.train()
    train_loss = []
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # Runs the forward pass with autocasting.
        with torch.autocast(device_type='cuda', dtype=dtype):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        train_loss.append(loss.item())
        scaler.step(optimizer)
        scaler.update()
    return np.mean(train_loss)

def test(model, val_loader, device):
    model.eval()
    # this confusion statistics is useful when the classes are imbalanced!
    confusion_mat = {}

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            outputs = model(inputs).detach().cpu()

            _, predicted = outputs.max(1)
            for g, p in zip(targets.tolist(), predicted.tolist()):
                if g in confusion_mat:
                    confusion_mat[g].append(p)
                else:
                    confusion_mat[g] = [p]
    
    total = 0
    correct = 0
    class_acc = []
    for k, v in confusion_mat.items():
        total += len(v)
        bool_v = torch.Tensor(v) == k
        class_correct = bool_v.sum()
        correct += class_correct
        class_acc.append(class_correct / len(v))

    print (f"Total Accuracy: {correct / total:.3f}")
    print (f"Avg Accuracy: {np.mean(class_acc):.3f}")
    return confusion_mat

def main(args):

    device = torch.device("cuda") if args.cuda and torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    print (f"Running experiments on {device} using dtype {dtype}")
    train_loader, val_loader, index_to_string, image_size = dataloader.build_dataset(data_name=args.data_name, batch_size=args.batch_size)
    num_classes = len(index_to_string)

    model = model_builder.create_model(args.model_name, image_size=image_size, num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                        momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    scaler = torch.amp.GradScaler()

    if device == torch.device('cuda'):
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    for epoch in range(args.num_epochs):
        start = time.time()
        epoch_loss = train(model, train_loader, criterion, optimizer, scaler, device, dtype)
        elapsed = time.time() - start
        print (f"Epoch: {epoch}, Loss: {epoch_loss}, Elapsed Time: {elapsed:.3f}")
        conf_mat = test(model, val_loader, device)
        scheduler.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="cifar10")
    parser.add_argument("--model_name", type=str, default="ResNet18")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16"])
    args = parser.parse_args()
    main(args)
