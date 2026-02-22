
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
from src.loss import contrastive_loss as loss_func


# Training
def train(model, train_loader, criterion, optimizer, scaler, device, dtype):
    model.train()
    train_loss = []
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        """
        optimizer.zero_grad()

        # Runs the forward pass with autocasting.
        with torch.autocast(device_type='cuda', dtype=dtype):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        train_loss.append(loss.item())
        scaler.step(optimizer)
        scaler.update()
        """
    return None

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

    image_encoder, text_encoder = model_builder.create_nano_clip(device=device, dtype=dtype) 

    loss_function = loss_func.ContrastiveLoss()

    optimizer_params = [
        {"params": image_encoder.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": text_encoder.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
    ]
    optimizer = torch.optim.AdamW(optimizer_params)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.lr_mult
    )  

    data_loader = dataloader.build_dataset(data_name=args.data_name, batch_size=args.batch_size, tokenizer=text_encoder.tokenizer)
    for epoch in range(args.num_epochs):
        for batch in data_loader:
            optimizer.zero_grad()
            image, text_tokens = batch
            image_features = image_encoder(image.to(device))
            text_features = text_encoder(input_ids=text_tokens["input_ids"].to(device).squeeze(1), attention_mask=text_tokens["attention_mask"].to(device).squeeze(1))
            loss, acc  = loss_function(image_features, text_features)
            loss.backward()
            optimizer.step()
            print (f"Epoch: {epoch}, Loss: {loss.item():.3f}, Acc: {acc.item():.3f}")
            break
        scheduler.step()

    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                    weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    scaler = torch.amp.GradScaler()

    #if device == torch.device('cuda'):
    #    model = torch.nn.DataParallel(model)
    #    cudnn.benchmark = True

    for epoch in range(args.num_epochs):
        start = time.time()
        epoch_loss = train(model, train_loader, criterion, optimizer, scaler, device, dtype)
        elapsed = time.time() - start
        print (f"Epoch: {epoch}, Loss: {epoch_loss}, Elapsed Time: {elapsed:.3f}")
        conf_mat = test(model, val_loader, device)
        scheduler.step()
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="flickr8k")
    parser.add_argument("--vision_model", type=str, default="convnext_tiny")
    parser.add_argument("--text_model", type=str, default="convnext_tiny")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr_mult", type=float, default=0.1)
    parser.add_argument("--milestones", nargs="+", type=int, default=[30, 40])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16"])
    args = parser.parse_args()
    main(args)
