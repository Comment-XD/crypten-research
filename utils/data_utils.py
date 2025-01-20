import argparse
import os

from typing import Tuple

import torch
from torchvision.datasets import *
import torchvision.transforms as transforms

def load_dataset(args: argparse.ArgumentParser) -> torch.Tensor:
    """
    Loads a dataset depending on the dataset name


    Args:
        args (argparse.ArgumentParser): argument parser

    Returns:
        torch.Tensor:: Dataset in the form Torch Tensors
    """

    train_dataset, val_dataset = None, None
    # Check if the data folder has the dataset if not download in that root
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)
    
    if args.dataset == "mnist":
        mnist_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(32),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = MNIST(root=args.data_path, 
                              train=True,      
                              download=True, 
                              transform=mnist_transforms)
        
        val_dataset = MNIST(root=args.data_path, 
                            train=False,
                            download=True, 
                            transform=mnist_transforms)

    if args.dataset == "cifar10":
        train_dataset = CIFAR10(root=args.data_path, 
                                train=True,
                                download=True, 
                                transform=mnist_transforms)
        
        val_dataset = CIFAR10(root=args.data_path, 
                              train=False,
                              download=True, 
                              transform=mnist_transforms)

    if args.dataset == "cifar100":
        train_dataset = CIFAR100(root=args.data_path, 
                                 train=True,
                                 download=True, 
                                 transform=mnist_transforms)
        
        val_dataset = CIFAR100(root=args.data_path, 
                               train=False,
                               download=True, 
                               transform=mnist_transforms)

    if args.dataset == "tiny_imagenet":
        pass

    if args.dataset == "imagenet":
        dataset = ImageNet()

    
    return train_dataset, val_dataset

def load_dataloader(args: argparse.ArgumentParser) -> Tuple[torch.utils.data.DataLoader]:
    """
    Returns a Dataloader object depending on the batch_size and data

    Args:
        args (argparse.ArgumentParser): argument parser

    Returns:
        torch.utils.data.DataLoader: Dataloader from PyTorch
    """

    train_dataset, val_dataset = load_dataset(args)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                             batch_size=args.batch_size, 
                                             shuffle=True, 
                                             num_workers=args.workers,
                                             drop_last=True,
                                             pin_memory=True)

    val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                             batch_size=args.batch_size, 
                                             shuffle=False, 
                                             num_workers=args.workers,
                                             pin_memory=True)
    
    return train_dataloader, val_dataloader

