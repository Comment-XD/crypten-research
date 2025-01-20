import os
from datetime import datetime

import glob

import numpy as np
import torch


def compute_accuracy(output, target, topk=(1,)):
    """Computes the precision @k for the specified values of k"""
    pred = output.argmax(1)
    correct = pred.eq(target)
    correct_count = correct.sum(0, keepdim=True).float()

    accuracy = correct_count.mul_(100.0 / output.size(0))
    return accuracy

class AverageMeter(object):
    """Stores values such as loss, batch_time"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)

def check_filepath(fpath):
    """checks to see if a file exists"""
    return os.path.exists(fpath)

def load_checkpoint(model, optimizer, load_path):
    if check_filepath(load_path):
        print('==> Loading...')
        state = torch.load(load_path)
        model.load_from_state_dict(state["model"])
        optimizer.load_from_state_dict(state["optimizer"])

        return state["epoch"]
    
def load_pretrained(args):
    model_name = args.model


    args.save_model_path

def load_optimizer(args, model:torch.nn.Module)->torch.optim:

    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), 
                                lr=args.lr)
    
    if args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), 
                               lr=args.lr, 
                               momentum=args.momentum)

    else:
        raise Exception("Optimizer has to be either Adam or Stochastic Gradient Descent (SGD)")
    

def load_criterion(args):
    if args.loss == "bce":
        return torch.nn.BCELoss()
    
    if args.loss == "ce":
        return torch.nn.CrossEntropyLoss()

    else:
        raise Exception("Loss function has to be either Binary Cross Entropy (BCE) or Cross Entropy (CE")

def save_checkpoint(model, optimizer, epoch, train_losses, val_losses, accuracy, save_path):
    save_string = f"{save_path}/{model.__class__.__name__}_{epoch}.pt"
    
    print('==> Saving...')
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'accuracy': accuracy,
        'epoch': epoch
    }
    
    torch.save(state, save_string)
    del state 

def replace_relu(model):
    pass