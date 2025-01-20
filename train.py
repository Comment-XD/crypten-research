import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from src.model import models
from utils.utils import *
from utils.data_utils import *

import argparse
import time 

def get_args_parser():
    parser = argparse.ArgumentParser("MPC ReLU Comparison")


    # ========== Data/Model Selection ============== # 
    parser.add_argument("--data_path",
                        type=str,
                        help="Path for where the data is being downloaded",
                        default="./data")
    
    parser.add_argument("--dataset",
                        type=str,
                        choices=["mnist", "cifar10", "cifar100", "tiny_imagenet", "imagenet"],
                        help="Choices for which dataset you want for training",
                        default="mnist")
    
    parser.add_argument("--model",
                        type=str,
                        choices=["lenet5", "alexnet", "resnet18", "vgg16", "resnet50"],
                        help="Choices for which model you want for training",
                        default="lenet5")
    
    # =============== Data Loading ============== #

    parser.add_argument("--batch_size",
                        type=int,
                        help="Size of Batches for Training and Validation",
                        default=64)
    
    parser.add_argument("--workers",
                        type=int,
                        help="Number of workers for Data Loading",
                        default=1)
    
    # ============ Optimizer Selection ============== #
    parser.add_argument("--epoches", 
                        type=int, 
                        help="Number of epoches for training",
                        default=100)
    
    parser.add_argument("--lr", 
                        type=float, 
                        help="learning rate for training",
                        default=1e-4)
    
    parser.add_argument("--momentum", 
                        type=float, 
                        help="momentum rate for optimizer",
                        default=1e-4)
    
    parser.add_argument("--optimizer",
                        type=str,
                        choices=["adam", "sgd"],
                        help="What optimizer is used for training",
                        default="adam")
    
    # =========== Loss Selection ============ #
    parser.add_argument("--loss",
                        type=str,
                        choices=["bce", "ce"],
                        help="What Loss function is used for training",
                        default="ce")
    
    # ============ Checkpoint Settings =============== #
    parser.add_argument("--save_model_path", 
                        type=str, 
                        help="",
                        default="./models")
    
    parser.add_argument("--save_check_freq", 
                        type=int, 
                        default=10, 
                        help="How often the model is saved every epoch")
    
    parser.add_argument("--save_check_path", 
                        type=str, 
                        default="./checkpoints", 
                        help="Save path for checkpoints")
    
    parser.add_argument("--resume",
                        type=bool,
                        default=False,
                        help="Resumes Training from a previous checkpoint")
    
    args = parser.parse_args()

    return args

class Trainer:
    def __init__(self, 
                 model,
                 train_dataloader,
                 val_dataloader,
                 optimizer, 
                 loss_function) -> None:
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.loss_function = loss_function

        self.train_losses = []
        self.val_losses = []
        
        
    def train_one_epoch(self):
        self.model.train()
        
        train_loss = AverageMeter() 
        batch_time = AverageMeter()
        
        start = time.time()
        
        for _, (X,y) in tqdm(enumerate(self.train_dataloader)):
            X = X.to(self.device)
            y = y.to(self.device)
            
            with torch.autocast(device_type=self.device):
                # Later on for distributed parallel training
                pass
            
            y_pred = self.model(X)
            loss = self.loss_function(y_pred, y)
            train_loss.update(loss.item())
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # train_accuracy += accuracy(y_pred, y)
            
        batch_time.update(time.time() - start)
        self.train_losses.append(train_loss.avg)
        # train_accuracy /= len(self.train_dataloader)
        print(f"Train Batch Elapsed: {batch_time.val: .4f}(s) | Train Loss: {train_loss.avg : .4f}\n")
        
    
    def val_one_epoch(self):
        self.model.eval()
        
        val_loss = AverageMeter()
        batch_time = AverageMeter()
        
        start = time.time()
        with torch.no_grad():
            for _, (X,y) in tqdm(enumerate(self.val_dataloader)):
                X = X.to(self.device)
                y = y.to(self.device)
                
                y_pred = self.model(X)
                loss = self.loss_function(y_pred, y)
                
                val_loss.update(loss.item())
                batch_time.update(time.time() - start)

        self.val_losses.append(val_loss.avg)
        print(f"Validation Batch Elapsed: {batch_time.val: .4f}(s) | Validation Loss: {val_loss.avg : .4f}\n")
    
    def run(self, args):
        
        #TODO: Check to see args is training from checkpoint
        start_epoch = 0
        
        # if args.resume:
        #     start_epoch = load_checkpoint(self.model, self.optimizer)
        
        for epoch in range(start_epoch, args.epoches):
            print(f"Epoch {epoch + 1}\n")
            self.train_one_epoch()
            self.val_one_epoch()

            if epoch % args.save_check_freq == 0:

                # TODO: Add Accuracy and if the model is best or not
                save_checkpoint(self.model, 
                                self.optimizer, 
                                epoch, 
                                self.train_losses,
                                self.val_losses,
                                None,
                                args.save_check_path)
                
        save_checkpoint(self.model, 
                            self.optimizer, 
                            epoch, 
                            self.train_losses,
                            self.val_losses,
                            None,
                            args.save_model_path)
        
    def plot_losses(self):
        pass

def main():
    args = get_args_parser()
    train_dataloader, val_dataloader = load_dataloader(args)

    model = models[args.model]

    optimizer = load_optimizer(args, model)
    criterion = load_criterion(args)

    trainer = Trainer(model, 
                      train_dataloader,
                      val_dataloader,
                      optimizer,
                      criterion)
    
    trainer.run(args)

if __name__ == "__main__":
    main()