import argparse

import torch
import torch.nn as nn

# from torch.amp import autocast_mode
# from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.utils import *
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import time
from tqdm.auto import tqdm

from src.model import models
import crypten

def get_args_parser():
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument("--batch_size",  type=int, default=64, help="batch size for data loading")
    parser.add_argument("--num_workers", type=int, default=1, help="workers for data loading")
    
    parser.add_argument("--epoches", type=int, default=10, help="number of epoches for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for training")
    
    parser.add_argument("--optim_type", type=str, default="adam", 
                        choices=["adam", "sgd"], help="which optimizer type to use")
    parser.add_argument("--decay_rate", type=float, default=1e-4, help="decay rate for optimizer")
     
    parser.add_argument("--arch", type=str, default="resnet18", 
                        choices=["resnet18, resnet34, resnet50, alexnet"], help="")
    
    parser.add_argument("--save_model_path", type=str, default="./models", help="")
    parser.add_argument("--save_check_freq", type=int, default=5, help="")
    parser.add_argument("--save_check_path", type=str, default="./checkpoints", help="")
    parser.add_argument("--resume", type=bool, default=False, help="")
    
    # parser.add_argument("-amp", type=bool, default=False, help="")
   
    return parser

class Trainer:
    def __init__(self, 
                 model,
                 train_dataloader,
                 val_dataloader,
                 optimizer, 
                 loss_function) -> None:
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __encypt_model(self):
        feature, _ =  next(iter(self.train_dataloader))
        dummy_input = torch.empty(*feature.shape)

        self.model = crypten.nn.from_pytorch(self.model, dummy_input)
        self.model.encrypt()
    
    def train_one_epoch(self):
        self.model.train()
        
        train_loss = AverageMeter() 
        batch_time = AverageMeter()
        
        start = time.time()
        
        for batch_size, (X,y) in tqdm(enumerate(self.train_dataloader)):
            X = X.to(self.device)
            y = y.to(self.device)

            
            y_pred = self.model(X)
            loss = self.loss_function(y_pred, y)
            train_loss.update(loss.item())
            
            self.model.zero_grad()
            loss.backward()
            self.model.update_parameters(self.op)
            
            # train_accuracy += accuracy(y_pred, y)
            
        batch_time.update(time.time() - start)
        # train_accuracy /= len(self.train_dataloader)
        print(f"Train Batch Elapsed: {batch_time.val: .4f}(s) | Train Loss: {train_loss.avg : .4f}\n")
    
    def val_one_epoch(self):
        self.model.eval()
        
        val_loss = AverageMeter()
        batch_time = AverageMeter()
        
        start = time.time()
        with torch.no_grad():
            for batch_size, (X,y) in tqdm(enumerate(self.train_dataloader)):
                X = X.to(self.device)
                y = y.to(self.device)
                
                y_pred = self.model(X)
                loss = self.loss_function(y_pred, y)
                
                val_loss.update(loss.item())
                batch_time.update(time.time() - start)
    
        print(f"Validation Batch Elapsed: {batch_time.val: .4f}(s) | Validation Loss: {val_loss.avg : .4f}\n")
    
    def run(self, args):
        
        #TODO: Check to see args is training from checkpoint
        start_epoch = 0
        
        # if args.resume:
        #     start_epoch = load_checkpoint(self.model, self.optimizer, None, )
        
        for epoch in range(start_epoch, args.epoches):
            print(f"Epoch {epoch + 1}\n")
            self.train_one_epoch()
            self.val_one_epoch()

            if epoch % args.save_check_freq == 0:
                save_checkpoint(self.model, self.optimizer, None, epoch, args.save_check_path)
            
        save_checkpoint(self.model, 
                        self.optimizer, 
                        None, epoch, 
                        args.save_model_path)

                               
def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    mnist_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_train_data = datasets.MNIST(root="./mnist_data",
                                      train=True, 
                                      download=True,
                                      transform=mnist_transforms)
    
    mnist_val_data = datasets.MNIST(root="./mnist_data", 
                                    train=False, 
                                    download= True, 
                                    transform=mnist_transforms)
    
    train_dataloader = torch.utils.data.DataLoader(mnist_train_data, 
                                                   batch_size=args.batch_size, 
                                                   num_workers=args.num_workers, 
                                                   shuffle=True,
                                                   pin_memory=True)
    
    val_dataloader = torch.utils.data.DataLoader(mnist_val_data, 
                                                 batch_size=args.batch_size, 
                                                 num_workers=args.num_workers, 
                                                 shuffle=False,
                                                 pin_memory=True)
    
    model = Model(args.arch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.CrossEntropyLoss()
    
    trainer = Trainer(model, 
                      train_dataloader, 
                      val_dataloader, 
                      optimizer, 
                      loss_function)
    
    
    trainer.run(args)
    
if __name__ == "__main__":
    main()