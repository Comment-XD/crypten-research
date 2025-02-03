import argparse
import time

import crypten
import crypten.communicator
import crypten.mpc as mpc

import torch
import torch.nn as nn

from src.model import models
from utils.data_utils import *
from utils.utils import *

import warnings


def get_parser():
    parser = argparse.ArgumentParser("ReLU MPC Inference")

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
                        help="Choices for which model you want for inferencing",
                        default="lenet5")
    
    parser.add_argument("--batch_size",
                        type=int,
                        help="Size of Batches for Training and Validation",
                        default=64)
    
    parser.add_argument("--workers",
                        type=int,
                        help="Number of workers for Data Loading",
                        default=1)
    
    
    args = parser.parse_args()

    return args

@mpc.run_multiprocess(world_size=2)
def relu_mpc_inference():
    crypten.init()

    args = get_parser()
    model = models[args.model]

    _, val_dataloader = load_dataloader(args)
    state = torch.load("./models/LeNet_99.pt", map_location=torch.device("cpu"), weights_only=True)

    accuracy_meter = AverageMeter()

    model.load_state_dict(state["model"])
    encrypted_model = crypten.nn.from_pytorch(model, 
                                              dummy_input=torch.empty(64, 1, 32, 32))
    encrypted_model.encrypt(src=0)
    replace_relu(encrypted_model)

    # encrypted_model.encrypt(src=0)
    encrypted_model.eval()

    start = time.time()
    
    for batch_size, (X,y) in enumerate(val_dataloader):

        X_enc = crypten.cryptensor(X, src=1)
        y_pred = encrypted_model(X_enc).get_plain_text()
        acc = compute_accuracy(y_pred, y)

        accuracy_meter.update(acc)
    
    print(f"Total Time Spent: {time.time() - start: .4f}")
    print(f"Average Inference Accuracy: {accuracy_meter.avg : .4f}")


def replace_relu(model):
    for child_name, child in model.named_children():
        if isinstance(child, crypten.nn.ReLU):
            setattr(model, child_name, CustomReLU(0.1))
        else:
            replace_relu(child)


def crypten_model_eval():
    crypten.init()

    args = get_parser()
    model = models[args.model]

    _, val_dataloader = load_dataloader(args)
    state = torch.load("./models/LeNet_99.pt", map_location=torch.device("cpu"), weights_only=True)

    accuracy_meter = AverageMeter()

    model.load_state_dict(state["model"])
    encrypted_model = crypten.nn.from_pytorch(model, 
                                              dummy_input=torch.empty(64, 1, 32, 32))
    encrypted_model.encrypt(src=0)

    replace_relu(encrypted_model)
    for name, layer in encrypted_model.named_modules():
        print(name, layer)

if __name__ == "__main__":
    relu_mpc_inference()