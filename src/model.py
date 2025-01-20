import torch
import torch.nn as nn

from torchvision.models.alexnet import *
from torchvision.models.resnet import *
from torchvision.models.vgg import *

device = "cuda" if torch.cuda.is_available() else "cpu"

class LeNet(nn.Module):
    def __init__(self, in_chans=1, out_chans=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_chans,
                      out_channels=6,
                      kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,
                         stride=2),
            
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,
                         stride=2),
            nn.Conv2d(in_channels=16,
                      out_channels=120,
                      kernel_size=5))
        
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, out_chans),
            nn.Softmax(dim=-1))
    
    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.shape[0], -1)

        out = self.classifier(x)

        return out

def lenet5():
    return LeNet(in_chans=1, out_chans=10)

models = {"lenet5": lenet5(),
          "alexnet": alexnet(),
          "resnet18": resnet18(),
          "resnet50": resnet50(),
          "vgg16": vgg16()}
    