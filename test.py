import torch

from src.model import models
from utils.data_utils import load_dataloader

import torchvision.datasets as datasets




if __name__ == "__main__":
    model_name = "lenet5"
    model = models[model_name]
    
    state = torch.load("./models/LeNet_99.pt", weights_only=True)
    
    model.load_state_dict(state["model"])
    model.eval()
    
    feature, label = load_dataloader(args)