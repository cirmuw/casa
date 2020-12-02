import torch.nn as nn
import torch
import math
#from torchvision.ops import nms
import numpy as np
from nms import nms
import torchvision.models as tvmodels

class NaiveRegressionNet(nn.Module):

    def __init__(self):
        super(NaiveRegressionNet, self).__init__()
        self.resnet = tvmodels.resnet50(pretrained=True)
        self.resnet.fc = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, 2))

    def forward(self, x):
        regression = self.resnet(x)
        return regression
