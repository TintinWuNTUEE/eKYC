import os
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s,EfficientNet_V2_S_Weights
class binaryClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1000,1)
    def forward(self,x):
        x = self.fc1(x)
        return x
class myEfficientNet(nn.Module):
    def __init__(self) :
        super().__init__()
        self.efficientnet = efficientnet_v2_s(EfficientNet_V2_S_Weights.DEFAULT)
        self.Classifier = binaryClassifier()
    def forward(self, x):
        x = self.efficientnet(x)
        x = self.Classifier(x)
        return x
def get_models():
    efficientNet = myEfficientNet()
    return efficientNet
