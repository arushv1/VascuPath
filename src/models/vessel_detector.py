import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from collections import Counter
import random

# MODEL DEFINITION
class HVesselDetector(nn.Module):
    """ResNet18 classifier with custom head"""
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        
        self.resnet = models.resnet18(pretrained=pretrained)
        
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)
    

class EVesselDetector(nn.Module):
    """ResNet18 classifier with custom head"""
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        
        self.resnet = models.resnet18(pretrained=pretrained)
        
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)
