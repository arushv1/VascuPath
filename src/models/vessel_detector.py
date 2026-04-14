import torch
import torch.nn as nn
from torchvision import models
import random
from dinov2.models.vision_transformer import vit_large
from config import FOUNDATION_CHECKPOINT


class VesselDetector(nn.Module):
    """ResNet18 classifier with custom head"""
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        
        self.resnet = models.resnet18(pretrained=pretrained)
        
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)
    
class FoundationVesselDetector(nn.Module):
    """
    Professor Li's DINOv2 ViT-Large backbone + linear classifier.
    """

    def __init__(self, num_classes=3, freeze_backbone=True):
        super().__init__()

        # Foundation model backbone
        self.backbone = vit_large(
            patch_size=16,
            img_size=224,
            init_values=1.0,
            ffn_layer="mlp",
            block_chunks=4,
            num_register_tokens=0,
        )

        # Pretrained neuropath weights
        checkpoint_path = str(FOUNDATION_CHECKPOINT)
        state_dict = torch.load(checkpoint_path, map_location="cpu")["teacher"]
        state_dict = {k.replace("backbone.", ""): v 
                      for k, v in state_dict.items() if "backbone" in k}
        self.backbone.load_state_dict(state_dict)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ViT-Large embed_dim = 1024
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        # backbone outputs class token features
        features = self.backbone(x)  # (B, 1024)
        return self.classifier(features)