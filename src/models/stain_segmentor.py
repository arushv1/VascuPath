import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import sys
from pathlib import Path
from collections import Counter
from dinov2.models.vision_transformer import vit_large


class FoundationClassifier(nn.Module):
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
        checkpoint_path = "src/models/teacher_checkpoint.pth"
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
        with torch.no_grad() if not self.training else torch.enable_grad():
            features = self.backbone(x)  # (B, 1024)
        return self.classifier(features)
    

