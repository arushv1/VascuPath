import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import sys
from pathlib import Path
from collections import Counter
from dinov2.models.vision_transformer import vit_large
from config import FOUNDATION_CHECKPOINT


class VascuPathMultiHead(nn.Module):
    def __init__(self, embed_dim=1024, num_stain_classes=3, freeze_backbone=True):
        """
        Args;
            foundation_model: Pre-loaded DIONv2 NeuroPath model.
            embed_dim: 1024 for DINOv2 ViT-L.
            num_stain_classes: 3 (white, grey, background).
        """

        super(VascuPathMultiHead, self).__init__()

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


        # Head 1: Stain Segmentor
        self.stain_head = nn.Linear(embed_dim, num_stain_classes)

        # Head 2: Conditional Vessel Classifier
        self.vessel_head = nn.Linear(embed_dim + num_stain_classes, 1)
    
    def forward(self, x):
        f_cls = self.backbone(x) # (B, 1024)
        
        # Head 1: Stain classification
        z_stain = self.stain_head(f_cls)

        # Conditioning: Get softmax probs of the stain
        p_stain = F.softmax(z_stain, dim=1)

        # Concatenate CLS token with stain prediction
        f_combined = torch.cat([f_cls, p_stain], dim=1)

        # Head 2: Vessel Classification
        z_vessel = self.vessel_head(f_combined)

        return z_stain, z_vessel
    

