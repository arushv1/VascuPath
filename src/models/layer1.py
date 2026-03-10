import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, randond_split
from torchvision import transforms, datasets
import sys
from pathlib import Path
from collections import Counter

sys.path.append()

current_dir = Path(__file__).parent
dinov2_path = current_dir / "dinov2"

sys.path.append(str(dinov2_path))

class FoundationClassifier(nn.Module):
    """
    Professor Li's DINOv2 ViT-Large backbone + linear classifier.
    
    The backbone is frozen (no gradients) — we only train the
    linear head. This is much faster than training ResNet from
    scratch and leverages the foundation model's learned
    representations of neuropathology tissue.
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

        # Load pretrained neuropath weights
        checkpoint_path = "/projectnb/rise2019/Shuying_AI_path/dinov2/neuropath_v2/eval/training_2499999/teacher_checkpoint.pth"
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