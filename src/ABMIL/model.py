import os
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import Dataset


class MILSlideDataset(Dataset):
    """
    Dataset where each item is a full slide (bag of tiles)
    
    Parameters:
    feature_dir: Path
    - Directory containing per-slide .pt files.
    slide_labels: dict
    - Mapping of svs_stem --> (binary_label, case_id)
    """

    def __init__(self, feature_dir, slide_labels):
        self.feature_dir = Path(feature_dir)
        self.slides = []
        self.labels = []
        self.case_ids = []

        for pt_file in sorted(self.feature_dir.glob("*.pt")):
            stem = pt_file.stem
            if stem in slide_labels:
                label, case_id = slide_labels[stem]
                self.slides.append(pt_file)
                self.labels.append(label)
                self.case_ids.append(case_id)

    def __len__(self):
        return(len(self.slides))
    
    def __getitem__(self, idx):
        data = torch.load(self.slides[idx], map_location="cpu")
        features = data["features"] # (N_tiles, 1024)
        label = self.labels[idx]
        return features, label, idx
    
    def get_case_ids(self):
        return self.case_ids
    
    def get_labels(self):
        return self.labels
    
    def summary(self):
        counts = Counter(self.labels)
        return (
            f"Slides: {len(self.slides)},"
            f"Class 0: {counts.get(0, 0)}, Class 1: {counts.get(1, 0)},"
            f"Cases: {len(set(self.case_ids))}"
        )
    

class AttentionMIL(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, attention_dim=128, dropout=0.5):
        super().__init__()

        # Feature compressor
        self.compressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Gated attention mechanism
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Sigmoid(),
        )
        self.attention_W = nn.Linear(attention_dim, 1)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):

        # Compresses (N, 1024) --> (N, hidden_dim)
        h = self.compressor(x)

        # Gated attention (N, hidden_him) --> (N, 1)
        a_V = self.attention_V(h)
        a_U = self.attention_U(h)
        a = self.attention_W(a_V * a_U)

        # Softmax over tiles for weights
        a = F.softmax(a, dim=0) # (N, 1)
        a_W = a.squeeze(1) # (N, 1)

        # Weighted average -> slide representation 
        slide_repr = torch.mm(a.T, h) # (1, hidden_dim)

        #Classification
        logits = self.classifier(slide_repr.squeeze(0)) # (2,0)

        return logits, a_W