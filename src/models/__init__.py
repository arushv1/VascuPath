import torch
from models.stain_segmentor import FoundationClassifier
from models.vessel_detector import VesselDetector


def load_classifier(checkpoint_path=None, device=None):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model_type = checkpoint.get("model_type", "resnet")
    
    if model_type == "foundation":
        model = FoundationClassifier(num_classes=num_classes).to(device)
    else:
        model = VesselDetector(num_classes=num_classes, pretrained=False).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, class_names, metadata