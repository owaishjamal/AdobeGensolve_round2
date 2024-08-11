import torch
from torchvision import models
import torch.nn as nn

def initialize_model(weights_path, num_classes=5):
    # Set up the model (matching architecture used during training)
    model = models.resnet18(pretrained=False)
    feature_count = model.fc.in_features
    model.fc = nn.Linear(feature_count, num_classes)

    # Load the trained weights
    model.load_state_dict(torch.load(weights_path))
    model.eval()  # Switch the model to evaluation mode
    return model

if __name__ == "__main__":
    weights_path = '/content/drive/MyDrive/curve_project/models/shape_recognition_model.pth'
    model = initialize_model(weights_path)
    print(f"Model loaded from {weights_path}")
