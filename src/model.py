import torch.nn as nn
from torchvision import models

def build_model(num_classes: int = 13, pretrained:bool = True):
    """
    Builds a ResNet50 model for multi-class classifciation

    Args: 
    num_classes (int): number of input classes
    pretrained (bool): Uses Imagenet pretrained weights?

    Returns:
    torch.nn.Module
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights)

    #freeze backbone
    for params in model.parameters():
        params.requires_grad = False

    #unfreeze final layer 
    for params in model.layer4.parameters():
        params.requires_grad = True

    model.fc = nn.Sequential(
        nn.Linear(in_features=model.fc.in_features, out_features=128),
        nn.ReLU(),
        nn.Dropout(.4),
        nn.Linear(128,num_classes)
    )
    #unfreeze classifier head
    for params in model.fc.parameters():
        params.requires_grad = True

    return model