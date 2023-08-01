import torchvision.models as models
import torch.nn as nn

def build_model(pretrained=True, fine_tune=True, num_classes=1):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    elif not pretrained:
        print('[INFO]: Not loading pre-trained weights')
    model = models.resnet34(pretrained=pretrained)

    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
            
    # change the final classification head, it is trainable
    model.fc = nn.Linear(512, num_classes)
    return model

