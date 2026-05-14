"""
Modelo: CNN_ResNet18_AudioClassifier

Arquitetura:
CNN (Convolutional Neural Network)

Backbone:
ResNet18 (pré-treinada, adaptada para 1 canal)

Temporal:
Não possui (uso de pooling global)

Descrição:
CNN extrai padrões do espectrograma → pooling global agrega no tempo → FC classifica

Entrada:
[Batch, 1, Freq, Tempo]

Saída:
[Batch, num_classes]
"""


import torch
import torch.nn as nn
import torchvision.models as models

class ResNetExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        
        # Adaptar para 1 canal
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data = resnet.conv1.weight.data.sum(dim=1, keepdim=True)
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        
        self.feature_extractor = ResNetExtractor(pretrained=True)

        # Pooling global (reduz H e W para 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classificador
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)  # [B, 512, H, W]

        features = self.global_pool(features) # [B, 512, 1, 1]
        features = features.view(features.size(0), -1) # [B, 512]

        logits = self.fc(features)

        return logits


if __name__ == "__main__":
    from pathlib import Path
    dataset_path = Path(__file__).parent.parent.parent / "dataset"
    num_classes = len([d for d in dataset_path.iterdir() if d.is_dir()]) if dataset_path.exists() else 11
    
    model = Model(num_classes=num_classes)
    test_input = torch.randn(2, 1, 1025, 427)
    output = model(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")  # [2, 11]