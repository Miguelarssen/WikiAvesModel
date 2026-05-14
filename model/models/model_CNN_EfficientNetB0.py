"""
Modelo: CNN_EfficientNetB0_AudioClassifier

Arquitetura:
EfficientNet-B0 (Convolutional Neural Network)

Backbone:
EfficientNet-B0 (pré-treinada, adaptada para 1 canal)

Descrição:
Usa escalonamento composto para melhor eficiência e acurácia que ResNet18.
Adaptada para receber espectrogramas de 1 canal.

Entrada:
[Batch, 1, Freq, Tempo]

Saída:
[Batch, num_classes]
"""

import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNetExtractor, self).__init__()
        # Usamos as Weights recomendadas se disponíveis (torchvision 0.13+)
        if hasattr(models, 'EfficientNet_B0_Weights'):
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            efficientnet = models.efficientnet_b0(weights=weights)
        else:
            efficientnet = models.efficientnet_b0(pretrained=pretrained)
        
        # O backbone do EfficientNet no torchvision fica em .features
        self.features = efficientnet.features
        
        # Adaptar o primeiro layer (conv0) para 1 canal
        # O layer original é Conv2dNormActivation(3, 32, ...)
        original_conv = self.features[0][0]
        self.features[0][0] = nn.Conv2d(
            1, 
            original_conv.out_channels, 
            kernel_size=original_conv.kernel_size, 
            stride=original_conv.stride, 
            padding=original_conv.padding, 
            bias=False
        )
        
        # Copiar e somar pesos para manter o aprendizado
        with torch.no_grad():
            self.features[0][0].weight.copy_(original_conv.weight.sum(dim=1, keepdim=True))

    def forward(self, x):
        return self.features(x)


class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        
        self.feature_extractor = EfficientNetExtractor(pretrained=True)
        
        # Pooling global
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Dropout para regularização (EfficientNet costuma usar 0.2 no B0)
        self.dropout = nn.Dropout(p=0.2, inplace=True)
        
        # Classificador (EfficientNet-B0 termina com 1280 canais)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        # x: [Batch, 1, Freq, Tempo]
        features = self.feature_extractor(x)  # [Batch, 1280, H_feat, W_feat]
        
        features = self.global_pool(features) # [Batch, 1280, 1, 1]
        features = torch.flatten(features, 1) # [Batch, 1280]
        
        features = self.dropout(features)
        logits = self.fc(features)
        
        return logits


if __name__ == "__main__":
    from pathlib import Path
    dataset_path = Path(__file__).parent.parent.parent / "dataset"
    num_classes = len([d for d in dataset_path.iterdir() if d.is_dir()]) if dataset_path.exists() else 11
    
    model = Model(num_classes=num_classes)
    
    # Input exemplo: [Batch, Channel, Freq, Time]
    test_input = torch.randn(2, 1, 1025, 427)
    output = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")  # Deve ser [2, 11]
    
    # Verificar se o número de parâmetros é razoável (~4M para B0)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total de parâmetros: {total_params:,}")
