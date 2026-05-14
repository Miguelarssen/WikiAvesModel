
"""
Modelo: CRNN_ResNet18_BiLSTM_AudioClassifier

Arquitetura:
CRNN (CNN + RNN)

Backbone:
ResNet18 (pré-treinada, adaptada para 1 canal)

Temporal:
BiLSTM (2 camadas)

Descrição:
CNN extrai padrões do espectrograma → LSTM modela o tempo → FC classifica

Entrada:
[Batch, 1, Freq, Tempo]

Saída:
[Batch, num_classes]
"""


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, random_split, Dataset
from pathlib import Path
import numpy as np


# Usamos a ResNet como um extrator de características, para destacar as características
# do Espectrograma

class ResNetExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        
        # Resnet18 funciona com rgb, ou seja, precisamos adaptar para 1 plano, com 1 dimensão só
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Usamos a média para manter a escala das ativações original da ResNet
        with torch.no_grad():
            self.conv1.weight.copy_(resnet.conv1.weight.data.mean(dim=1, keepdim=True))
        
        #batch normalization 1
        self.bn1 = resnet.bn1

        #função de ativação ReLU
        self.relu = resnet.relu

        #subamostragem com maxpooling
        self.maxpool = resnet.maxpool

        #definições das camadas padrão da classe da ResNet
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
    
# x: [Batch, 1, 1025, 427]. formato do esptrograma agrupado nos batches

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
    def __init__(self, num_classes, rnn_hidden_size=256, rnn_layers=2):
        super(Model, self).__init__()
        self.feature_extractor = ResNetExtractor(pretrained=True)
        
        # ResNet18 layer4 tem 512 canais de saída
        self.input_size_rnn = 512 
        
        # Removido AdaptiveAvgPool2d problemático com None
        # Vamos fazer o pooling manualmente no forward para garantir compatibilidade
        
        self.rnn = nn.LSTM(
            input_size=self.input_size_rnn,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(rnn_hidden_size * 2, num_classes)
        
    def forward(self, x):

        features = self.feature_extractor(x) # [Batch, 512, H_feat, W_feat]
        
        # Reduzir a dimensão de "altura" (H) para 1 tirando a média (Pooling Global de Frequência)
        # Mantém a dimensão de "largura" (W) como o eixo temporal para a LSTM
        features = torch.mean(features, dim=2) # [Batch, 512, W_feat]
        features = features.permute(0, 2, 1)   # [Batch, Seq_Len, Features]
        
        # RNN
        self.rnn.flatten_parameters()
        out, _ = self.rnn(features) # [Batch, Seq_Len, hidden*2]
        
        # Tomar a última saída temporal ou fazer pooling temporal
        # Para classificação global de áudio, pooling temporal costuma ser melhor
        out = torch.mean(out, dim=1) # [Batch, hidden*2]
        
        out = self.dropout(out)
        logits = self.fc(out)
        return logits

if __name__ == "__main__":
    from pathlib import Path
    dataset_path = Path(__file__).parent.parent.parent / "dataset"
    num_classes = len([d for d in dataset_path.iterdir() if d.is_dir()]) if dataset_path.exists() else 11
    
    model = Model(num_classes=num_classes)
    test_input = torch.randn(2, 1, 1025, 427)
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}") # Deve ser [2, 15]
