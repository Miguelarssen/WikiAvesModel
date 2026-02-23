from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import torch
import torch
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split


class ResNetExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetExtractor, self).__init__()
        # Usando ResNet18 como extrator de características
        # O espectrograma tem 1 canal (Grayscale), mas a ResNet espera 3 (RGB).
        # Vamos adaptar a primeira camada.
        resnet = models.resnet18(pretrained=pretrained)
        
        # Adaptar entrada para 1 canal
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data = resnet.conv1.weight.data.sum(dim=1, keepdim=True)
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Removemos a camada fc e avgpool original para manter a estrutura espacial/temporal
        # Saída da ResNet18 para entrada 1025x427:
        # Original (224x224) -> layer4 sai 7x7
        # Aqui (1025x427) -> layer4 sai aprox (33 x 14)
        
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

class CRNNResNet(nn.Module):
    def __init__(self, num_classes, rnn_hidden_size=256, rnn_layers=2):
        super(CRNNResNet, self).__init__()
        self.feature_extractor = ResNetExtractor(pretrained=True)
        
        # ResNet18 layer4 tem 512 canais de saída
        self.input_size_rnn = 512 
        
        # A CRNN geralmente trata a largura do mapa de características como o eixo temporal
        # Após a ResNet, teremos [Batch, 512, H_feat, W_feat]
        # Vamos fazer um Adaptive Pooling para fixar a altura em 1, mantendo a largura temporal
        self.avgpool = nn.AdaptiveAvgPool2d((1, None))
        
        self.rnn = nn.LSTM(
            input_size=self.input_size_rnn,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.fc = nn.Linear(rnn_hidden_size * 2, num_classes)
        
    def forward(self, x):
        # x: [Batch, 1, 1025, 427]
        features = self.feature_extractor(x) # [Batch, 512, H_feat, W_feat]
        
        # Reduzir altura para 1: [Batch, 512, 1, W_feat]
        features = self.avgpool(features)
        
        # Remover dimensão de altura e permutar para [Batch, W_feat, 512] para a LSTM
        features = features.squeeze(2) 
        features = features.permute(0, 2, 1) # [Batch, Seq_Len, Features]
        
        # RNN
        self.rnn.flatten_parameters()
        out, _ = self.rnn(features) # [Batch, Seq_Len, hidden*2]
        
        # Tomar a última saída temporal ou fazer pooling temporal
        # Para classificação global de áudio, pooling temporal costuma ser melhor
        out = torch.mean(out, dim=1) # [Batch, hidden*2]
        
        logits = self.fc(out)
        return logits

if __name__ == "__main__":
    # Teste rápido de formato
    model = CRNNResNet(num_classes=15)
    test_input = torch.randn(2, 1, 1025, 427)
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}") # Deve ser [2, 15]
