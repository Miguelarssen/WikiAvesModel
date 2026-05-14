"""
Modelo: CRNN_EfficientNetB0_BiLSTM_AudioClassifier

Arquitetura:
CRNN (EfficientNet-B0 + BiLSTM)

Backbone:
EfficientNet-B0 (pré-treinada, adaptada para 1 canal)

Temporal:
BiLSTM (2 camadas)

Descrição:
CNN extrai padrões do espectrograma -> LSTM modela o tempo -> FC classifica.
Usa EfficientNet-B0 para extração de características superior.

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
        if hasattr(models, 'EfficientNet_B0_Weights'):
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            efficientnet = models.efficientnet_b0(weights=weights)
        else:
            efficientnet = models.efficientnet_b0(pretrained=pretrained)
        
        self.features = efficientnet.features
        
        # Adaptar para 1 canal
        original_conv = self.features[0][0]
        self.features[0][0] = nn.Conv2d(
            1, 
            original_conv.out_channels, 
            kernel_size=original_conv.kernel_size, 
            stride=original_conv.stride, 
            padding=original_conv.padding, 
            bias=False
        )
        
        with torch.no_grad():
            self.features[0][0].weight.copy_(original_conv.weight.sum(dim=1, keepdim=True))

    def forward(self, x):
        return self.features(x)


class Model(nn.Module):
    def __init__(self, num_classes, rnn_hidden_size=256, rnn_layers=2):
        super(Model, self).__init__()
        
        self.feature_extractor = EfficientNetExtractor(pretrained=True)
        
        # EfficientNet-B0 extrai 1280 canais
        self.input_size_rnn = 1280 
        
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
        # x: [Batch, 1, Freq, Tempo]
        features = self.feature_extractor(x) # [Batch, 1280, H_feat, W_feat]
        
        # Pooling Global de Frequência (reduz H para 1, mantém W para a LSTM)
        # Em áudio, a dimensão 2 costuma ser Frequência e a 3 o Tempo
        features = torch.mean(features, dim=2) # [Batch, 1280, W_feat]
        features = features.permute(0, 2, 1)   # [Batch, Seq_Len, Features]
        
        # RNN
        self.rnn.flatten_parameters()
        out, _ = self.rnn(features) # [Batch, Seq_Len, hidden*2]
        
        # Pooling Temporal (média das saídas da LSTM ao longo do tempo)
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
    print(f"Output shape: {output.shape}")  # [2, 11]
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total de parâmetros: {total_params:,}")
