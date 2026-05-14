"""
Modelo: CRNN_ConvNeXt_Tiny_BiLSTM_AudioClassifier

Arquitetura:
CRNN (ConvNeXt-Tiny + BiLSTM)

Backbone:
ConvNeXt-Tiny (pré-treinada, adaptada para 1 canal)

Temporal:
BiLSTM (2 camadas)

Descrição:
Combina o poder de extração do ConvNeXt com a capacidade sequencial da BiLSTM.
Adaptada para receber espectrogramas de 1 canal.

Entrada:
[Batch, 1, Freq, Tempo]

Saída:
[Batch, num_classes]
"""

import torch
import torch.nn as nn
import torchvision.models as models

class ConvNeXtExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ConvNeXtExtractor, self).__init__()
        if hasattr(models, 'ConvNeXt_Tiny_Weights'):
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            convnext = models.convnext_tiny(weights=weights)
        else:
            convnext = models.convnext_tiny(pretrained=pretrained)
        
        self.features = convnext.features
        
        # Adaptar o Stem para 1 canal (usando média para estabilidade)
        original_stem_conv = self.features[0][0]
        self.features[0][0] = nn.Conv2d(
            1, 
            original_stem_conv.out_channels, 
            kernel_size=original_stem_conv.kernel_size, 
            stride=original_stem_conv.stride, 
            padding=original_stem_conv.padding, 
            bias=original_stem_conv.bias is not None
        )
        
        if pretrained:
            with torch.no_grad():
                self.features[0][0].weight.copy_(original_stem_conv.weight.mean(dim=1, keepdim=True))
                if original_stem_conv.bias is not None:
                    self.features[0][0].bias.copy_(original_stem_conv.bias)

    def forward(self, x):
        return self.features(x)


class Model(nn.Module):
    def __init__(self, num_classes, rnn_hidden_size=256, rnn_layers=2):
        super(Model, self).__init__()
        
        self.backbone = ConvNeXtExtractor(pretrained=True)
        
        # ConvNeXt-Tiny tem 768 canais na saída das features
        self.input_size_rnn = 768 
        
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
        features = self.backbone(x) # [Batch, 768, H_feat, W_feat]
        
        # Pooling Global de Frequência (mantém a dimensão temporal para a LSTM)
        features = torch.mean(features, dim=2) # [Batch, 768, W_feat]
        features = features.permute(0, 2, 1)   # [Batch, Seq_Len, Features]
        
        # RNN
        self.rnn.flatten_parameters()
        out, _ = self.rnn(features) # [Batch, Seq_Len, hidden*2]
        
        # Pooling Temporal
        out = torch.mean(out, dim=1) # [Batch, hidden*2]
        
        out = self.dropout(out)
        logits = self.fc(out)
        return logits


if __name__ == "__main__":
    # Teste rápido de formato
    from pathlib import Path
    
    # Tenta detectar o número de classes real do dataset
    dataset_path = Path(__file__).parent.parent.parent / "dataset"
    if dataset_path.exists():
        num_classes = len([d for d in dataset_path.iterdir() if d.is_dir()])
        print(f"Detectadas {num_classes} classes no dataset.")
    else:
        num_classes = 11
        print(f"Dataset não encontrado em {dataset_path}. Usando fallback de {num_classes} classes.")

    model = Model(num_classes=num_classes)
    
    test_input = torch.randn(2, 1, 1025, 427)
    output = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total de parâmetros: {total_params:,}")
