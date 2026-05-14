"""
Modelo: ATT_CRNN_ConvNeXt_Tiny_AudioClassifier

Arquitetura:
Attention-CRNN (ConvNeXt-Tiny + BiLSTM + Multi-Head Self-Attention)

Descrição:
Utiliza Multi-Head Self-Attention para realizar o pooling temporal, permitindo que o modelo
aprenda quais segmentos do áudio são mais importantes para a classificação.

Entrada:
[Batch, 1, Freq, Tempo]

Saída:
[Batch, num_classes]
"""

import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path

class ConvNeXtExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ConvNeXtExtractor, self).__init__()
        if hasattr(models, 'ConvNeXt_Tiny_Weights'):
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            convnext = models.convnext_tiny(weights=weights)
        else:
            convnext = models.convnext_tiny(pretrained=pretrained)
        
        self.features = convnext.features
        
        # Adaptar o Stem para 1 canal
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
    def __init__(self, num_classes, rnn_hidden_size=256, rnn_layers=2, nhead=8):
        super(Model, self).__init__()
        
        self.backbone = ConvNeXtExtractor(pretrained=True)
        
        # ConvNeXt-Tiny tem 768 canais
        self.input_size_rnn = 768 
        
        self.rnn = nn.LSTM(
            input_size=self.input_size_rnn,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Dimensão da saída da BiLSTM
        d_model = rnn_hidden_size * 2
        
        # Multi-Head Attention para pooling temporal
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        
        # Vetor de consulta (Query) aprendível para o pooling por atenção
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x: [Batch, 1, Freq, Tempo]
        features = self.backbone(x)           # [Batch, 768, H_feat, W_feat]
        
        # Global Frequency Pooling
        features = torch.mean(features, dim=2) # [Batch, 768, W_feat]
        features = features.permute(0, 2, 1)   # [Batch, Seq_Len, Features]
        
        # RNN
        self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(features)        # [Batch, Seq_Len, d_model]
        
        # Multi-Head Attention Pooling
        # Query é o parâmetro aprendível, Key e Value são as saídas da RNN
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1) # [Batch, 1, d_model]
        
        attn_out, _ = self.attention(query, rnn_out, rnn_out) # [Batch, 1, d_model]
        attn_out = attn_out.squeeze(1) # [Batch, d_model]
        
        out = self.dropout(attn_out)
        logits = self.fc(out)
        return logits


if __name__ == "__main__":
    # Teste rápido de formato
    from pathlib import Path
    dataset_path = Path(__file__).parent.parent.parent / "dataset"
    num_classes = len([d for d in dataset_path.iterdir() if d.is_dir()]) if dataset_path.exists() else 11
    
    model = Model(num_classes=num_classes)
    
    test_input = torch.randn(2, 1, 1025, 427)
    output = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total de parâmetros: {total_params:,}")
