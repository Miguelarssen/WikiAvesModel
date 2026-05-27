"""
Modelo: CRNN_ConvNeXt_Tiny_DeepConv_BiLSTM

Arquitetura:
CRNN com Pescoço Convolucional Profundo (ConvNeXt-Tiny + DeepConv Neck + BiLSTM)

Backbone:
    ConvNeXt-Tiny (pré-treinada, adaptada para 1 canal)

Deep Conv Neck:
    3 blocos conv com skip connections (tipo ResNet) aplicados APÓS o backbone.
    Cada bloco: Conv2d → BatchNorm → GELU → Conv2d → BatchNorm + residual
    Canais: 768 → 512 → 384 → 256

    Propósito experimental: permitir que o modelo aprenda padrões de textura
    acústica mais refinados antes de passar para a camada recorrente.

Temporal:
    BiLSTM (2 camadas)

Diferença em relação ao modelo base (CRNN_ConvNeXt_Tiny_BiLSTM):
    + 3 blocos convolucionais extras com skip connections após o backbone
    + Canal de entrada da LSTM reduzido de 768 para 256 (menor, mais eficiente)
    + Permite comparar se extração extra de features melhora a classificação

Entrada:
    [Batch, 1, Freq, Tempo]

Saída:
    [Batch, num_classes]
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ── Backbone: mesmo ConvNeXt-Tiny do modelo base ──────────────────────────────

class ConvNeXtExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if hasattr(models, 'ConvNeXt_Tiny_Weights'):
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            convnext = models.convnext_tiny(weights=weights)
        else:
            convnext = models.convnext_tiny(pretrained=pretrained)

        self.features = convnext.features

        # Adaptar o Stem para 1 canal (mesma técnica do modelo base)
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
                self.features[0][0].weight.copy_(
                    original_stem_conv.weight.mean(dim=1, keepdim=True)
                )
                if original_stem_conv.bias is not None:
                    self.features[0][0].bias.copy_(original_stem_conv.bias)

    def forward(self, x):
        return self.features(x)


# ── Bloco residual convolucional 2D ──────────────────────────────────────────

class ConvResBlock(nn.Module):
    """
    Bloco conv com skip connection.
    Se in_channels != out_channels, usa uma projeção 1x1 para o residual.

    Estrutura:
        Conv2d(in, out, 3x3) → BN → GELU → Conv2d(out, out, 3x3) → BN
        + skip connection (projeção se necessário)
        → GELU
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.act   = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Projeção residual (apenas se os canais diferem)
        if in_channels != out_channels:
            self.residual_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x):
        residual = self.residual_proj(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


# ── Modelo principal ──────────────────────────────────────────────────────────

class Model(nn.Module):
    def __init__(self, num_classes, rnn_hidden_size=256, rnn_layers=2):
        super().__init__()

        # Backbone ConvNeXt-Tiny (saída: 768 canais)
        self.backbone = ConvNeXtExtractor(pretrained=True)

        # ── Deep Conv Neck ────────────────────────────────────────────────────
        # 3 blocos residuais que refinam as features espacialmente
        # antes de entrar na BiLSTM.
        # Redução progressiva de canais: 768 → 512 → 384 → 256
        self.neck = nn.Sequential(
            ConvResBlock(768, 512),   # bloco 1: comprime de 768 para 512
            ConvResBlock(512, 384),   # bloco 2: comprime de 512 para 384
            ConvResBlock(384, 256),   # bloco 3: comprime de 384 para 256
        )

        # Entrada da LSTM = 256 (saída do neck)
        self.input_size_rnn = 256

        # ── BiLSTM ────────────────────────────────────────────────────────────
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
        features = self.backbone(x)          # [Batch, 768, H_feat, W_feat]

        # Aplicar neck convolucional
        features = self.neck(features)        # [Batch, 256, H_feat, W_feat]

        # Pooling Global de Frequência → mantém dimensão temporal para a LSTM
        features = torch.mean(features, dim=2)  # [Batch, 256, W_feat]
        features = features.permute(0, 2, 1)    # [Batch, Seq_Len, 256]

        # BiLSTM
        self.rnn.flatten_parameters()
        out, _ = self.rnn(features)          # [Batch, Seq_Len, hidden*2]

        # Pooling Temporal
        out = torch.mean(out, dim=1)         # [Batch, hidden*2]

        out = self.dropout(out)
        logits = self.fc(out)                # [Batch, num_classes]
        return logits


if __name__ == "__main__":
    from pathlib import Path

    dataset_path = Path(__file__).parent.parent.parent / "dataset"
    if dataset_path.exists():
        num_classes = len([d for d in dataset_path.iterdir() if d.is_dir()])
        print(f"Detectadas {num_classes} classes no dataset.")
    else:
        num_classes = 11
        print(f"Dataset não encontrado. Usando fallback: {num_classes} classes.")

    model = Model(num_classes=num_classes)

    test_input = torch.randn(2, 1, 1025, 427)
    output = model(test_input)

    print(f"Input shape:  {test_input.shape}")
    print(f"Output shape: {output.shape}")

    total_params    = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    neck_params     = sum(p.numel() for p in model.neck.parameters())
    rnn_params      = sum(p.numel() for p in model.rnn.parameters())
    fc_params       = sum(p.numel() for p in model.fc.parameters())

    print(f"\nParâmetros totais:      {total_params:>12,}")
    print(f"  └─ Backbone:          {backbone_params:>12,}")
    print(f"  └─ Deep Conv Neck:    {neck_params:>12,}")
    print(f"  └─ BiLSTM:            {rnn_params:>12,}")
    print(f"  └─ FC:                {fc_params:>12,}")
