"""
Modelo: CNN_ConvNeXt_Tiny_TCN

Arquitetura:
100% Convolucional (ConvNeXt-Tiny + Temporal Convolutional Network)

Backbone:
    ConvNeXt-Tiny (pré-treinada, adaptada para 1 canal). Extrai features espaciais.

Neck (Temporal Convolutional Network):
    Substitui a antiga BiLSTM.
    Aplica convoluções 1D ao longo da dimensão do tempo com "dilatação" crescente.
    Isso permite que a rede correlacione eventos distantes no tempo (ritmos, cantos longos)
    de forma puramente convolucional (mais rápido e fácil de otimizar que RNNs).

Entrada:
    [Batch, 1, Freq, Tempo]

Saída:
    [Batch, num_classes]
"""

import torch
import torch.nn as nn
import torchvision.models as models

# ── Backbone: ConvNeXt-Tiny adaptado ──────────────────────────────────────────

class ConvNeXtExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
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
                self.features[0][0].weight.copy_(
                    original_stem_conv.weight.mean(dim=1, keepdim=True)
                )
                if original_stem_conv.bias is not None:
                    self.features[0][0].bias.copy_(original_stem_conv.bias)

    def forward(self, x):
        return self.features(x)


# ── TCN: Bloco Residual 1D ────────────────────────────────────────────────────

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        # Calculo do padding para manter a mesma dimensão temporal (same padding)
        # padding = (kernel_size - 1) * dilation // 2  (funciona bem para kernel ímpar)
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            padding=padding, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.gelu1 = nn.GELU()

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, 
            padding=padding, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.gelu2 = nn.GELU()

        # Se mudou de canal, ajusta o residual com conv 1x1
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        return self.gelu2(out + residual)


# ── Modelo Principal ──────────────────────────────────────────────────────────

class Model(nn.Module):
    def __init__(self, num_classes, tcn_channels=256):
        super().__init__()

        # Backbone
        self.backbone = ConvNeXtExtractor(pretrained=True)
        # A saída do ConvNeXt-Tiny tem shape: [Batch, 768, F_feat, T_feat]

        # Redutor de dimensionalidade: comprimir canais e achatar frequência
        # O ConvNeXt deixa F_feat = 4 (se a entrada Freq for ~128)
        # Vamos comprimir os canais para tcn_channels
        self.channel_compress = nn.Sequential(
            nn.Conv2d(768, tcn_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(tcn_channels),
            nn.GELU()
        )

        # Após o squeeze espacial (pooling na Freq), os canais serão tcn_channels
        
        # Temporal Convolutional Network
        # Blocos com dilatação crescente (aumenta o campo de visão no tempo)
        self.tcn = nn.Sequential(
            TCNBlock(tcn_channels, tcn_channels, kernel_size=3, dilation=1),
            TCNBlock(tcn_channels, tcn_channels, kernel_size=3, dilation=2),
            TCNBlock(tcn_channels, tcn_channels, kernel_size=3, dilation=4),
            TCNBlock(tcn_channels, tcn_channels, kernel_size=3, dilation=8),
        )

        # Adaptive Pooling Global no tempo
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classificador final
        self.fc = nn.Sequential(
            nn.LayerNorm(tcn_channels),
            nn.Dropout(0.3),
            nn.Linear(tcn_channels, num_classes)
        )

    def forward(self, x):
        # x: [Batch, 1, Freq, Tempo]
        x = self.backbone(x)           
        # x: [Batch, 768, F_feat, T_feat]

        x = self.channel_compress(x)
        # x: [Batch, tcn_channels, F_feat, T_feat]

        # Squeeze na dimensão de frequência (dim 2)
        # Tira a média na altura (F_feat)
        x = x.mean(dim=2)              
        # x: [Batch, tcn_channels, T_feat]

        # Passa pela TCN
        x = self.tcn(x)
        # x: [Batch, tcn_channels, T_feat]

        # Global Average Pooling no tempo
        x = self.global_pool(x)        
        # x: [Batch, tcn_channels, 1]

        x = x.squeeze(-1)              
        # x: [Batch, tcn_channels]

        logits = self.fc(x)
        # logits: [Batch, num_classes]

        return logits


if __name__ == "__main__":
    # Teste rápido
    model = Model(num_classes=11)
    
    # Batch=2, Canal=1, Mel=128, Tempo=427 (Aprox 5s de áudio)
    test_input = torch.randn(2, 1, 128, 427)
    output = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total de parâmetros: {total_params:,}")
