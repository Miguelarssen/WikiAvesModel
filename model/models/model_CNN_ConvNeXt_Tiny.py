"""
Modelo: CNN_ConvNeXt_Tiny_AudioClassifier

Arquitetura:
ConvNeXt-Tiny (Pure CNN)

Backbone:
ConvNeXt-Tiny (pré-treinada, adaptada para 1 canal)

Descrição:
Arquitetura moderna que utiliza kernels grandes e design inspirado em Transformers.
Considerado um dos modelos convolucionais mais robustos atualmente.

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
        
        # Carregar pesos pré-treinados
        if hasattr(models, 'ConvNeXt_Tiny_Weights'):
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            convnext = models.convnext_tiny(weights=weights)
        else:
            convnext = models.convnext_tiny(pretrained=pretrained)
        
        self.features = convnext.features
        
        # Adaptar o Stem para 1 canal
        # convnext.features[0] é o DownsampleLayer (stem)
        # convnext.features[0][0] é a Conv2d(3, 96, kernel_size=4, stride=4)
        original_stem_conv = self.features[0][0]
        
        self.features[0][0] = nn.Conv2d(
            1, 
            original_stem_conv.out_channels, 
            kernel_size=original_stem_conv.kernel_size, 
            stride=original_stem_conv.stride, 
            padding=original_stem_conv.padding, 
            bias=original_stem_conv.bias is not None
        )
        
        # Copiar pesos usando a média (preserva a escala das ativações original)
        if pretrained:
            with torch.no_grad():
                self.features[0][0].weight.copy_(original_stem_conv.weight.mean(dim=1, keepdim=True))
                if original_stem_conv.bias is not None:
                    self.features[0][0].bias.copy_(original_stem_conv.bias)

    def forward(self, x):
        return self.features(x)


class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        
        self.backbone = ConvNeXtExtractor(pretrained=True)
        
        # Pooling global
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # No ConvNeXt, o LayerNorm é essencial antes da camada linear
        # Renomeamos para self.fc para compatibilidade com o script de relatório
        self.fc = nn.Sequential(
            nn.LayerNorm(768, eps=1e-06),
            nn.Flatten(1),
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        # x: [Batch, 1, Freq, Tempo]
        features = self.backbone(x)           # [Batch, 768, H_feat, W_feat]
        features = self.global_pool(features) # [Batch, 768, 1, 1]
        
        # Prepara para o LayerNorm (espera o canal na última dimensão ou tensor flat)
        features = torch.flatten(features, 1) # [Batch, 768]
        
        logits = self.fc(features)
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
