# Sugestões para Melhorar a Acurácia do Modelo

## Diagnóstico do Estado Atual

| Componente | Configuração Atual |
|---|---|
| **Backbone** | ConvNeXt-Tiny (pré-treinado, adaptado para 1 canal) |
| **Temporal** | BiLSTM (2 camadas, hidden=256) |
| **Dropout** | 0.3 |
| **Augmentation** | Apenas 1 técnica aleatória por amostra |
| **Dataset** | 21 classes × 200 amostras (totalmente balanceado) |
| **Split** | 80/10/10 (random, seed=42) |
| **Loss** | CrossEntropyLoss (sem pesos) |
| **LR** | 0.0001 (Adam) |
| **Early stopping** | Val Loss, patience=10 |

> [!NOTE]
> O dataset está perfeitamente **balanceado** (200 amostras/classe), então o problema de confusão entre *pariri* e *juriti-de-testa-branca* é quase certamente **acústico** — as vocalizações têm padrões espectrais similares — e não de desbalanceamento.

---

## Sugestões Priorizadas (Maior Impacto Primeiro)

### 🥇 1. Augmentation Composta + SpecAugment Mais Agressivo

**Problema atual:** `apply_random_augmentation` aplica **apenas 1** técnica por amostra. Isso limita a diversidade sintética.

**Solução:** Aplicar **2–3 técnicas encadeadas** com probabilidade independente, e aumentar os parâmetros:

```python
def apply_augmentation_pipeline(spectrogram, p=0.5):
    """Aplica múltiplas técnicas com probabilidade independente."""
    if random.random() < p:
        spectrogram = add_noise(spectrogram, noise_factor=0.005)
    if random.random() < p:
        spectrogram = shift_time(spectrogram, max_shift=20)
    if random.random() < p:
        spectrogram = frequency_mask(spectrogram, num_masks=2, max_mask_width=15)
    if random.random() < p:
        spectrogram = time_mask(spectrogram, num_masks=2, max_mask_width=20)
    if random.random() < p:
        spectrogram = gain_adjustment(spectrogram)
    return spectrogram
```

> [!TIP]
> Adicione também **pitch shift simulado** (deslocar linhas de frequência inteiras) e **mixup** entre espectrogramas da mesma classe para forçar o modelo a aprender padrões invariantes.

---

### 🥇 2. Substituir o Split Aleatório por Split Estratificado

**Problema atual:** `random_split` pode colocar espectrogramas do **mesmo áudio** tanto no treino quanto no teste (data leakage por áudio).

**Solução:** Usar `sklearn.model_selection.StratifiedShuffleSplit` agrupando por arquivo de áudio original (prefix de nome de arquivo), garantindo que segmentos do mesmo gravação fiquem no mesmo fold.

```python
from sklearn.model_selection import train_test_split

# Obtém índices e labels
indices = list(range(len(full_dataset)))
labels = [full_dataset.samples[i][1] for i in indices]

# Split estratificado
train_idx, temp_idx, _, temp_labels = train_test_split(
    indices, labels, test_size=0.2, stratify=labels, random_state=42
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, stratify=temp_labels, random_state=42
)
```

---

### 🥈 3. Label Smoothing na Loss Function

**Problema:** `CrossEntropyLoss` pura pode fazer o modelo ser super-confiante em classes confusas, agravando a confusão entre pariri/juriti.

**Solução:** Usar `label_smoothing=0.1` (disponível desde PyTorch 1.10):

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

Isso força o modelo a manter alguma incerteza, melhorando a calibração e geralmente a generalização em classes similares.

---

### 🥈 4. Learning Rate Scheduler

**Problema atual:** LR fixo em 0.0001 para todas as épocas. Com ConvNeXt pré-treinado, isso pode ser agressivo demais no início e lento demais no fim.

**Solução:** Adicionar `CosineAnnealingLR` ou `ReduceLROnPlateau`:

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
# ou
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# No loop de treino, após optimizer.step():
scheduler.step(val_loss)  # para ReduceLROnPlateau
# scheduler.step()        # para CosineAnnealingLR
```

---

### 🥈 5. Fine-tuning em Camadas (Diferencial de LR)

**Problema:** O backbone ConvNeXt é treinado com o mesmo LR que a cabeça BiLSTM+FC, podendo destruir os pesos pré-treinados.

**Solução:** Usar LR diferencial — menor LR para o backbone, maior para as camadas novas:

```python
optimizer = optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-5},   # backbone: LR pequeno
    {'params': model.rnn.parameters(),      'lr': 1e-4},   # RNN
    {'params': model.fc.parameters(),       'lr': 1e-4},   # cabeça
])
```

---

### 🥉 6. Atenção Temporal na LSTM

**Problema:** O modelo usa `torch.mean(out, dim=1)` para pooling temporal — isso trata todos os timesteps igualmente, mesmo que só parte do espectrograma tenha o canto relevante.

**Solução:** Adicionar um mecanismo de **atenção** simples:

```python
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):  # [B, T, H]
        scores = self.attn(lstm_out).squeeze(-1)  # [B, T]
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, T, 1]
        return (lstm_out * weights).sum(dim=1)  # [B, H]
```

Substitui o `torch.mean(out, dim=1)` no forward do modelo. Isso é especialmente útil quando o canto da espécie ocupa apenas uma fração do espectrograma.

---

### 🥉 7. Análise de Confusão Focada + Hard Negative Mining

**Para as classes problemáticas específicas (pariri × juriti-de-testa-branca):**

1. **Visualize os espectrogramas** das amostras confundidas — verifique se há realmente sobreposição espectral ou se são erros de rotulagem no dataset WikiAves.
2. **Hard Negative Mining:** Dê peso maior para as amostras erradas nas épocas seguintes usando `WeightedRandomSampler`.
3. **Verifique a qualidade das gravações** dessas duas classes — gravações com ruído de fundo alto podem ser a causa raiz.

---

### 🥉 8. Normalização Global vs. Local

**Problema atual:** A normalização `(spec - spec.mean()) / (spec.std() + 1e-6)` é **por amostra** (local). Isso remove a informação de amplitude absoluta que pode diferenciar espécies.

**Solução:** Calcular média e std global do dataset e usar para normalizar:

```python
# Pré-calcular estatísticas globais (fazer uma vez)
all_means = [np.load(f).mean() for f, _ in dataset.samples]
global_mean = np.mean(all_means)
global_std = np.std(all_means)

# No __getitem__:
spec = (spec - global_mean) / (global_std + 1e-6)
```

---

## Plano de Experimentos Sugerido

| Prioridade | Mudança | Impacto Esperado | Implementação |
|---|---|---|---|
| 1 | Augmentation composta (pipeline) | Alto | `augmenter.py` |
| 2 | Label smoothing (0.1) | Médio | `train.py`, 1 linha |
| 3 | LR Scheduler (CosineAnnealing) | Médio | `train.py` |
| 4 | Split estratificado | Médio | `run.py` |
| 5 | LR diferencial backbone/cabeça | Médio | `run.py` |
| 6 | Atenção temporal | Alto | `model_CRNN_ConvNeXt_Tiny_BiLSTM.py` |
| 7 | Análise visual das confusões | Diagnóstico | Notebook |

> [!IMPORTANT]
> Faça as mudanças **uma de cada vez**, re-treinando e comparando o CSV de histórico para isolar o impacto de cada melhoria. O sistema de sufixo numérico de pastas (`result/modelo_2`, `_3`...) já está pronto para isso.

---

## Arquitetura Alternativa: Attention-CRNN

Se após as melhorias acima o modelo ainda confundir as espécies, considere criar um `model_CRNN_ConvNeXt_Tiny_AttentionBiLSTM.py` adicionando:
- Atenção temporal (item 6 acima)
- **Multi-head self-attention** após a LSTM para capturar dependências de longo alcance
- **Squeeze-and-Excitation** no backbone para reponderar canais de frequência relevantes

Isso é a versão mais poderosa e pode ser implementada como um novo arquivo de modelo sem alterar o pipeline de treino.
