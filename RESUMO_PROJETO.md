# WikiAves Bioacoustics Classifier — Resumo do Projeto

> **TCC:** Uma Arquitetura MLOps para o Ciclo de Vida de Modelos de Bioacústica: Um Estudo de Caso com Dados do WikiAves  
> **Curso:** Ciência da Computação — UTFPR  
> **Data de conclusão dos experimentos:** maio–junho de 2026

---

## 1. Visão Geral

Este projeto desenvolveu um sistema de **classificação automática de vocalizações de aves brasileiras** da família Columbidae, integrando técnicas modernas de Deep Learning com uma infraestrutura de MLOps para rastreamento e versionamento de experimentos.

O escopo final foi redimensionado (em alinhamento com a banca avaliadora) em relação ao projeto original, focando em:

- Extração e preparação de uma base de dados bioacústica original do WikiAves
- Treinamento e avaliação de modelos de Deep Learning com backbone pré-treinado
- Infraestrutura de MLOps de Nível 1 de maturidade (rastreamento + versionamento)

---

## 2. Base de Dados

### Fonte
- **Plataforma:** [WikiAves](https://www.wikiaves.com.br) — repositório colaborativo de aves brasileiras
- **Módulo:** `scrapping/spiderSTFT.ipynb`

### Critérios de seleção
| Parâmetro | Valor |
|-----------|-------|
| Família taxonômica | Columbidae (pombas, juritis, avoante...) |
| Mínimo de gravações por espécie | 200 |
| Espécies selecionadas | **27** |
| Amostras totais | **10.794** |

### Pipeline de extração e processamento
1. **Web Scraping** — listagem das espécies elegíveis via parsing HTML da tabela de espécies do WikiAves
2. **Download** — áudios em MP3 via endpoint JSON `getRegistrosJSON.php`
3. **Pré-processamento por áudio:**
   - Reamostrado para **44.100 Hz** (fixo para todo o dataset)
   - Convertido para **mono**
   - Segmentado em **janelas de 5 segundos** centradas na região de maior energia (RMS)
4. **Transformação em espectrograma (STFT):**
   - `n_fft = 2048`, `hop_length = 512`, `center = False`
   - Convertido para escala logarítmica (dB)
   - Salvo como array NumPy `.npy` — shape: `(1025 × N_frames)`
5. **Versionamento** com DVC — hash do dataset: `4ad1c1b`

### Estrutura do dataset
```
dataset/
├── pomba-asa-branca/    # 200 arquivos .npy
├── pomba-galega/
├── pariri/
├── juriti-pupu/
├── avoante/
└── ... (27 espécies no total)
```

---

## 3. Modelos Desenvolvidos

Todos os modelos foram implementados em **PyTorch** com backbone **ConvNeXt-Tiny pré-treinado** (ImageNet), adaptado para entrada monocanal (1 canal em vez de 3 RGB).

### 3.1 Arquiteturas disponíveis

| Arquivo | Descrição |
|---------|-----------|
| `model_CNN_ResNet18.py` | ResNet-18 adaptada para 1 canal |
| `model_CNN_EfficientNetB0.py` | EfficientNet-B0 para espectrogramas |
| `model_CNN_ConvNeXt_Tiny.py` | ConvNeXt-Tiny puro (sem módulo temporal) |
| `model_CNN_ConvNeXt_Tiny_TCN.py` | ConvNeXt-Tiny + Temporal Convolutional Network |
| `model_CRNN_ResNet18_BiLSTM.py` | ResNet-18 + BiLSTM |
| `model_CRNN_EfficientNetB0_BiLSTM.py` | EfficientNet-B0 + BiLSTM |
| `model_CRNN_ConvNeXt_Tiny_BiLSTM.py` | **ConvNeXt-Tiny + BiLSTM** ⭐ |
| `model_CRNN_ConvNeXt_Tiny_DeepConv_BiLSTM.py` | **ConvNeXt-Tiny + DeepConv Neck + BiLSTM** ⭐ |

### 3.2 Modelos com experimentos completos

#### CRNN_ConvNeXt_Tiny_BiLSTM
- **Backbone:** ConvNeXt-Tiny (768 canais de saída)
- **Temporal:** BiLSTM bidirecional — 2 camadas, 256 unidades ocultas
- **Pooling:** média temporal → Linear + Dropout(0.3)

#### CRNN_ConvNeXt_Tiny_DeepConv_BiLSTM
- **Backbone:** ConvNeXt-Tiny (768 canais)
- **Neck convolucional profundo:** 3 blocos residuais 2D (768→512→384→256)
- **Temporal:** BiLSTM — 2 camadas, 256 unidades ocultas
- Diferença: refinamento extra das features espaciais antes da recorrência

---

## 4. Configurações de Treinamento

| Hiperparâmetro | Valor |
|----------------|-------|
| Split do dataset | 80% treino / 10% validação / 10% teste |
| Split estratificado | Sim (semente 42) |
| Épocas máximas | 50 |
| Batch size | 32 |
| Otimizador | Adam |
| Taxa de aprendizagem | 0,0001 |
| Função de perda | Cross-Entropy + Label Smoothing (α = 0,1) |
| Early stopping (paciência) | 10 épocas |
| Early stopping (delta mínimo) | 0,0001 |
| Dispositivo | CUDA (GPU) |

### Augmentação de dados (aplicada apenas no treino)

Pipeline de 6 técnicas com probabilidade independente de 50% cada:

1. **Ruído gaussiano** — `noise_factor = 0.005`
2. **Deslocamento temporal** — `max_shift = 20 frames`
3. **Máscara de frequência** *(SpecAugment)* — 2 máscaras, até 15 bins
4. **Máscara temporal** *(SpecAugment)* — 2 máscaras, até 20 frames
5. **Ajuste de ganho** — fator entre 0,7 e 1,3
6. **Pitch shift simulado** — deslocamento vertical de até 8 bins

---

## 5. Resultados Obtidos

### Tabela comparativa — conjunto de teste (1.080 amostras, 27 classes)

| Métrica | CRNN-BiLSTM | CRNN-DeepConv-BiLSTM |
|---------|:-----------:|:--------------------:|
| Épocas (early stopping) | 27 | 25 |
| Melhor Acc. Validação | 85,08% | **86,10%** |
| Acurácia no Teste | 85,37% | 85,37% |
| Precisão Ponderada | 0,8592 | **0,8656** |
| Recall Ponderado | 0,8537 | 0,8537 |
| F1-Score Ponderado | 0,8542 | **0,8567** |
| Tempo de treinamento | 01:18:11 | 01:15:40 |

### Destaques por classe (CRNN-DeepConv-BiLSTM)
- **Melhores classes:** classes 24 (F1 = 0,962), 1 e 11 (F1 = 0,950)
- **Classes mais difíceis:** classe 18 (F1 = 0,653) e classe 6 (F1 = 0,700)
- Dificuldade atribuída à semelhança acústica entre espécies da família Columbidae

### Comparação com literatura
| Trabalho | Dataset | Classes | Acc. / F1 aves |
|----------|---------|---------|----------------|
| Sanchez et al. (2021) | NIPS4Bplus | 87 | ≤ 75% (aves) |
| **Este trabalho** | WikiAves (Columbidae) | 27 | **85,37% / 0,857** |

---

## 6. Infraestrutura de MLOps

### Nível de maturidade atingido: **Nível 1** (Google Cloud, 2024)

### Ferramentas integradas

| Ferramenta | Papel |
|------------|-------|
| **MLflow** | Rastreamento de experimentos, Model Registry |
| **DVC** | Versionamento do dataset |
| **Git** | Versionamento do código |

### O que é registrado automaticamente por experimento

- ✅ Hiperparâmetros (lr, batch, epochs, patience, etc.)
- ✅ Métricas por época: `train_loss`, `val_loss`, `train_acc`, `val_acc`
- ✅ Métricas finais de teste: acurácia, precisão, recall, F1
- ✅ Hash do dataset (DVC MD5 — ex: `4ad1c1b`)
- ✅ Commit Git do código (`git_commit` tag)
- ✅ Modelo treinado (`.pth`) como artefato MLflow
- ✅ Relatório em PDF (curvas de aprendizado, matriz de confusão, stats por classe)
- ✅ Registro automático no **MLflow Model Registry** com versão incremental
- ✅ `metrics.json` por experimento para rastreamento externo

### O que NÃO foi implementado (Nível 2 — fora do escopo)

- ❌ Monitoramento de drift em produção
- ❌ Re-treinamento automático por trigger de novos dados do WikiAves
- ❌ Pipeline completo de CI/CD/CT
- ❌ Deploy em ambiente de produção

---

## 7. Estrutura do Repositório

```
WikiAvesModel/
├── scrapping/
│   ├── spiderSTFT.ipynb       # Web scraping + geração de espectrogramas STFT
│   └── spiderMel.ipynb        # Variante com Mel-espectrogramas (experimental)
│
├── dataset/                   # Base de dados (gerenciada pelo DVC)
│   └── <especie>/             # 200 arquivos .npy por espécie
│
├── model/
│   ├── dataset.py             # avesDataset, SpectrogramAugment, TransformedSubset
│   ├── train.py               # Loop de treinamento + integração MLflow + DVC
│   ├── run.ipynb              # Notebook de execução dos experimentos
│   ├── predict.py             # Inferência com modelo treinado
│   ├── models/                # 8 arquiteturas implementadas
│   ├── utils/
│   │   ├── augmenter.py       # Pipeline de augmentação de espectrogramas
│   │   ├── cleaner.py         # Utilitários de limpeza de dados
│   │   └── report.py          # Geração do relatório PDF por experimento
│   └── result/
│       ├── historico_geral_treinos.csv
│       ├── CRNN_ConvNeXt_Tiny_BiLSTM_2/
│       │   ├── best_model.pth
│       │   └── report.pdf
│       └── CRNN_ConvNeXt_Tiny_DeepConv_BiLSTM/
│           ├── best_model.pth
│           ├── metrics.json
│           └── report.pdf
│
├── mlops/
│   └── mlflow_config.py       # Setup do MLflow + extração de versão do DVC
│
├── mlruns/                    # Experimentos rastreados pelo MLflow
│
├── dataset.dvc                # Referência versionada do dataset (hash MD5)
├── dvc.yaml                   # Pipeline DVC
│
└── Metodologia_Resultados_Conclusao_TCC.pdf   # Documento gerado
```

---

## 8. Documento Final Gerado

**Arquivo:** `Metodologia_Resultados_Conclusao_TCC.pdf`  
**Páginas:** 12 | **Formatação:** ABNT (Times 12pt, margens 3/2cm, espaçamento 1,5)

### Seções
- **3. Metodologia** — base de dados, arquiteturas, treinamento, augmentação, MLOps
- **4. Resultados Obtidos** — tabelas comparativas, análise por classe, avaliação do MLOps
- **5. Conclusão** — síntese, limitações, propostas para mestrado (4 direções de pesquisa futuras)
- **Referências** (ABNT) — 18 referências verificadas

---

## 9. Dependências Principais

```
torch / torchvision    # Deep Learning
mlflow                 # Rastreamento de experimentos
dvc                    # Versionamento de dados
librosa                # Processamento de áudio
numpy / pandas         # Manipulação de dados
scikit-learn           # Métricas e split estratificado
requests / bs4         # Web scraping
reportlab              # Geração de PDFs
```
