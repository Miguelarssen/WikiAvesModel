from models.model_CRNN_ConvNeXt_Tiny_BiLSTM import Model
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from dataset import avesDataset, SpectrogramAugment, TransformedSubset
from train import train_model


data_dir = "../dataset"
full_dataset = avesDataset(data_dir)

# Pega o número real de classes do dataset
num_classes = len(full_dataset.label_map)
print(f"[INFO] Classes detectadas: {num_classes}")

# Instancia o modelo com o número correto de classes
model = Model(num_classes=num_classes)

# ──────────────────────────────────────────────────────────────────────────────
# Tópico 2: Split estratificado por label (evita data leakage por classe e
# garante que cada fold tenha a mesma proporção de cada espécie).
# ──────────────────────────────────────────────────────────────────────────────
all_indices = list(range(len(full_dataset)))
all_labels  = [full_dataset.samples[i][1] for i in all_indices]

# 80% treino | 20% temp
train_idx, temp_idx, _, temp_labels = train_test_split(
    all_indices, all_labels,
    test_size=0.2,
    stratify=all_labels,
    random_state=42
)

# 50% do temp → val  |  50% do temp → test  (= 10% / 10% do total)
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    stratify=temp_labels,
    random_state=42
)

print(f"[INFO] Split estratificado — treino: {len(train_idx)} | val: {len(val_idx)} | teste: {len(test_idx)}")

train_subset = Subset(full_dataset, train_idx)
val_subset   = Subset(full_dataset, val_idx)
test_subset  = Subset(full_dataset, test_idx)

# Augmentation apenas no treino
train_dataset = TransformedSubset(train_subset, transform=SpectrogramAugment())
val_dataset   = val_subset
test_dataset  = test_subset

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False)

train_model(
    model=model,
    num_epochs=50,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    patience=10,
    min_delta=0.0001,
    lr=0.0001
)
