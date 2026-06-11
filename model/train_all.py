import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import importlib
import traceback
import sys

from dataset import avesDataset, SpectrogramAugment, TransformedSubset
from train import train_model

def main():
    print("[INFO] Preparando dataset para treinamento global...")
    data_dir = "../dataset"
    full_dataset = avesDataset(data_dir)
    num_classes = len(full_dataset.label_map)
    print(f"[INFO] Classes detectadas: {num_classes}")

    # Split estratificado por label idêntico ao do run.ipynb
    all_indices = list(range(len(full_dataset)))
    all_labels  = [full_dataset.samples[i][1] for i in all_indices]

    train_idx, temp_idx, _, temp_labels = train_test_split(
        all_indices, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=42
    )

    print(f"[INFO] Split estratificado — treino: {len(train_idx)} | val: {len(val_idx)} | teste: {len(test_idx)}")

    train_subset = Subset(full_dataset, train_idx)
    val_subset   = Subset(full_dataset, val_idx)
    test_subset  = Subset(full_dataset, test_idx)

    # Augmentation apenas no treino
    train_dataset = TransformedSubset(train_subset, transform=SpectrogramAugment())
    val_dataset   = val_subset
    test_dataset  = test_subset

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

    # Lista de todos os modelos
    model_modules = [
        "models.model_CNN_ResNet18",
        "models.model_CNN_EfficientNetB0",
        "models.model_CNN_ConvNeXt_Tiny",
        "models.model_CNN_ConvNeXt_Tiny_TCN",
        "models.model_CRNN_ResNet18_BiLSTM",
        "models.model_CRNN_EfficientNetB0_BiLSTM",
        "models.model_CRNN_ConvNeXt_Tiny_BiLSTM",
        "models.model_CRNN_ConvNeXt_Tiny_DeepConv_BiLSTM",
    ]

    print(f"\n[INFO] Iniciando pipeline de treinamento para {len(model_modules)} modelos.")
    
    for idx, mod_name in enumerate(model_modules):
        print(f"\n{'='*60}")
        print(f"[MODELO {idx+1}/{len(model_modules)}] Treinando: {mod_name}")
        print(f"{'='*60}")
        
        try:
            mod = importlib.import_module(mod_name)
            # Instancia o modelo
            model = mod.Model(num_classes=num_classes)
            
            # Treina
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
            print(f"[OK] Treinamento de {mod_name} concluído com sucesso!")
        except Exception as e:
            print(f"[ERRO] Falha ao treinar {mod_name}: {e}")
            traceback.print_exc()
            print("[INFO] Continuando com o próximo modelo...\n")

    print(f"\n{'='*60}")
    print("[INFO] Pipeline global finalizado!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
