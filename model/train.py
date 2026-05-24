import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import inspect
import csv
import mlflow
import subprocess
import mlflow.pytorch
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from dataset import avesDataset
from utils.report import generate_report

# Permite importar mlops/ estando dentro da pasta model/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from mlops.mlflow_config import setup_mlflow

def get_dataset():

    data_dir = "../dataset"

    dataset = avesDataset(root_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    return dataloader


def get_model_name(model):
    # pega o arquivo onde a classe foi definida
    file_path = inspect.getfile(model.__class__)
    
    # pega só o nome do arquivo
    file_name = os.path.basename(file_path)
    
    # remove .py
    name = file_name.replace(".py", "")
    
    # remove prefixo model_
    name = name.replace("model_", "")
    
    return name

def train_model(model, num_epochs, train_loader, val_loader, test_loader=None, 
                lr=0.001, patience=5, min_delta=0.0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Tenta descobrir o num_classes dinamicamente do dataset
    curr_ds = train_loader.dataset
    real_ds = None
    temp_ds = curr_ds
    while temp_ds is not None:
        if hasattr(temp_ds, 'label_map'):
            real_ds = temp_ds
            break
        # Tenta .subset (TransformedSubset) ou .dataset (random_split Subset)
        next_ds = getattr(temp_ds, 'subset', None)
        if next_ds is None:
            next_ds = getattr(temp_ds, 'dataset', None)
        temp_ds = next_ds
    
    if real_ds:
        num_classes = len(real_ds.label_map)
        print(f"[INFO] Detectadas {num_classes} classes dinamicamente.")
    else:
        num_classes = 11 # fallback
        print(f"[AVISO] Nao foi possivel detectar classes. Usando padrao: {num_classes}")

    # Modelo
    # Se o modelo passado já for uma instância (nn.Module), apenas move para o device
    if isinstance(model, nn.Module):
        model = model.to(device)
    else:
        # Se for uma classe, instancia com o num_classes detectado
        model = model(num_classes=num_classes).to(device)

    model_name = get_model_name(model)
    save_dir = os.path.join("result", model_name)
    
    # Se a pasta já existir, adiciona um sufixo numérico (_2, _3, etc.)
    if os.path.exists(save_dir):
        counter = 2
        new_save_dir = f"{save_dir}_{counter}"
        while os.path.exists(new_save_dir):
            counter += 1
            new_save_dir = f"{save_dir}_{counter}"
        save_dir = new_save_dir
        
    os.makedirs(save_dir, exist_ok=True)

    # ── MLflow: iniciar rastreamento do experimento ────────────────────────────
    setup_mlflow()
    mlflow.start_run(run_name=model_name)
    mlflow.log_params({
        "lr":          lr,
        "num_epochs":  num_epochs,
        "patience":    patience,
        "num_classes": num_classes,
        "batch_size":  train_loader.batch_size,
        "model_name":  model_name,
    })

    # ── DVC + MLflow: rastrear versão do dataset ───────────────────────────────
    # Lê o dataset.dvc commitado no Git e extrai o hash MD5 dos dados.
    # Isso conecta cada run MLflow à versão exata do dataset que o gerou.
    _repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    try:
        dvc_file_content = subprocess.check_output(
            ["git", "show", "HEAD:dataset.dvc"],
            cwd=_repo_root, text=True, stderr=subprocess.DEVNULL
        )
        _dataset_hash = "unknown"
        for _line in dvc_file_content.splitlines():
            if "md5:" in _line and ".dir" in _line:
                _dataset_hash = _line.strip().split(": ")[1]
                break
        mlflow.set_tag("dataset_dvc_hash", _dataset_hash)
    except Exception:
        mlflow.set_tag("dataset_dvc_hash", "unknown")

    # Loga também o commit Git atual para rastreabilidade total do código
    try:
        _git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        mlflow.set_tag("git_commit", _git_commit)
    except Exception:
        mlflow.set_tag("git_commit", "unknown")


    # Tópico 3: Label Smoothing — força o modelo a manter incerteza,
    # melhorando calibração em classes acusticamente similares (ex: pariri × juriti).
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Histórico
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # Early stopping (Monitorando VAL_LOSS)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    start_time = time.time()

    # Loop de treino
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        # Validação
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total

        # Salvar histórico
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # ── MLflow: métricas por época ─────────────────────────────────────────
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "train_acc":  train_acc,
            "val_acc":    val_acc,
        }, step=epoch)

        print("Epoca [" + str(epoch+1) + "/" + str(num_epochs) + "] - "
              "Loss Treino: " + format(train_loss, ".4f") + ", Acc Treino: " + format(train_acc, ".2f") + "% | "
              "Loss Val: " + format(val_loss, ".4f") + ", Acc Val: " + format(val_acc, ".2f") + "%")

        # Checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_loss': train_loss,
                'val_loss': val_loss
            }, checkpoint_path)

            print("--- Checkpoint salvo em: " + checkpoint_path)

        # Early stopping (Melhora se a Loss diminuir)
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0

            best_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)

            print("[OK] Melhor modelo atualizado!")
        else:
            epochs_no_improve += 1
            print("[WAIT] Sem melhora por " + str(epochs_no_improve) + " épocas")


        if epochs_no_improve >= patience:
            print("[STOP] Early stopping ativado!")
            break

    end_time = time.time()
    duration = end_time - start_time
    hours, rem = divmod(duration, 3600)
    minutes, seconds = divmod(rem, 60)
    training_time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    print(f"[INFO] Treinamento concluído em {training_time_str}")

    # --- Geração de Relatório ---
    if test_loader:
        print("\n[STATS] Avaliando no conjunto de teste para o relatório final...")
        model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth")))
        model.eval()
        
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        # Obter nomes das classes
        curr_ds = train_loader.dataset
        real_ds = None
        
        # Procura recursivamente pelo dataset original que contém o label_map
        temp_ds = curr_ds
        while temp_ds is not None:
            if hasattr(temp_ds, 'label_map'):
                real_ds = temp_ds
                break
            temp_ds = getattr(temp_ds, 'dataset', None)
        
        if real_ds:
            class_names = [name for name, _ in sorted(real_ds.label_map.items(), key=lambda x: x[1])]
        else:
            print("[AVISO] Nao foi possivel encontrar label_map no dataset. Usando nomes genéricos.")
            num_classes = model.fc.out_features
            class_names = [format(i, "d") for i in range(num_classes)]
        
        history = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_acc': train_accs,
            'val_acc': val_accs
        }
        
        generate_report(
            save_dir=save_dir,
            history=history,
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            training_time=training_time_str
        )

        print("[STATS] Salvando dados no historico geral CSV...")
        csv_path = os.path.join("result", "historico_geral_treinos.csv")
        headers = ['Data/Hora', 'Modelo', 'Total Epocas', 'Melhor Acc Val', 'Acc Teste', 'Precisao Teste', 'Recall Teste', 'F1 Teste', 'Classes', 'Tempo de Execução']
        
        # Verifica se o arquivo existe e se precisa de atualização de colunas
        if os.path.exists(csv_path):
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                existing_headers = next(reader, None)
            
            if existing_headers and 'Tempo de Execução' not in existing_headers:
                print("[INFO] Atualizando estrutura do CSV para incluir novas colunas...")
                rows = []
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        for h in headers:
                            if h not in row: row[h] = "" # Preenche campos antigos como nulos
                        rows.append(row)
                
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(rows)

        file_exists = os.path.exists(csv_path)
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model_name,
                len(train_losses),
                f"{max(val_accs):.2f}",
                f"{acc:.4f}",
                f"{precision:.4f}",
                f"{recall:.4f}",
                f"{f1:.4f}",
                len(class_names),
                training_time_str
            ])
        print("[OK] Dados salvos em " + csv_path)

        # ── MLflow: métricas finais de teste e artefatos ───────────────────────
        mlflow.log_metrics({
            "test_acc":       acc,
            "test_precision": precision,
            "test_recall":    recall,
            "test_f1":        f1,
        })
        mlflow.log_artifact(os.path.join(save_dir, "best_model.pth"))
        report_file = os.path.join(save_dir, "report.pdf")
        if os.path.exists(report_file):
            mlflow.log_artifact(report_file)
        mlflow.pytorch.log_model(model, artifact_path="model")

    mlflow.end_run()
    return model, train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, loader, criterion, device, phase_name="Teste"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(loader)
    print(f"\n--- Resultado {phase_name} ---")
    print(f"Loss: {avg_loss:.4f} | Acurácia: {accuracy:.2f}%")
    return accuracy
