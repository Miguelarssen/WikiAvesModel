"""
predict.py — Inferência usando modelos do MLflow Model Registry.

Carrega um modelo registrado no MLflow Model Registry e faz predição
em arquivos de espectrograma (.npy).

Uso:
    python predict.py --model CRNN_ConvNeXt_Tiny_BiLSTM --input audio.npy
    python predict.py --model CRNN_ConvNeXt_Tiny_BiLSTM --version 2 --input audio.npy
    python predict.py --list-models
    python predict.py --model CRNN_ConvNeXt_Tiny_BiLSTM --list-versions
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import mlflow
import mlflow.pytorch
from pathlib import Path

# Permite importar mlops/ estando dentro da pasta model/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from mlops.mlflow_config import setup_mlflow


def list_registered_models():
    """Lista todos os modelos registrados no MLflow Model Registry."""
    setup_mlflow()
    client = mlflow.tracking.MlflowClient()
    models = client.search_registered_models()

    if not models:
        print("[INFO] Nenhum modelo registrado no MLflow Model Registry.")
        return

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║              MODELOS REGISTRADOS NO MLFLOW                 ║")
    print("╠══════════════════════════════════════════════════════════════╣")

    for rm in models:
        latest = rm.latest_versions
        versions_str = ", ".join([f"v{v.version}" for v in latest])
        print(f"║  📦 {rm.name}")
        print(f"║     Versões: {versions_str}")
        print("╠══════════════════════════════════════════════════════════════╣")

    print("╚══════════════════════════════════════════════════════════════╝\n")


def list_model_versions(model_name: str):
    """Lista todas as versões de um modelo com detalhes do dataset usado."""
    setup_mlflow()
    client = mlflow.tracking.MlflowClient()

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception:
        print(f"[ERRO] Modelo '{model_name}' não encontrado no Registry.")
        return

    if not versions:
        print(f"[INFO] Nenhuma versão encontrada para '{model_name}'.")
        return

    print(f"\n╔══════════════════════════════════════════════════════════════╗")
    print(f"║  Versões do modelo: {model_name}")
    print(f"╠══════════════════════════════════════════════════════════════╣")

    for v in sorted(versions, key=lambda x: int(x.version)):
        # Buscar as tags do run original para mostrar o dataset
        run = client.get_run(v.run_id)
        tags = run.data.tags
        ds_version = tags.get("dataset_version", "?")
        ds_hash = tags.get("dataset_hash_full", "?")
        git_commit = tags.get("git_commit", "?")

        # Buscar métricas de teste
        metrics = run.data.metrics
        test_acc = metrics.get("test_acc", None)
        test_f1 = metrics.get("test_f1", None)

        print(f"║")
        print(f"║  🔖 Versão {v.version}")
        print(f"║     Dataset:    {ds_version} ({ds_hash})")
        print(f"║     Git commit: {git_commit}")
        if test_acc is not None:
            print(f"║     Acurácia:   {test_acc:.4f}")
        if test_f1 is not None:
            print(f"║     F1-Score:   {test_f1:.4f}")
        print(f"║     Criado em:  {v.creation_timestamp}")
        print(f"╠══════════════════════════════════════════════════════════════╣")

    print("╚══════════════════════════════════════════════════════════════╝\n")


def predict(model_name: str, version: int | None, input_path: str):
    """Carrega modelo do Registry e faz predição."""
    setup_mlflow()

    # Monta o URI do modelo
    if version:
        model_uri = f"models:/{model_name}/{version}"
        print(f"[INFO] Carregando modelo '{model_name}' versão {version}...")
    else:
        model_uri = f"models:/{model_name}/latest"
        print(f"[INFO] Carregando modelo '{model_name}' (versão mais recente)...")

    try:
        model = mlflow.pytorch.load_model(model_uri)
    except Exception as e:
        print(f"[ERRO] Não foi possível carregar o modelo: {e}")
        sys.exit(1)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Carregar e pré-processar o espectrograma
    print(f"[INFO] Processando arquivo: {input_path}")
    spec = np.load(input_path)
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, Freq, Tempo]
    spec = spec.to(device)

    # Predição
    with torch.no_grad():
        outputs = model(spec)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = probabilities.max(1)

    # Tentar carregar label_map do dataset
    dataset_dir = Path(__file__).parent.parent / "dataset"
    class_names = {}
    if dataset_dir.exists():
        for idx, class_dir in enumerate(sorted(dataset_dir.iterdir())):
            if class_dir.is_dir():
                class_names[idx] = class_dir.name

    predicted_idx = predicted_class.item()
    predicted_name = class_names.get(predicted_idx, f"Classe_{predicted_idx}")
    conf_pct = confidence.item() * 100

    print(f"\n{'='*50}")
    print(f"  🐦 Predição: {predicted_name}")
    print(f"  📊 Confiança: {conf_pct:.2f}%")
    print(f"  🔢 Índice:   {predicted_idx}")
    print(f"{'='*50}")

    # Top-5 predições
    top5_prob, top5_idx = probabilities.topk(min(5, probabilities.shape[1]))
    print(f"\n  Top-{min(5, probabilities.shape[1])} predições:")
    for i in range(top5_prob.shape[1]):
        idx = top5_idx[0, i].item()
        name = class_names.get(idx, f"Classe_{idx}")
        prob = top5_prob[0, i].item() * 100
        print(f"    {i+1}. {name:30s} {prob:6.2f}%")

    return predicted_idx, conf_pct


def main():
    parser = argparse.ArgumentParser(
        description="Inferência usando modelos do MLflow Model Registry"
    )
    parser.add_argument("--model", type=str, help="Nome do modelo registrado")
    parser.add_argument("--version", type=int, default=None,
                        help="Versão do modelo (padrão: mais recente)")
    parser.add_argument("--input", type=str, help="Caminho para o arquivo .npy")
    parser.add_argument("--list-models", action="store_true",
                        help="Listar todos os modelos registrados")
    parser.add_argument("--list-versions", action="store_true",
                        help="Listar versões de um modelo (requer --model)")

    args = parser.parse_args()

    if args.list_models:
        list_registered_models()
        return

    if args.list_versions:
        if not args.model:
            print("[ERRO] --list-versions requer --model <nome>")
            sys.exit(1)
        list_model_versions(args.model)
        return

    if not args.model or not args.input:
        parser.print_help()
        print("\n[ERRO] --model e --input são obrigatórios para predição.")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"[ERRO] Arquivo não encontrado: {args.input}")
        sys.exit(1)

    predict(args.model, args.version, args.input)


if __name__ == "__main__":
    main()
