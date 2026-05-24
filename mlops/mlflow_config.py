import mlflow
import os

EXPERIMENT_NAME = "WikiAves-BioacousticClassifier"

def setup_mlflow():
    # Salva os runs na pasta mlruns/ dentro do projeto
    # Caminho relativo à raiz do projeto (funciona em qualquer máquina)
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Força minúscula para consistência com o artifact_uri salvo pelo MLflow no Windows
    _mlruns_path = os.path.join(_project_root, "mlruns").replace("\\", "/").lower()
    mlflow.set_tracking_uri(f"file:///{_mlruns_path}")
    mlflow.set_experiment(EXPERIMENT_NAME)
