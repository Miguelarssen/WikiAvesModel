import mlflow
import os

EXPERIMENT_NAME = "WikiAves-BioacousticClassifier"

def setup_mlflow():
    # Salva os runs na pasta mlruns/ dentro do projeto
    mlflow.set_tracking_uri("file:///d:/Estudos UTFPR/WikiAvesModel/mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)
