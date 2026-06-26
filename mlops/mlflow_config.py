import mlflow
import os
import subprocess
import yaml

EXPERIMENT_NAME = "WikiAves-BioacousticClassifier"

# Raiz do repositório (um nível acima de mlops/)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DVC_FILE = os.path.join(_PROJECT_ROOT, "dataset.dvc")


def setup_mlflow():
    """Configura o tracking URI e o experimento MLflow."""
    # Usa o banco SQLite na raiz do projeto.
    _db_path = os.path.join(_PROJECT_ROOT, "mlflow.db").replace("\\", "/")
    mlflow.set_tracking_uri(f"sqlite:///{_db_path}")
    mlflow.set_experiment(EXPERIMENT_NAME)




def get_dataset_version() -> dict:
    """
    Extrai a versão do dataset rastreado pelo DVC.

    Retorna um dict com:
        hash_short : str  —  primeiros 7 caracteres do MD5 do dataset (ex: '4ad1c1b')
        hash_full  : str  —  hash MD5 completo (ex: '4ad1c1b6dd0d00930acc944481d23767.dir')
        git_commit : str  —  commit Git que alterou dataset.dvc pela última vez
        tag        : str  —  tag Git legível associada ao commit (ex: 'dataset-v2'), ou ''
    """
    result = {
        "hash_short": "unknown",
        "hash_full": "unknown",
        "git_commit": "unknown",
        "tag": "",
    }

    # ── 1. Ler o hash MD5 direto do dataset.dvc ────────────────────────────────
    try:
        with open(_DVC_FILE, "r", encoding="utf-8") as f:
            dvc_meta = yaml.safe_load(f)
        # dataset.dvc tem formato: outs: [{ md5: "...", path: "dataset", ... }]
        md5_full = dvc_meta["outs"][0]["md5"]
        result["hash_full"] = md5_full
        result["hash_short"] = md5_full.split(".")[0][:7]
    except Exception:
        pass

    # ── 2. Commit Git que alterou dataset.dvc por último ───────────────────────
    try:
        commit = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", "dataset.dvc"],
            cwd=_PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if commit:
            result["git_commit"] = commit[:7]
    except Exception:
        pass

    # ── 3. Tag Git associada a esse commit (se existir) ────────────────────────
    try:
        if result["git_commit"] != "unknown":
            tag = subprocess.check_output(
                ["git", "tag", "--points-at", result["git_commit"]],
                cwd=_PROJECT_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            # Se houver múltiplas tags, pega a primeira que contenha "dataset"
            if tag:
                tags = tag.splitlines()
                ds_tags = [t for t in tags if "dataset" in t.lower()]
                result["tag"] = ds_tags[0] if ds_tags else tags[0]
    except Exception:
        pass

    return result
