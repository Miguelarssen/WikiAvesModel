import os
import sys
import glob
import numpy as np

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)           # raiz do projeto
sys.path.insert(0, SCRIPT_DIR)

from bird_segmentation import segment_spectrogram

INPUT_DIR  = os.path.join(PROJECT_DIR, "dataset")       # ..\dataset\
OUTPUT_DIR = os.path.join(PROJECT_DIR, "dataset_seg")   # ..\dataset_seg\

# Descobre todas as subpastas de espécies dentro de dataset/
species_dirs = sorted([
    d for d in os.listdir(INPUT_DIR)
    if os.path.isdir(os.path.join(INPUT_DIR, d))
])

if not species_dirs:
    print(f"[ERRO] Nenhuma subpasta de espécie encontrada em: {INPUT_DIR}")
    sys.exit(1)

print(f"[INFO] {len(species_dirs)} espécies encontradas: {', '.join(species_dirs)}")
print(f"[INFO] Saída será salva em '{OUTPUT_DIR}'")
print("=" * 60)

n_total_processados   = 0
n_total_sem_segmento  = 0
n_total_gerados       = 0

for especie in species_dirs:
    especie_input_dir  = os.path.join(INPUT_DIR, especie)
    especie_output_dir = os.path.join(OUTPUT_DIR, especie)
    os.makedirs(especie_output_dir, exist_ok=True)

    npy_files = sorted(glob.glob(os.path.join(especie_input_dir, "*.npy")))
    total = len(npy_files)

    if total == 0:
        print(f"[SKIP] '{especie}' — nenhum arquivo .npy encontrado")
        continue

    print(f"\n[ESPÉCIE] {especie}  ({total} arquivos)")
    print("-" * 60)

    n_processados      = 0
    n_sem_segmento     = 0
    n_arquivos_gerados = 0
    arquivos_sem_seg   = []

    for idx, filepath in enumerate(npy_files, 1):
        basename = os.path.basename(filepath)

        try:
            S = np.load(filepath)
        except Exception as e:
            print(f"  [WARN] ({idx}/{total}) {basename} — erro ao carregar: {e}")
            continue

        try:
            _, _, _, _, merged, Espec_seg = segment_spectrogram(S)
        except Exception as e:
            print(f"  [WARN] ({idx}/{total}) {basename} — erro na segmentação: {e}")
            continue

        if len(merged) == 0:
            n_sem_segmento += 1
            arquivos_sem_seg.append(basename)
            print(f"  [SKIP] ({idx}/{total}) {basename} — nenhum segmento detectado")
            continue

        out_path = os.path.join(especie_output_dir, basename)
        np.save(out_path, Espec_seg)
        n_arquivos_gerados += 1
        n_processados += 1
        print(f"  [ OK ] ({idx}/{total}) {basename} | shape {Espec_seg.shape}")

    n_total_processados  += n_processados
    n_total_sem_segmento += n_sem_segmento
    n_total_gerados      += n_arquivos_gerados

    print(f"  [SUB-FIM] OK: {n_processados} | Sem segmento: {n_sem_segmento} | Gerados: {n_arquivos_gerados}")
    if arquivos_sem_seg:
        print(f"  [SUB-FIM] Sem segmento: {', '.join(arquivos_sem_seg)}")

print("\n" + "=" * 60)
print(f"[FIM GERAL] Arquivos processados com segmento : {n_total_processados}")
print(f"[FIM GERAL] Arquivos sem segmento detectado   : {n_total_sem_segmento}")
print(f"[FIM GERAL] Total de arquivos gerados         : {n_total_gerados}")

