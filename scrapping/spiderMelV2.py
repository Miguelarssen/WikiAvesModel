import requests
from bs4 import BeautifulSoup
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import re
import io
import librosa
import numpy as np

# ─── CONFIGURAÇÕES ───────────────────────────────────────────────────────────

tamanhoDataset      = 200       # amostras por espécie
familiaSelect       = "Columbidae"

targetSeconds       = 5         # duração da janela de áudio em segundos
targetSr            = 44100     # sample rate alvo (todas amostras reamostradas)

nFft                = 2048
hopLength           = 512
nMels               = 128
fmin                = 200       # Hz — remove ruído ambiental (vento, carro, vozes)
fmax                = 12000     # Hz — cobre toda vocalização de Columbidae/Tinamidae

rmsFrameLen         = 2048
rmsThresholdPctil   = 40        # percentil do RMS para detecção de atividade (0-100)
minDurationSec      = 1.0       # descarta áudios mais curtos que este valor (segundos)
minSnr              = 2.0       # razão max/mean do RMS — descarta gravações muito ruidosas

maxWorkers          = 6         # threads paralelas de download (aumente se a rede permitir)

# Caminho absoluto para a pasta dataset (sempre relativo ao projeto, não ao CWD)
datasetDir = Path(__file__).resolve().parent.parent / "dataset"

# ─────────────────────────────────────────────────────────────────────────────

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "pt-BR,pt;q=0.9",
    "Referer": "https://www.wikiaves.com.br/"
}


def selectAves(familiaSelect):

    urlInfo = "https://www.wikiaves.com.br/especies.php?t=t"

    responseInfo = requests.get(urlInfo, headers=headers)
    soup = BeautifulSoup(responseInfo.content, "html.parser")

    soup = soup.find_all("table", class_="wa-table-sp table m-table m-table--head-separator-metal wa-table-hover")[0]
    aves = soup.find("tbody").find_all("script")[1:]

    avesDf = []
    familiaAtual = None

    for script in aves:
        texto = script.get_text(strip=True)
        match = re.search(r"lsp\((.*)\);", texto)
        if not match:
            continue

        valores = [v for v in match.group(1).split(",")]

        if valores[1] != " ''":
            familiaAtual = valores[1]
        else:
            valores[1] = familiaAtual

        avesDf.append({
            "codigo":       int(valores[0].strip().strip("'")),
            "familia":      valores[1].strip().strip("'") or None,
            "nome_cientifico": valores[2].strip().strip("'"),
            "nome_popular": valores[3].strip().strip("'"),
            "slug":         valores[4].strip().strip("'"),
            "fotos":        int(valores[5]),
            "sons":         int(valores[6]),
        })

    avesDf = pd.DataFrame(avesDf)

    avesSelect = avesDf[
        (avesDf["familia"] == familiaSelect) &
        (avesDf["sons"] > tamanhoDataset)
    ][["codigo", "nome_popular"]]

    return avesSelect


def preProcess(audio, sr):
    targetSamples = targetSeconds * sr

    rms = librosa.feature.rms(y=audio, frame_length=rmsFrameLen, hop_length=hopLength)[0]

    # Threshold adaptativo por percentil (menos agressivo que o fixo de 20%)
    threshold = np.percentile(rms, rmsThresholdPctil)
    framesEvent = np.where(rms > threshold)[0]

    if len(framesEvent) == 0:
        return librosa.util.fix_length(audio, size=targetSamples)

    startSample = librosa.frames_to_samples(framesEvent[0],  hop_length=hopLength)
    endSample   = librosa.frames_to_samples(framesEvent[-1], hop_length=hopLength)
    audioEvent  = audio[startSample:endSample]

    if len(audioEvent) >= targetSamples:
        # Seleciona a janela de 5s com maior energia (não apenas o início)
        step = hopLength
        nWindows = max(1, (len(audioEvent) - targetSamples) // step)
        bestStart  = 0
        bestEnergy = -1.0
        for i in range(nWindows + 1):
            s = i * step
            if s + targetSamples > len(audioEvent):
                break
            e = float(np.mean(librosa.feature.rms(y=audioEvent[s:s + targetSamples],
                                                   frame_length=rmsFrameLen,
                                                   hop_length=hopLength)[0]))
            if e > bestEnergy:
                bestEnergy = e
                bestStart  = s
        audioEvent = audioEvent[bestStart:bestStart + targetSamples]
    else:
        padTotal = targetSamples - len(audioEvent)
        padLeft  = padTotal // 2
        padRight = padTotal - padLeft
        audioEvent = np.pad(audioEvent, (padLeft, padRight))

    return audioEvent


def _downloadAndProcess(registro, link, nomeAve, idx):
    """Baixa, filtra e converte um único áudio. Retorna (array mel_db, idx) ou None."""
    link = link.replace(".jpg", ".mp3").replace("#", "")

    try:
        resp = requests.get(link, headers=headers, timeout=15)
        resp.raise_for_status()

        audio, sr = librosa.load(io.BytesIO(resp.content), sr=None, mono=True)

        if sr != targetSr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=targetSr)
            sr = targetSr

        # Filtro 1: duração mínima
        if len(audio) / sr < minDurationSec:
            return None, f"[SKIP] {registro}: duração muito curta"

        # Filtro 2: SNR proxy
        rms = librosa.feature.rms(y=audio, frame_length=rmsFrameLen, hop_length=hopLength)[0]
        snr = float(np.max(rms)) / (float(np.mean(rms)) + 1e-8)
        if snr < minSnr:
            return None, f"[SKIP] {registro}: SNR baixo ({snr:.2f})"

        audio = preProcess(audio, sr)

        melSpec = librosa.feature.melspectrogram(
            y=audio, sr=sr,
            n_fft=nFft, hop_length=hopLength,
            n_mels=nMels, fmin=fmin, fmax=fmax
        )
        melSpecDb = librosa.power_to_db(melSpec, ref=np.max)

        return melSpecDb, None

    except Exception as e:
        return None, f"[ERRO] {registro}: {e}"


def createDataset(codigoAve, nomeAve, dwldLimit):

    savePath = datasetDir / nomeAve
    savePath.mkdir(parents=True, exist_ok=True)

    cont    = 0
    page    = 1

    while cont < dwldLimit:

        # Sem &o=mp: sem ordenação por popularidade → maior diversidade geográfica/temporal
        urlFiles = (
            "https://www.wikiaves.com.br/getRegistrosJSON.php?"
            f"tm=s&t=s&s={codigoAve}&p={page}"
        )

        try:
            responseFiles = requests.get(urlFiles, headers=headers, timeout=15)
            data = json.loads(responseFiles.text)["registros"]["itens"]
        except Exception as e:
            print(f"Sem mais registros na página {page}: {e}")
            break

        if not data:
            print(f"Sem mais registros na página {page}")
            break

        # Embaralha para não coletar sempre na mesma ordem dentro da página
        registros = list(data.items())
        random.shuffle(registros)

        # Download paralelo dos áudios da página atual
        futures = {}
        with ThreadPoolExecutor(max_workers=maxWorkers) as executor:
            for registro, item in registros:
                if cont + len(futures) >= dwldLimit:
                    break
                fut = executor.submit(_downloadAndProcess, registro, item["link"], nomeAve, cont)
                futures[fut] = registro

            for fut in as_completed(futures):
                if cont >= dwldLimit:
                    break
                melSpecDb, msg = fut.result()
                if msg:
                    print(f"  {msg}")
                    continue
                path = savePath / f"{nomeAve}_{cont}.npy"
                np.save(path, melSpecDb)
                cont += 1
                print(f"  [{cont}/{dwldLimit}] {nomeAve} salvo: {path.name}")

        page += 1

    print(f"\n[OK] {nomeAve}: {cont}/{dwldLimit} amostras salvas em {savePath}\n")


# ─────────────────────────────────────────────────────────────────────────────

avesSelect = selectAves(familiaSelect)

for x in range(0, len(avesSelect)):
    createDataset(avesSelect.iloc[x]["codigo"], avesSelect.iloc[x]["nome_popular"], tamanhoDataset)
