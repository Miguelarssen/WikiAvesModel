import requests
from bs4 import BeautifulSoup
import inspect
import json
import os   
from pathlib import Path
import pandas as pd
import re
import io
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import soundfile as sf

tamanhoDataset = 200

def selectAves(FamiliaSelect):

    urlInfo="https://www.wikiaves.com.br/especies.php?t=t"

    # Está parte acessa a lista de espécies do wiki aves direto da página

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "pt-BR,pt;q=0.9",
        "Referer": "https://www.wikiaves.com.br/"
    }

    responseInfo = requests.get(urlInfo, headers=headers)
    soup = BeautifulSoup(responseInfo.content, "html.parser")

    soup = soup.find_all("table", class_="wa-table-sp table m-table m-table--head-separator-metal wa-table-hover")[0]
    aves = soup.find("tbody").find_all("script")[1:]

    # A lista das espécies está misturada com código html
    # Este trecho serve para limpar esta lista e transforma-la
    # em um dataframe pandas organizado

    avesDf = []

    # iteramos por item dentro do html
    for script in aves:
        texto = script.get_text(strip=True)
        
        # removemos a função "lsp" do html
        match = re.search(r"lsp\((.*)\);", texto)

        valores = []

        # fazemos separação dos valores por ,
        for v in match.group(1).split(","):
            v.strip().strip("'") 
            valores.append(v)

        # A listagem das espécies no site não lista a família da ave em todos os registros
        # apenas no primeiro registro que identifica aquela família, por isso o replicamos
        # para que no meu dataframe tenhamos a família da ave colocada corretamente nas colunas
        if valores[1] != " ''":
            familiaAtual = valores[1]

        else:
            valores[1] = familiaAtual
        
        avesDf.append({
            "codigo": int(valores[0].strip().strip("'")),
            "familia": valores[1].strip().strip("'") or None,
            "nome_cientifico": valores[2].strip().strip("'"),
            "nome_popular": valores[3].strip().strip("'"),
            "slug": valores[4].strip().strip("'"),
            "fotos": int(valores[5]),
            "sons": int(valores[6]),
        })

    avesDf = pd.DataFrame(avesDf)

    avesSelect = avesDf[
        (avesDf["familia"] == FamiliaSelect) &
        (avesDf["sons"] > tamanhoDataset)
    ][["codigo", "nome_popular"]]

    return avesSelect

def preProcess(audio, sr):
    TARGET_SECONDS = 5
    TARGET_SAMPLES = TARGET_SECONDS * sr

    FRAME_LEN = 2048
    HOP = 512

    # Calcula a média quadrática para detectar evento principal 
    # Esse código desloca uma janela de amostragem pelo áudio 
    # calculando a média de energia, assim podemos saber qual é
    # o trecho com mais atividade

    rms = librosa.feature.rms(
        y=audio,
        frame_length=FRAME_LEN,
        hop_length=HOP
    )[0]

    # Pegamos o trecho com mais atividade do retorno da função 
    # e então aplicamos essa multiplicação para removermos os 20% 
    # menos relevante do audio, assim eliminando uma faixa de ruído
    
    threshold = 0.2 * np.max(rms)
    frames_event = np.where(rms > threshold)[0]

    # Se não detectar evento (silêncio, erro, etc)
    if len(frames_event) == 0:
        return librosa.util.fix_length(audio, size=TARGET_SAMPLES)

    start_sample = librosa.frames_to_samples(frames_event[0], hop_length=HOP)
    end_sample   = librosa.frames_to_samples(frames_event[-1], hop_length=HOP)

    audio_event = audio[start_sample:end_sample]

    # Centraliza o evento na janela fixa
    if len(audio_event) >= TARGET_SAMPLES:
        audio_event = audio_event[:TARGET_SAMPLES]
    else:
        pad_total = TARGET_SAMPLES - len(audio_event)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        audio_event = np.pad(audio_event, (pad_left, pad_right))

    return audio_event

def createDataset(codigoAve, nomeAve, dwldLimit):

    import requests, json, io
    import numpy as np
    import librosa
    from pathlib import Path

    TARGET_SR = 44100  # sample rate fixo

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "pt-BR,pt;q=0.9",
        "Referer": "https://www.wikiaves.com.br/"
    }

    dataset_path = Path("..") / "dataset" / nomeAve
    dataset_path.mkdir(parents=True, exist_ok=True)

    cont = 0
    page = 1

    while cont < dwldLimit:

        urlFiles = (
            "https://www.wikiaves.com.br/getRegistrosJSON.php?"
            f"tm=s&t=s&s={codigoAve}&p={page}"
        )

        responseFiles = requests.get(urlFiles, headers=headers)
        data = json.loads(responseFiles.text)["registros"]["itens"]

        if not data:
            print(f"Sem mais registros na página {page}")
            break

        for registro in data:
            link = data[registro]["link"]

            link = link.replace(".jpg", ".mp3")
            link = link.replace("#", "")

            try:
                responseAudio = requests.get(link, timeout=10)

                audio, sr = librosa.load(
                    io.BytesIO(responseAudio.content),
                    sr=None,
                    mono=True
                )

                # Reamostragem
                if sr != TARGET_SR:
                    audio = librosa.resample(
                        audio,
                        orig_sr=sr,
                        target_sr=TARGET_SR
                    )
                    sr = TARGET_SR

                # Pré-processamento
                audio = preProcess(audio, sr)

            except Exception as e:
                print(f"Erro no arquivo {registro}: {e}")
                continue

            # ===== MEL SPECTROGRAM =====
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=2048,
                hop_length=512,
                n_mels=128,
                fmin=0,
                fmax=sr // 2
            )

            # Converter para dB
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            path = dataset_path / f"{nomeAve}_{cont}.npy"
            np.save(path, mel_spec_db)

            cont += 1

            if cont >= dwldLimit:
                break

        page += 1


#Aqui selecione o nome da família das aves que deseja baixar
avesSelect = selectAves("Tinamidae")

for x in range(0, len(avesSelect)):
    #createDatasetAudio(avesSelect.iloc[x]["codigo"], avesSelect.iloc[x]["nome_popular"], tamanhoDataset)
    createDataset(avesSelect.iloc[x]["codigo"], avesSelect.iloc[x]["nome_popular"], tamanhoDataset)