import numpy as np
import cv2
import librosa

def to_uint8(arr):
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)

def segment_spectrogram(S,
                        margin=3,
                        morph_close=(20, 3),
                        morph_dilate=(5, 3),
                        min_area=100,
                        min_width=10,
                        top_percentile=99):

    imagem_original = to_uint8(S)

    #HPSS — separa harmônicos (canto) do percussivo (ruído)
    Espec_power = librosa.db_to_power(S)
    Espec_harmonic, _ = librosa.decompose.hpss(Espec_power, margin=margin)
    Espec_harm_db = librosa.power_to_db(Espec_harmonic, ref=Espec_power.max())

    imagem_limpa = to_uint8(Espec_harm_db)

    #Limiar por percentil — seleciona o top (100 - top_percentile)% de pixels mais intensos
    thresh_val = np.percentile(imagem_limpa, top_percentile)
    binary = (imagem_limpa >= thresh_val).astype(np.uint8) * 255

    #Morfologia: kernel horizontal para unir linhas do canto
    kc = cv2.getStructuringElement(cv2.MORPH_RECT, morph_close)
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kc)
    
    kd = cv2.getStructuringElement(cv2.MORPH_RECT, morph_dilate)
    mask = cv2.dilate(mask, kd, iterations=1)

    #Componentes conectados → bounding boxes
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    segs = []
    for lbl in range(1, n_labels):
        x    = stats[lbl, cv2.CC_STAT_LEFT]
        w    = stats[lbl, cv2.CC_STAT_WIDTH]
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= min_area and w >= min_width:
            segs.append((x, x + w))

    segs.sort()
    merged = []
    for x1, x2 in segs:
        if merged and x1 <= merged[-1][1] + 20:
            merged[-1] = (merged[-1][0], max(merged[-1][1], x2))
        else:
            merged.append((x1, x2))

    mask_float = (mask / 255.0).astype(np.float32)
    Espec_seg = np.where(mask_float > 0.5, Espec_harm_db, S.min())

    return imagem_original, imagem_limpa, binary, mask, merged, Espec_seg


def spec_to_audio(Espec_db, sr=22050, n_fft=2048, hop_length=512):
    Espec_power = librosa.db_to_power(Espec_db)
    Espec_linear = librosa.feature.inverse.mel_to_stft(Espec_power, sr=sr, n_fft=n_fft)
    return librosa.griffinlim(Espec_linear, n_iter=32, hop_length=hop_length, win_length=n_fft)
