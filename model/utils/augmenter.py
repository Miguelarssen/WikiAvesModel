import numpy as np
import random


# ──────────────────────────────────────────────────────────────────────────────
# Primitivos de augmentation
# ──────────────────────────────────────────────────────────────────────────────

def add_noise(spectrogram, noise_factor=0.005):
    """Adiciona ruído gaussiano ao espectrograma."""
    noise = np.random.randn(*spectrogram.shape)
    return spectrogram + noise_factor * noise


def shift_time(spectrogram, max_shift=20):
    """Desloca o espectrograma no eixo do tempo com zero-padding (não circular)."""
    shift = random.randint(-max_shift, max_shift)
    result = np.zeros_like(spectrogram)
    if shift > 0:
        result[:, shift:] = spectrogram[:, :-shift]
    elif shift < 0:
        result[:, :shift] = spectrogram[:, -shift:]
    else:
        result = spectrogram.copy()
    return result


def frequency_mask(spectrogram, num_masks=2, max_mask_width=15):
    """Aplica máscaras de frequência (SpecAugment)."""
    augmented = spectrogram.copy()
    num_freq_bins = augmented.shape[0]
    for _ in range(num_masks):
        mask_width = random.randint(1, max_mask_width)
        f0 = random.randint(0, max(0, num_freq_bins - mask_width - 1))
        augmented[f0:f0 + mask_width, :] = 0
    return augmented


def time_mask(spectrogram, num_masks=2, max_mask_width=20):
    """Aplica máscaras de tempo (SpecAugment)."""
    augmented = spectrogram.copy()
    num_time_steps = augmented.shape[1]
    for _ in range(num_masks):
        mask_width = random.randint(1, max_mask_width)
        t0 = random.randint(0, max(0, num_time_steps - mask_width - 1))
        augmented[:, t0:t0 + mask_width] = 0
    return augmented


def gain_adjustment(spectrogram, min_gain=0.7, max_gain=1.3):
    """Ajusta o ganho (volume) do espectrograma."""
    gain = random.uniform(min_gain, max_gain)
    return spectrogram * gain


def pitch_shift(spectrogram, max_shift=8):
    """
    Simula um pitch shift deslocando as linhas de frequência verticalmente.
    Usa zero-padding (não circular) para evitar artefatos físicos impossíveis.
    """
    shift = random.randint(-max_shift, max_shift)
    result = np.zeros_like(spectrogram)
    if shift > 0:
        result[shift:, :] = spectrogram[:-shift, :]
    elif shift < 0:
        result[:shift, :] = spectrogram[-shift:, :]
    else:
        result = spectrogram.copy()
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline composto (Tópico 1 da sugestão de melhoria)
# ──────────────────────────────────────────────────────────────────────────────

def apply_augmentation_pipeline(spectrogram, p=0.5):
    """
    Aplica MÚLTIPLAS técnicas de augmentation em sequência, cada uma com
    probabilidade independente `p`. Isso gera amostras sintéticas muito mais
    diversas do que a abordagem anterior (1 técnica aleatória por amostra).

    Args:
        spectrogram: np.ndarray com shape (freq_bins, time_steps)
        p: probabilidade de cada técnica ser aplicada (padrão: 0.5)

    Returns:
        Espectrograma augmentado como np.ndarray.
    """
    if random.random() < p:
        spectrogram = add_noise(spectrogram, noise_factor=0.005)
    if random.random() < p:
        spectrogram = shift_time(spectrogram, max_shift=20)
    if random.random() < p:
        spectrogram = frequency_mask(spectrogram, num_masks=2, max_mask_width=15)
    if random.random() < p:
        spectrogram = time_mask(spectrogram, num_masks=2, max_mask_width=20)
    if random.random() < p:
        spectrogram = gain_adjustment(spectrogram)
    if random.random() < p:
        spectrogram = pitch_shift(spectrogram, max_shift=8)
    return spectrogram


# ──────────────────────────────────────────────────────────────────────────────
# Mantido para retrocompatibilidade (agora redireciona para o pipeline)
# ──────────────────────────────────────────────────────────────────────────────

def apply_random_augmentation(spectrogram):
    """
    [DEPRECATED] Mantido para compatibilidade.
    Redireciona para apply_augmentation_pipeline.
    """
    return apply_augmentation_pipeline(spectrogram)
