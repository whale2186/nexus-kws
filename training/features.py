"""Offline MFCCs using the firmware's per-frame log-power convention.

Input is already conditioned PCM. This does not emulate I2S, DC blocking,
clipping, or the alignment of a continuously rolling firmware window.
"""
import librosa
import numpy as np


def extract_mfcc(y):
    y = np.asarray(y, dtype=np.float32)[:16000]
    y = np.pad(y, (0, max(0, 16000 - len(y))))
    power = librosa.feature.melspectrogram(
        y=y, sr=16000, n_fft=512, hop_length=320, win_length=512,
        window="hann", n_mels=40, center=False, power=2.0,
        htk=False, norm="slaney",
    )
    # Firmware floors each mel energy at 1e-10, with no clip-wide 80 dB floor.
    log_power = librosa.power_to_db(power, ref=1.0, amin=1e-10, top_db=None)
    return librosa.feature.mfcc(S=log_power, n_mfcc=13, dct_type=2,
                                norm="ortho").astype(np.float32)
