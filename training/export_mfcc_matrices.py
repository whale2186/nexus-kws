import librosa
import numpy as np
import scipy.fftpack

SR = 16000
N_FFT = 512
N_MELS = 40
N_MFCC = 13

# Get Librosa Mel filterbank (Slaney, area-normalized)
mel_basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)

# Get orthogonal DCT-II matrix
dct_basis = scipy.fftpack.dct(np.eye(N_MELS), axis=0, type=2, norm='ortho')[:N_MFCC]


def format_1d(arr):
    return "{" + ", ".join([f"{x}f" for x in arr]) + "}"
    
def format_2d(arr):
    return "{" + ",\n ".join([format_1d(row) for row in arr]) + "}"

header = f"""
#ifndef MFCC_COEFFS_H
#define MFCC_COEFFS_H

#define N_FFT {N_FFT}
#define N_FFT_BINS {(N_FFT//2) + 1}
#define N_MELS {N_MELS}
#define N_MFCC {N_MFCC}

// Shape: [N_MELS][N_FFT_BINS]
const float mel_basis[N_MELS][N_FFT_BINS] = {format_2d(mel_basis)};

// Shape: [N_MFCC][N_MELS]
const float dct_basis[N_MFCC][N_MELS] = {format_2d(dct_basis)};

#endif
"""

with open("mfcc_coeffs.h", "w") as f:
    f.write(header)
