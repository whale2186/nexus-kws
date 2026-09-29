#!/usr/bin/env python3
"""
Nexus KWS v4 Training Pipeline — Real + Synthetic Data

Improvements over v3:
1. Loads real human "Nexus" recordings from dataset/real_positive/ (INMP441 mic captures)
2. Loads Google Speech Commands v2 as real human negative samples
3. Keeps synthetic positives + hard negatives from v3
4. Better class balancing with real data priority
5. Exports model.h, mfcc_norm.h to firmware/nexus_kws/ for direct flashing

Run:
    python3 scripts/download_speech_commands.py   # Once (downloads ~2.3GB)
    python3 scripts/record_wake_word.py           # Record real samples
    python3 training/train_kws_v4.py                       # Train and export

Prerequisites:
    pip install numpy soundfile librosa scipy tensorflow
"""

import os
import glob
import json
import random
from pathlib import Path
from features import extract_mfcc
import numpy as np
import soundfile as sf
import librosa
from scipy import signal
import tensorflow as tf
from tensorflow.keras import layers, models

# --- Audio / Feature Constants (must match firmware) ---
SR = 16000
AUDIO_LEN = 16000       # 1 second
N_FFT = 512
WIN_LENGTH = 512
HOP_LENGTH = 320
N_MELS = 40
NUM_MFCC = 13
NUM_FRAMES = 49

# --- Output Paths ---
ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = ROOT / "training"
FIRMWARE_DIR = ROOT / "firmware/nexus_kws"
SEED = 42

MODEL_TFLITE_PATH = TRAINING_DIR / "model.tflite"
MODEL_H_INO_PATH = FIRMWARE_DIR / "model.h"
NORM_JSON_PATH = TRAINING_DIR / "mfcc_norm_stats.json"
NORM_H_INO_PATH = FIRMWARE_DIR / "mfcc_norm.h"

# --- Dataset Paths ---
REAL_POSITIVE_DIR = ROOT / "dataset/real_positive"
REAL_NEGATIVE_DIR = ROOT / "dataset/real_negative"
SYNTH_POSITIVE_DIR = ROOT / "dataset/positive"
NEGATIVE_DIR = ROOT / "dataset/negative"
UNKNOWN_DIR = ROOT / "dataset/unknown"
NEGATIVES_EXPANDED_DIR = ROOT / "dataset/negatives_expanded"
SPEECH_COMMANDS_DIR = ROOT / "dataset/speech_commands_v2"

# Hard negative keywords (phonetic confusers)
HARD_KEYWORDS = [
    "texas", "lexus", "alexis", "plexus", "census", "next", "necklace",
    "netflix", "nectar", "exit", "access", "excess", "fixus", "mixus",
    "maxus", "taxus", "faxus", "flexes", "boxes", "foxes", "sixes",
    "neck", "exes", "hexes"
]

# Speech Commands words to use as negatives (skip _background_noise_)
SPEECH_COMMANDS_WORDS = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow",
    "backward", "forward", "follow", "learn", "visual"
]


def spec_augment(mfcc, max_time_mask=6, max_freq_mask=2):
    augmented = mfcc.copy()
    if max_freq_mask > 0:
        f = np.random.randint(0, max_freq_mask + 1)
        f0 = np.random.randint(0, NUM_MFCC - f)
        augmented[f0:f0 + f, :] = 0
    if max_time_mask > 0:
        t = np.random.randint(0, max_time_mask + 1)
        t0 = np.random.randint(0, NUM_FRAMES - t)
        augmented[:, t0:t0 + t] = 0
    return augmented


def load_ambient_noises():
    noises = []
    for f in sorted(glob.glob(str(ROOT / "baseline*.wav"))):
        try:
            y, _ = librosa.load(f, sr=SR)
            noises.append(y)
        except Exception:
            pass
    print(f"Loaded {len(noises)} baseline ambient noise tracks.")
    return noises


def add_noise(y, noises, snr_db_range=(10, 30)):
    if not noises or np.random.random() < 0.2:
        return y
    noise = noises[np.random.randint(len(noises))]
    if len(noise) < len(y):
        noise = np.tile(noise, int(np.ceil(len(y) / len(noise))))
    start = np.random.randint(0, len(noise) - len(y) + 1)
    noise_slice = noise[start:start + len(y)]
    rms_y = np.sqrt(np.mean(y**2)) + 1e-8
    rms_noise = np.sqrt(np.mean(noise_slice**2)) + 1e-8
    target_snr = np.random.uniform(snr_db_range[0], snr_db_range[1])
    target_noise_rms = rms_y / (10 ** (target_snr / 20.0))
    scale = target_noise_rms / rms_noise
    return y + noise_slice * scale


def augment_sample(y_trimmed, noises, max_offset=True):
    """Apply random augmentation to a trimmed audio sample."""
    y_aug = y_trimmed.copy()

    # Time stretch
    if np.random.random() < 0.5:
        rate = np.random.uniform(0.85, 1.15)
        y_aug = librosa.effects.time_stretch(y_aug, rate=rate)

    # Pitch shift
    if np.random.random() < 0.4:
        steps = np.random.uniform(-2.0, 2.0)
        y_aug = librosa.effects.pitch_shift(y_aug, sr=SR, n_steps=steps)

    # Trim to window
    if len(y_aug) > AUDIO_LEN:
        y_aug = y_aug[:AUDIO_LEN]

    # Random placement in 1s window
    y_window = np.zeros(AUDIO_LEN, dtype=np.float32)
    if max_offset and len(y_aug) < AUDIO_LEN:
        start = np.random.randint(0, AUDIO_LEN - len(y_aug) + 1)
    else:
        start = 0
    y_window[start:start + len(y_aug)] = y_aug

    # Volume scaling
    y_window *= np.random.uniform(0.3, 1.2)

    # Noise injection
    y_window = add_noise(y_window, noises, snr_db_range=(10, 30))

    return y_window


# =============================================
# POSITIVE SAMPLE LOADERS
# =============================================

def load_real_positives(noises, augments_per_file=60):
    """Load real human recordings of 'Nexus' from INMP441 mic captures."""
    features = []
    real_files = sorted(glob.glob(os.path.join(REAL_POSITIVE_DIR, "*.wav")))
    if not real_files:
        print("No real positive samples found in dataset/real_positive/")
        print("  Run: python3 scripts/record_wake_word.py")
        return features

    print(f"Loading {len(real_files)} REAL positive 'Nexus' files ({augments_per_file} augments each)...")

    for filepath in real_files:
        try:
            y, _ = librosa.load(filepath, sr=SR)
            yt, _ = librosa.effects.trim(y, top_db=22)
            if len(yt) < 2000:
                continue

            # Always include the clean (unaugmented) version
            y_clean = np.zeros(AUDIO_LEN, dtype=np.float32)
            y_clean[:min(len(yt), AUDIO_LEN)] = yt[:min(len(yt), AUDIO_LEN)]
            features.append(extract_mfcc(y_clean))

            # Augmented versions
            for _ in range(augments_per_file):
                y_window = augment_sample(yt, noises)
                mfcc = extract_mfcc(y_window)
                if np.random.random() < 0.2:
                    mfcc = spec_augment(mfcc)
                features.append(mfcc)
        except Exception as e:
            print(f"  WARNING: Failed to load {filepath}: {e}")
            continue

    print(f"  -> Generated {len(features)} real positive samples.")
    return features


def load_synthetic_positives(noises, augments_per_file=30):
    """Load synthetic TTS 'Nexus' samples."""
    features = []
    pos_files = sorted(glob.glob(os.path.join(SYNTH_POSITIVE_DIR, "*.wav")) + glob.glob(os.path.join(SYNTH_POSITIVE_DIR, "*.mp3")))
    if not pos_files:
        print("No synthetic positive samples found in dataset/positive/")
        return features

    print(f"Loading {len(pos_files)} SYNTHETIC positive files ({augments_per_file} augments each)...")

    for filepath in pos_files:
        try:
            y, _ = librosa.load(filepath, sr=SR)
            yt, _ = librosa.effects.trim(y, top_db=22)
            if len(yt) < 2000:
                continue

            for _ in range(augments_per_file):
                y_window = augment_sample(yt, noises)
                mfcc = extract_mfcc(y_window)
                if np.random.random() < 0.2:
                    mfcc = spec_augment(mfcc)
                features.append(mfcc)
        except Exception:
            continue

    print(f"  -> Generated {len(features)} synthetic positive samples.")
    return features


# =============================================
# NEGATIVE SAMPLE LOADERS
# =============================================

def is_hard_negative(filepath):
    fn = os.path.basename(filepath).lower()
    return any(k in fn for k in HARD_KEYWORDS)


def load_tts_negatives(noises, hard_augments=20, general_augments=4):
    """Load synthetic TTS hard negatives and unknown words."""
    features = []
    speech_files = (
        glob.glob(os.path.join(NEGATIVE_DIR, "*.wav")) +
        glob.glob(os.path.join(NEGATIVE_DIR, "*.mp3")) +
        glob.glob(os.path.join(UNKNOWN_DIR, "*.wav")) +
        glob.glob(os.path.join(UNKNOWN_DIR, "*.mp3")) +
        glob.glob(os.path.join(NEGATIVES_EXPANDED_DIR, "*.mp3"))
    )
    hard_count = general_count = 0

    print(f"Loading {len(speech_files)} TTS negative files (Hard: {hard_augments}x, General: {general_augments}x)...")

    for filepath in sorted(speech_files):
        try:
            is_hard = is_hard_negative(filepath)
            n_aug = hard_augments if is_hard else general_augments
            if is_hard:
                hard_count += 1
            else:
                general_count += 1

            y, _ = librosa.load(filepath, sr=SR)
            yt, _ = librosa.effects.trim(y, top_db=22)
            if len(yt) < 2000:
                continue

            for _ in range(n_aug):
                y_aug = yt.copy()
                if is_hard and np.random.random() < 0.4:
                    rate = np.random.uniform(0.9, 1.1)
                    y_aug = librosa.effects.time_stretch(y_aug, rate=rate)

                y_window = np.zeros(AUDIO_LEN, dtype=np.float32)
                if len(y_aug) > AUDIO_LEN:
                    start = np.random.randint(0, len(y_aug) - AUDIO_LEN + 1)
                    y_window = y_aug[start:start + AUDIO_LEN]
                else:
                    shift = np.random.randint(0, AUDIO_LEN - len(y_aug) + 1)
                    y_window[shift:shift + len(y_aug)] = y_aug

                y_window *= np.random.uniform(0.3, 1.2)
                y_window = add_noise(y_window, noises, snr_db_range=(10, 30))
                mfcc = extract_mfcc(y_window)
                if np.random.random() < 0.2:
                    mfcc = spec_augment(mfcc)
                features.append(mfcc)
        except Exception:
            continue

    print(f"  -> {hard_count} hard + {general_count} general TTS files -> {len(features)} samples.")
    return features


def load_speech_commands_negatives(noises, samples_per_word=80):
    """Load real human speech from Google Speech Commands v2 as negatives."""
    features = []

    if not os.path.isdir(SPEECH_COMMANDS_DIR):
        print(f"Speech Commands dataset not found at {SPEECH_COMMANDS_DIR}/")
        print("  Run: python3 scripts/download_speech_commands.py")
        return features

    available_words = [w for w in SPEECH_COMMANDS_WORDS
                       if os.path.isdir(os.path.join(SPEECH_COMMANDS_DIR, w))]

    print(f"Loading Speech Commands v2: {len(available_words)} words, {samples_per_word} samples each...")

    for word in available_words:
        word_dir = os.path.join(SPEECH_COMMANDS_DIR, word)
        wav_files = sorted(glob.glob(os.path.join(word_dir, "*.wav")))
        selected = random.sample(wav_files, min(samples_per_word, len(wav_files)))

        for filepath in selected:
            try:
                y, _ = librosa.load(filepath, sr=SR)
                if len(y) < 2000:
                    continue

                # Place in 1s window
                y_window = np.zeros(AUDIO_LEN, dtype=np.float32)
                if len(y) > AUDIO_LEN:
                    y_window = y[:AUDIO_LEN]
                else:
                    y_window[:len(y)] = y

                y_window *= np.random.uniform(0.5, 1.2)
                if np.random.random() < 0.6:
                    y_window = add_noise(y_window, noises, snr_db_range=(12, 35))

                mfcc = extract_mfcc(y_window)
                features.append(mfcc)
            except Exception:
                continue

    print(f"  -> Generated {len(features)} real human speech negative samples.")
    return features


def load_real_negatives(noises, speech_augments=35, ambient_augments=25):
    """Load real negative samples recorded directly from the INMP441 ESP32 microphone.
    Includes both ambient room noise clips and spoken non-wake words."""
    features = []
    if not os.path.isdir(REAL_NEGATIVE_DIR):
        print(f"No real negative directory found at {REAL_NEGATIVE_DIR}")
        return features

    speech_files = sorted(glob.glob(os.path.join(REAL_NEGATIVE_DIR, "speech_neg_*.wav")))
    ambient_files = sorted(glob.glob(os.path.join(REAL_NEGATIVE_DIR, "ambient_real_*.wav")))
    all_other = sorted(f for f in glob.glob(os.path.join(REAL_NEGATIVE_DIR, "*.wav")) if f not in speech_files and f not in ambient_files)

    print(f"Loading REAL INMP441 negatives: {len(speech_files)} spoken clips ({speech_augments}x) + {len(ambient_files)} ambient clips ({ambient_augments}x)...")

    # Spoken negatives from user
    for filepath in speech_files + all_other:
        try:
            y, _ = librosa.load(filepath, sr=SR)
            yt, _ = librosa.effects.trim(y, top_db=20)
            if len(yt) < 2000:
                continue

            # Clean version
            y_clean = np.zeros(AUDIO_LEN, dtype=np.float32)
            y_clean[:min(len(yt), AUDIO_LEN)] = yt[:min(len(yt), AUDIO_LEN)]
            features.append(extract_mfcc(y_clean))

            for _ in range(speech_augments):
                y_window = augment_sample(yt, noises)
                mfcc = extract_mfcc(y_window)
                if np.random.random() < 0.2:
                    mfcc = spec_augment(mfcc)
                features.append(mfcc)
        except Exception:
            continue

    # Ambient room noise negatives from user's room
    for filepath in ambient_files:
        try:
            y, _ = librosa.load(filepath, sr=SR)
            if len(y) < AUDIO_LEN:
                continue

            for _ in range(ambient_augments):
                start = np.random.randint(0, len(y) - AUDIO_LEN + 1)
                chunk = y[start:start + AUDIO_LEN].copy()
                chunk *= np.random.uniform(0.7, 1.3)
                features.append(extract_mfcc(chunk))
        except Exception:
            continue

    print(f"  -> Generated {len(features)} real INMP441 negative samples.")
    return features


def generate_impulse_negatives(count=800, noises=None):
    """Synthetic claps, snaps, taps, double claps."""
    features = []
    print(f"Generating {count} impulse/transient negatives...")

    for _ in range(count):
        y = np.zeros(AUDIO_LEN, dtype=np.float32)
        transient_type = np.random.choice(["clap", "snap", "tap", "double_clap"])
        start = np.random.randint(int(0.1 * SR), int(0.7 * SR))

        if transient_type in ["clap", "double_clap"]:
            dur = np.random.randint(int(0.015 * SR), int(0.045 * SR))
            t = np.arange(dur) / SR
            decay = np.exp(-t / np.random.uniform(0.005, 0.015))
            noise_burst = np.random.randn(dur).astype(np.float32)
            sos = signal.butter(4, [800, 4500], btype='bandpass', fs=SR, output='sos')
            burst = signal.sosfilt(sos, noise_burst) * decay
            burst /= (np.max(np.abs(burst)) + 1e-6)
            burst *= np.random.uniform(0.3, 1.0)
            y[start:start+dur] = burst
            if transient_type == "double_clap":
                gap = np.random.randint(int(0.07 * SR), int(0.18 * SR))
                if start + dur + gap + dur < AUDIO_LEN:
                    y[start + dur + gap:start + dur + gap + dur] = burst * np.random.uniform(0.7, 1.1)

        elif transient_type == "snap":
            dur = np.random.randint(int(0.005 * SR), int(0.018 * SR))
            t = np.arange(dur) / SR
            decay = np.exp(-t / np.random.uniform(0.002, 0.006))
            noise_burst = np.random.randn(dur).astype(np.float32)
            sos = signal.butter(4, [3000, 7000], btype='bandpass', fs=SR, output='sos')
            burst = signal.sosfilt(sos, noise_burst) * decay
            burst /= (np.max(np.abs(burst)) + 1e-6)
            burst *= np.random.uniform(0.3, 0.9)
            y[start:start+dur] = burst

        elif transient_type == "tap":
            dur = np.random.randint(int(0.025 * SR), int(0.070 * SR))
            t = np.arange(dur) / SR
            decay = np.exp(-t / np.random.uniform(0.010, 0.025))
            f0 = np.random.uniform(150, 400)
            burst = np.sin(2 * np.pi * f0 * t) * decay + 0.3 * np.random.randn(dur) * decay
            burst /= (np.max(np.abs(burst)) + 1e-6)
            burst *= np.random.uniform(0.3, 0.9)
            y[start:start+dur] = burst

        y = add_noise(y, noises, snr_db_range=(15, 35))
        features.append(extract_mfcc(y))

    print(f"  -> Generated {len(features)} impulse negatives.")
    return features


def generate_silence_negatives(count=600, noises=None):
    """Silence, ambient noise, mains hum."""
    features = []
    print(f"Generating {count} silence/ambient negatives...")

    for _ in range(count):
        choice = np.random.random()
        if choice < 0.25:
            y = np.zeros(AUDIO_LEN, dtype=np.float32)
        elif choice < 0.50:
            y = np.random.randn(AUDIO_LEN).astype(np.float32) * np.random.uniform(0.00005, 0.002)
        elif noises and choice < 0.85:
            noise = noises[np.random.randint(len(noises))]
            if len(noise) > AUDIO_LEN:
                start = np.random.randint(0, len(noise) - AUDIO_LEN)
                y = noise[start:start + AUDIO_LEN].copy()
            else:
                y = np.pad(noise, (0, AUDIO_LEN - len(noise)), mode='wrap')
            y *= np.random.uniform(0.2, 1.0)
        else:
            t = np.arange(AUDIO_LEN) / SR
            hum_freq = np.random.choice([50.0, 60.0, 100.0, 120.0])
            y = np.sin(2 * np.pi * hum_freq * t).astype(np.float32) * np.random.uniform(0.001, 0.01)

        features.append(extract_mfcc(y))

    print(f"  -> Generated {len(features)} silence/ambient negatives.")
    return features


# =============================================
# MODEL ARCHITECTURE
# =============================================

def build_model():
    """Fast, low-latency DS-CNN optimized for real-time ESP32 inference (~30ms vs 320ms)."""
    model = models.Sequential([
        layers.Input(shape=(NUM_MFCC, NUM_FRAMES, 1)),

        layers.Conv2D(12, (3, 3), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.DepthwiseConv2D((3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(16, (1, 1), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.DepthwiseConv2D((3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(20, (1, 1), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.25),
        layers.Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# =============================================
# EXPORT
# =============================================

def export_model(model, X_train, mean_flat, std_flat):
    """INT8 quantize and export to .tflite + .h files."""

    print("\n--- Converting to INT8 Quantized TFLite ---")
    def rep_dataset():
        for _ in range(200):
            idx = np.random.randint(len(X_train))
            yield [X_train[idx:idx+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_quant_model = converter.convert()

    with open(MODEL_TFLITE_PATH, "wb") as f:
        f.write(tflite_quant_model)
    print(f"Saved {MODEL_TFLITE_PATH} ({len(tflite_quant_model)} bytes).")

    # C header
    hex_array = ", ".join(f"0x{b:02x}" for b in tflite_quant_model)
    model_h_content = f"""// Auto-generated by train_kws_v4.py
#ifndef MODEL_H
#define MODEL_H

alignas(16) const unsigned char model_tflite[] = {{
  {hex_array}
}};
const unsigned int model_tflite_len = {len(tflite_quant_model)};

#endif // MODEL_H
"""
    for path in [MODEL_H_INO_PATH]:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(model_h_content)
    print(f"Saved {MODEL_H_INO_PATH}.")

    # Normalization header
    norm_h_content = f"""// Auto-generated by train_kws_v4.py
#ifndef MFCC_NORM_H
#define MFCC_NORM_H

const float mfcc_mean[13] = {{{', '.join(f'{x:.4f}f' for x in mean_flat)}}};
const float mfcc_std[13] = {{{', '.join(f'{x:.4f}f' for x in std_flat)}}};

#endif // MFCC_NORM_H
"""
    for path in [NORM_H_INO_PATH]:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(norm_h_content)
    print(f"Saved {NORM_H_INO_PATH}.")

    norm_data = {"mean": mean_flat, "std": std_flat}
    with open(NORM_JSON_PATH, "w") as f:
        json.dump(norm_data, f, indent=2)

    return tflite_quant_model


def verify_model(tflite_model, mean_flat, std_flat, noises):
    """Run edge case tests on quantized model."""
    print("\n--- TFLite Verification on Edge Cases ---")

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    in_scale = inp['quantization_parameters']['scales'][0]
    in_zero = inp['quantization_parameters']['zero_points'][0]
    out_scale = out['quantization_parameters']['scales'][0]
    out_zero = out['quantization_parameters']['zero_points'][0]

    mean_arr = np.array(mean_flat).reshape(13, 1)
    std_arr = np.array(std_flat).reshape(13, 1)

    def test(name, y_audio):
        mfcc = extract_mfcc(y_audio)
        norm = (mfcc - mean_arr) / std_arr
        q_in = np.clip(np.round(norm / in_scale) + in_zero, -128, 127).astype(np.int8)
        q_in = q_in[np.newaxis, ..., np.newaxis]
        interpreter.set_tensor(inp['index'], q_in)
        interpreter.invoke()
        q_out = interpreter.get_tensor(out['index'])
        score = (int(q_out[0][1]) - int(out_zero)) * out_scale
        status = "WAKE" if score > 0.65 else "----"
        print(f"  [{status}] {name:30s} -> P(Nexus)={score:.4f} (raw: {q_out[0]})")
        return score

    test("Zeros (Silence)", np.zeros(16000, dtype=np.float32))
    test("Low Ambient Hiss", np.random.randn(16000).astype(np.float32) * 0.001)

    # Clap
    t_clap = np.linspace(0, 0.2, int(16000 * 0.2))
    clap_wave = np.zeros(16000, dtype=np.float32)
    clap_wave[2000:2000+len(t_clap)] = np.random.randn(len(t_clap)).astype(np.float32) * np.exp(-t_clap * 80)
    test("Loud Clap Transient", clap_wave)

    if noises:
        noise_samp = noises[0][:16000] if len(noises[0]) >= 16000 else np.zeros(16000, dtype=np.float32)
        test("Real Mic Ambient Noise", noise_samp)

    # Test real negative ambient from mic
    real_amb = sorted(glob.glob(os.path.join(REAL_NEGATIVE_DIR, "ambient_real_*.wav")))
    if real_amb:
        y, _ = librosa.load(real_amb[0], sr=SR)
        test("Real INMP441 Ambient Noise", y[:16000])

    # Test real negative speech from user
    real_spk = sorted(glob.glob(os.path.join(REAL_NEGATIVE_DIR, "speech_neg_*.wav")))
    if real_spk:
        y, _ = librosa.load(real_spk[0], sr=SR)
        yt, _ = librosa.effects.trim(y, top_db=20)
        y_win = np.zeros(16000, dtype=np.float32)
        y_win[:min(len(yt), 16000)] = yt[:min(len(yt), 16000)]
        test("Real INMP441 Spoken Neg (User)", y_win)

    # Test real positive if available
    real_files = sorted(glob.glob(os.path.join(REAL_POSITIVE_DIR, "*.wav")))
    if real_files:
        y, _ = librosa.load(real_files[0], sr=SR)
        yt, _ = librosa.effects.trim(y, top_db=22)
        y_win = np.zeros(16000, dtype=np.float32)
        y_win[:min(len(yt), 16000)] = yt[:min(len(yt), 16000)]
        test("Real Positive 'Nexus'", y_win)

    # Test synthetic positive
    pos_files = sorted(glob.glob(os.path.join(SYNTH_POSITIVE_DIR, "*.wav")) + glob.glob(os.path.join(SYNTH_POSITIVE_DIR, "*.mp3")))
    if pos_files:
        y, _ = librosa.load(pos_files[0], sr=SR)
        yt, _ = librosa.effects.trim(y, top_db=22)
        y_win = np.zeros(16000, dtype=np.float32)
        y_win[:min(len(yt), 16000)] = yt[:min(len(yt), 16000)]
        test("Synth Positive 'Nexus'", y_win)

    # Hard negative
    tex_files = sorted(glob.glob(os.path.join(NEGATIVE_DIR, "*Texas*.wav")) + glob.glob(os.path.join(NEGATIVE_DIR, "*Texas*.mp3")))
    if tex_files:
        y, _ = librosa.load(tex_files[0], sr=SR)
        yt, _ = librosa.effects.trim(y, top_db=22)
        y_win = np.zeros(16000, dtype=np.float32)
        y_win[:min(len(yt), 16000)] = yt[:min(len(yt), 16000)]
        test("Hard Neg: Texas", y_win)

    # Speech Commands sample
    for word in ["yes", "no", "stop"]:
        word_dir = os.path.join(SPEECH_COMMANDS_DIR, word)
        if os.path.isdir(word_dir):
            wavs = sorted(glob.glob(os.path.join(word_dir, "*.wav")))
            if wavs:
                y, _ = librosa.load(wavs[0], sr=SR)
                y_win = np.zeros(16000, dtype=np.float32)
                y_win[:min(len(y), 16000)] = y[:min(len(y), 16000)]
                test(f"Speech Cmd: '{word}'", y_win)


# =============================================
# MAIN
# =============================================

def main():
    # Seed before augmentation, representative sampling, and model initialization.
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    print("=" * 60)
    print("Nexus KWS v4 — Real + Synthetic Data Training")
    print("=" * 60)

    noises = load_ambient_noises()

    # --- Positives ---
    f_real_pos = load_real_positives(noises, augments_per_file=60)
    f_synth_pos = load_synthetic_positives(noises, augments_per_file=30)
    f_pos = f_real_pos + f_synth_pos
    l_pos = [1] * len(f_pos)

    if not f_pos:
        print("\nERROR: No positive samples found!")
        print("  Need at least synthetic positives in dataset/positive/")
        print("  Ideally also real recordings in dataset/real_positive/")
        raise SystemExit(1)

    # --- Negatives ---
    f_real_neg = load_real_negatives(noises, speech_augments=35, ambient_augments=25)
    if not os.path.isdir(SPEECH_COMMANDS_DIR):
        print("\nNote: Speech Commands v2 not downloaded yet; balancing with enhanced hard negatives & impulse/silence.")
        f_tts_neg = load_tts_negatives(noises, hard_augments=25, general_augments=8)
        f_speech_cmd = []
        f_impulse = generate_impulse_negatives(count=800, noises=noises)
        f_silence = generate_silence_negatives(count=500, noises=noises)
    else:
        f_tts_neg = load_tts_negatives(noises, hard_augments=20, general_augments=4)
        f_speech_cmd = load_speech_commands_negatives(noises, samples_per_word=80)
        f_impulse = generate_impulse_negatives(count=600, noises=noises)
        f_silence = generate_silence_negatives(count=400, noises=noises)

    f_neg = f_real_neg + f_tts_neg + f_speech_cmd + f_impulse + f_silence
    l_neg = [0] * len(f_neg)

    # --- Assemble dataset ---
    X_all = np.array(f_pos + f_neg)
    y_all = np.array(l_pos + l_neg)

    print("\n--- Dataset Summary ---")
    print(f"Total samples:        {len(X_all)}")
    print(f"Positives (Nexus):    {np.sum(y_all == 1)} ({np.mean(y_all == 1)*100:.1f}%)")
    print(f"  - Real:             {len(f_real_pos)}")
    print(f"  - Synthetic:        {len(f_synth_pos)}")
    print(f"Negatives:            {np.sum(y_all == 0)} ({np.mean(y_all == 0)*100:.1f}%)")
    print(f"  - TTS Hard/General: {len(f_tts_neg)}")
    print(f"  - Speech Commands:  {len(f_speech_cmd)}")
    print(f"  - Impulse:          {len(f_impulse)}")
    print(f"  - Silence/Ambient:  {len(f_silence)}")

    # This is an augmented-clip development split, not an independent test set.
    indices = np.random.permutation(len(X_all))
    split_idx = int(0.85 * len(X_all))
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    X_train, X_val = X_all[train_indices], X_all[val_indices]
    y_train, y_val = y_all[train_indices], y_all[val_indices]

    # Fit normalization on training clips only.
    mean = np.mean(X_train, axis=(0, 2), keepdims=True)
    std = np.std(X_train, axis=(0, 2), keepdims=True) + 1e-6
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    mean_flat = mean.flatten().tolist()
    std_flat = std.flatten().tolist()

    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]

    print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}")

    # --- Train ---
    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
    ]

    print("\n--- Training ---")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=32,
        callbacks=callbacks
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nVal Loss: {val_loss:.4f} | Val Accuracy: {val_acc*100:.2f}%")

    # Confusion matrix
    preds = np.argmax(model.predict(X_val), axis=1)
    tn = np.sum((y_val == 0) & (preds == 0))
    fp = np.sum((y_val == 0) & (preds == 1))
    fn = np.sum((y_val == 1) & (preds == 0))
    tp = np.sum((y_val == 1) & (preds == 1))
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn:4d} FP={fp:4d}")
    print(f"  FN={fn:4d} TP={tp:4d}")
    if tn + fp > 0:
        print(f"False Positive Rate: {fp/(tn+fp)*100:.2f}%")
    if tp + fn > 0:
        print(f"False Negative Rate: {fn/(tp+fn)*100:.2f}%")

    # --- Export ---
    tflite_model = export_model(model, X_train, mean_flat, std_flat)

    # --- Verify ---
    verify_model(tflite_model, mean_flat, std_flat, noises)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"  Model:  {MODEL_TFLITE_PATH} ({os.path.getsize(MODEL_TFLITE_PATH)} bytes)")
    print(f"  Flash:  firmware/nexus_kws/model.h + firmware/nexus_kws/mfcc_norm.h updated")
    print(f"  Next:   arduino-cli compile -b esp32:esp32:esp32 firmware/nexus_kws/")
    print(f"          arduino-cli upload -b esp32:esp32:esp32 -p /dev/ttyUSB0 firmware/nexus_kws/")
    print("=" * 60)


if __name__ == '__main__':
    main()
