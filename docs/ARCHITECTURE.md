# Technical Architecture & Engineering Notes

Technical breakdown of the dual-core pipeline, acoustic matching, and model quantization used in Nexus KWS.

---

## 1. Dual-Core Architecture

During initial testing on the ESP32, detection was inconsistent: speaking "Nexus" often required 3 to 4 attempts before triggering, with perceptible latency.

Hardware cycle profiling identified the bottleneck:

- Audio hop size is 20 ms (320 samples at 16 kHz).
- On-device FFT and Mel filterbank extraction requires ~9 ms.
- Neural network inference with TFLite Micro takes ~120 ms (and earlier architectures exceeded 300 ms).
- Because audio capture and inference were executed sequentially in a single thread, the CPU stopped servicing the I2S DMA FIFO during inference.
- The I2S DMA FIFO holds 8 buffers (160 ms capacity). During the 120–300 ms inference window, the FIFO overflowed and dropped incoming audio frames.

Because the spoken word "Nexus" spans ~450 ms, the microcontroller dropped the middle or tail of the word whenever inference was active.

### Solution

The ESP32 features two physical cores: Core 0 (PRO_CPU) and Core 1 (APP_CPU). Decoupling audio capture from inference resolved the starvation:

```text
               ┌────────────────────────────────────────────────────────┐
               │                        ESP32                           │
               │                                                        │
               │  Core 0 (PRO_CPU)             Core 1 (APP_CPU)         │
               │  ─────────────────            ─────────────────        │
INMP441 Mic ───┼─► I2S DMA Buffer                                       │
(16kHz Mono)   │        │                                               │
               │   audio_task (50Hz)                                    │
               │   - DC IIR filter                                      │
               │   - 16x digital gain                                   │
               │   - 512pt FFT                                          │
               │   - 40 Mel filters                                     │
               │   - 13 MFCCs                                           │
               │        │                                               │
               │   mfcc_matrix [13x49]                                  │
               │        │                                               │
               │        └──── portENTER_CRITICAL ──► main loop()        │
               │              (Thread-safe copy)     - VAD check        │
               │                                     - INT8 Quantize    │
               │                                     - TFLM Invoke()    │
               │                                     - Confidence Logic │
               │                                            │           │
               │                                      WAKE TRIGGER      │
               └────────────────────────────────────────────┼───────────┘
                                                            ▼
                                                    UART / TCP Stream
```

1. **Core 0 runs `audio_task`:** Pinned to Core 0 at high priority. The task blocks on `i2s_read` waiting for DMA buffers, applies a DC-blocking IIR filter, computes the FFT and Mel bands, and writes the resulting 13 MFCCs into a rolling ring buffer. DSP execution requires 9.1 ms per 20 ms hop, leaving Core 0 idle ~55% of the time with zero dropped audio frames.
2. **Core 1 runs `loop()`:** When audio energy exceeds the VAD threshold, Core 1 acquires a thread-safe snapshot of `mfcc_matrix` using a FreeRTOS critical section (`portENTER_CRITICAL`), quantizes features to INT8, and executes inference asynchronously. Audio acquisition on Core 0 continues uninterrupted.

---

## 2. Feature Extraction: C++ to Librosa Parity

A common pitfall in TinyML audio deployments is numerical divergence between Python training features and embedded C++ feature extraction. Discrepancies in Mel filter shapes, DCT implementations, or normalization degrade on-device accuracy.

The embedded pipeline was configured for 1:1 parity with Librosa:

- **Sample Rate:** 16,000 Hz, 16-bit signed PCM mono.
- **Window:** 512-point periodic Hann window (32 ms), hop size of 320 samples (20 ms).
- **Mel Filterbank:** 40 triangular filters spanning 0 Hz to 8,000 Hz using Slaney area normalization (`htk=False, norm='slaney'`). Precomputed filter weights are exported to `mfcc_coeffs.h`.
- **DCT:** 13 MFCCs derived from an orthogonal DCT Type-II matrix, matching `scipy.fftpack.dct(..., norm='ortho')`.
- **Feature Normalization:** Mean and standard deviation vectors calculated over the full training set are stored in `mfcc_norm.h`. Each MFCC coefficient is normalized on-chip before writing to the matrix.
- **Input Tensor:** `[1, 13, 49, 1]` representing 13 MFCC coefficients over 49 temporal hops (~980 ms context).

---

## 3. Training Data & Hardware Noise Floor Bias

Initial models exhibited continuous false activations in quiet environments, scoring 0.99 confidence on background room silence.

Root cause analysis traced this to dataset distribution bias:
- Positive "Nexus" samples were recorded directly from the INMP441 microphone (which has a characteristic ~32 Hz low-frequency floor).
- Negative samples were clean, synthetic TTS audio generated on a PC.
- The classifier learned to detect the acoustic signature and noise profile of the INMP441 rather than the phonetic content of the wake word.

### Dataset Rebalancing

The training pipeline (`train_kws_v4.py`) was restructured to include physical microphone acoustics across both classes:

1. **51 Real Positives:** Spoken takes of "Nexus" recorded through the INMP441 using an in-RAM buffering script (`scripts/record_wake_word.py`) over 921600 baud serial.
2. **40 Real Ambient Negatives:** Room silence, PC fan noise, and background acoustics captured from the physical board (`scripts/record_ambient.py`).
3. **20 Real Spoken Negatives:** Non-wake vocabulary and phonetic confusers ("Texas", "Lexus", numbers, commands) spoken into the INMP441 (`scripts/record_negative.py`).
4. **Google Speech Commands v2:** ~3,000 human voice clips across 35 words ("yes", "no", "stop", "up", "down", etc.) providing broad acoustic and speaker diversity.
5. **Synthetic Augmentations:** Pitch modification (+-2 semitones), time stretching (0.85x to 1.15x), noise injection, and synthetic impulse transients.

Retraining with this distribution reduced ambient and conversational false alarms to 0.0% while retaining >0.99 confidence on vocal takes of "Nexus".

---

## 4. Quantization & Model Profile

The DS-CNN architecture was trained in TensorFlow/Keras and converted to full INT8 via TFLite representative dataset quantization:

- **Model size:** 10,272 bytes on Flash.
- **RAM arena:** 48 KB allocated in SRAM for activation tensors.
- **Inference latency:** ~120 ms on Xtensa LX6 @ 240 MHz.
- **Parameters:** 1,294 parameters across depthwise-separable convolutional layers.
- **Validation accuracy:** 97.90% on held-out test split.
