# Nexus KWS: Edge Voice Activator on ESP32

An edge keyword spotting system for the custom wake word **"Nexus"**, running locally on an ESP32 with an INMP441 MEMS microphone. Built with an open-source pipeline: TensorFlow Lite for Microcontrollers, FreeRTOS, and Python Librosa.


## Current Status

The wake word engine is functional and verified on physical hardware.

### Working and Verified

- **Dual-core audio pipeline:** Running FFT and inference on a single thread caused the I2S DMA buffer to overflow, dropping up to 80% of audio samples. The workload is split across both Xtensa cores: Core 0 handles audio capture and 13-bin MFCC extraction at 50 Hz, while Core 1 runs inference.

- **Inference on physical silicon:** A trimmed DS-CNN architecture (1,294 parameters, 10.2 KB INT8 binary) cuts inference latency from ~320 ms down to ~120 ms on an ESP32 at 240 MHz.

- **Single-utterance detection:** "Nexus" activates on the first attempt at normal speaking volume across the room.

- **Rejection of background noise and confusers:** Initial models trained only on synthetic audio mistook the microphone's hardware noise floor for the keyword. The dataset incorporates 40 ambient room noise recordings, 20 spoken confusers through the INMP441, and ~3,000 real speech clips from Google Speech Commands v2. False alarms sit at 0.0% during ambient room noise and non-wake speech.

### In Progress

- **Phase 6 ASR audio handoff:** When "Nexus" triggers, the ESP32 will open a TCP socket over Wi-Fi and stream subsequent 16 kHz PCM audio to a local server running Vosk or Whisper.

- **Idle power optimization:** Gating the FFT behind an energy VAD to let Core 0 sleep during silence, bringing idle listening below 10% CPU.


## Technical Stack

- **Microcontroller:** ESP32 Dev Module (WROOM-32, dual-core Xtensa LX6 @ 240 MHz, no PSRAM).

- **Audio hardware:** INMP441 omnidirectional MEMS microphone over I2S DMA (16 kHz, 16-bit mono).

- **Edge ML runtime:** TensorFlow Lite for Microcontrollers (`TensorFlowLite_ESP32`), using `MicroMutableOpResolver` with 9 registered ops.

- **On-device DSP:** `arduinoFFT` with a 512-point periodic Hann window, 40-bin triangular Slaney Mel filterbank, and orthogonal DCT Type-II yielding 13 MFCCs.

- **Model training:** Python 3.12, TensorFlow 2.18, and Keras. Feature extraction uses Librosa and SciPy.

- **Synthetic data generation:** Microsoft Edge TTS for synthetic base samples, augmented with Librosa pitch shifting and time stretching.

- **Host tooling:** Python `pyserial` for 921600 baud telemetry and dataset capture.


## Model Size and Footprint

- **TFLite binary (`training/model.tflite`):** 10,272 bytes (10.2 KB).

- **C header array (`firmware/nexus_kws/model.h`):** 61.8 KB source text.

- **Parameters:** 1,294 total parameters across depthwise-separable convolutional layers.

- **Quantization:** Full INT8 (inputs, weights, biases, and outputs quantized to `int8_t`).

- **SRAM usage on ESP32:** ~80 KB total dynamic allocation (including 48 KB Tensor Arena, I2S DMA buffers, and rolling MFCC frames), leaving ~247 KB of heap free.

- **Flash usage on ESP32:** 452 KB total sketch size (34% of the 1.3 MB application partition).


## Dataset Layout

The dataset combines recordings from the physical INMP441 microphone with synthetic speech and public benchmarks. Hardware recordings are tracked in the repository for reproducibility:

```
dataset/
├── real_positive/                 # 51 vocal takes of "Nexus" recorded via INMP441
│   ├── nexus_real_000.wav         # (16 kHz, 16-bit mono PCM, ~2.4 MB total)
│   └── ...
├── real_negative/                 # 60 microphone baseline recordings (~2.9 MB total)
│   ├── ambient_real_*.wav         # 40 ambient room and fan noise recordings
│   └── speech_neg_*.wav           # 20 spoken non-wake words ("Texas", "Lexus", numbers, commands)
├── positive/                      # 54 synthetic "Nexus" takes (generated via edge-tts)
├── negative/                      # Phonetic confusers ("Alexis", "Plexus", "Access", "Extra")
└── speech_commands_v2/            # ~3,000 human speech clips across 35 vocabulary words
```

*Note on Speech Commands:* Google Speech Commands v2 (~2.4 GB) is excluded from Git via `.gitignore`. Download it locally with:

```
python3 scripts/download_speech_commands.py
```


## Hardware Benchmarks

Measured on the physical breadboard prototype:

| Metric | Target | Measured Result |
| - | - | - |
| **Model Size (Flash)** | Lightweight | **10.2 KB** (`model.tflite` INT8) |
| **RAM Footprint** | \< 256 KB | **~80 KB total** (48 KB Tensor Arena + buffers), 247 KB heap free |
| **Inference Latency** | Low latency | **~120 ms** per pass on ESP32 @ 240 MHz |
| **Validation Accuracy** | High | **97.90%** on held-out test split |
| **True Positive Rate** | High | **99.6%** confidence on real vocal "Nexus" takes |
| **False Activation Rate** | Near-zero | **0.0%** on Speech Commands v2, ambient noise, and claps |
| **Audio Capture** | Lossless | **50 Hz continuous capture** on Core 0 with zero DMA overruns |



## Pipeline Architecture

```
INMP441 Mic (16kHz Mono over I2S DMA)
      │
[Core 0 - Audio Task] (Runs at 50 Hz, lossless)
  ├── DC-blocking IIR filter (R = 0.995)
  ├── 16x digital gain with clipping guard
  ├── 512-point Hann window + FFT via arduinoFFT
  ├── 40-bin Slaney Mel filterbank
  ├── Ortho DCT-II (13 MFCCs)
  └── Rolling matrix update [13 x 49 frames = ~1.0s context]
      │
      └── portENTER_CRITICAL (Thread-safe snapshot copy)
            │
[Core 1 - Main Loop] (Runs every ~120ms during speech)
  ├── Energy gate check (skips inference during silence)
  ├── INT8 tensor quantization
  ├── TFLM Invoke() [DS-CNN, 1,294 parameters]
  └── Confidence Evaluator:
        ├── Confidence >= 0.80 -> Immediate wake event
        └── 0.65 to 0.80 -> 2-frame confirmation streak
```


## Hardware Setup

- **MCU:** ESP32 Dev Module (WROOM-32, 240 MHz, no PSRAM)

- **Microphone:** INMP441 I2S MEMS microphone module

- **Wiring pinout:**

| INMP441 Pin | ESP32 Pin | Function |
| - | - | - |
| **VDD** | 3V3 | 3.3V power |
| **GND** | GND | Ground |
| **SD** | GPIO 32 | I2S Serial Data |
| **SCK** | GPIO 18 | I2S Bit Clock (BCLK) |
| **WS** | GPIO 19 | I2S Word Select (LRCLK) |
| **L/R** | GND | Left channel select |


*Power decoupling note:* The ESP32's Wi-Fi radio introduces noise on the shared 3.3V rail. Adding a 100nF ceramic capacitor across the INMP441 VDD and GND pins stabilizes audio readings.


## Repository Structure

```
├── docs/
│   ├── images/
│   │   └── esp32_breadboard_prototype.jpg # Breadboard prototype photo
│   └── ARCHITECTURE.md                    # Dual-core pipeline & DSP details
├── firmware/
│   └── nexus_kws/
│       ├── nexus_kws.ino                  # Dual-core FreeRTOS firmware
│       ├── model.h                        # Quantized INT8 model byte array (61.8 KB)
│       ├── mfcc_coeffs.h                  # Precomputed Mel & DCT matrices
│       └── mfcc_norm.h                    # Per-bin normalizer stats
├── dataset/
│   ├── real_positive/                     # 51 vocal takes of "Nexus" from INMP441
│   └── real_negative/                     # 40 ambient + 20 spoken non-wake takes
├── training/
│   ├── train_kws_v4.py                    # Training & INT8 quantization script
│   ├── export_mfcc_matrices.py            # Generates C header filterbanks
│   ├── generate_dataset.py                # Edge-TTS synthetic audio generator
│   ├── generate_negatives.py              # Phonetic confuser generator
│   ├── requirements.txt                   # Python dependencies
│   ├── model.tflite                       # 10.2 KB quantized model binary
│   └── mfcc_norm_stats.json               # Normalization statistics
├── scripts/
│   ├── record_wake_word.py                # In-RAM UART positive take recorder
│   ├── record_ambient.py                  # Automated ambient room noise collector
│   ├── record_negative.py                 # Spoken negative word recorder
│   ├── download_speech_commands.py        # Speech Commands v2 downloader
│   ├── monitor_kws.py                     # Real-time color serial telemetry viewer
│   └── verify_live.py                     # Live verification test script
└── tools/
    ├── server.py                          # Serial audio test server
    ├── index.html                         # Web visualizer for waveform & spectrum
    └── record_serial.py                   # Raw PCM serial dumper
```


## Quickstart

### 1. Python environment

```
cd training/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Flash firmware to ESP32

```
cd firmware/nexus_kws
arduino-cli compile -b esp32:esp32:esp32 .
arduino-cli upload -b esp32:esp32:esp32 -p /dev/ttyUSB0 .
```

### 3. Live serial monitor

```
python3 scripts/monitor_kws.py
```


