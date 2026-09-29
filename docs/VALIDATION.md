# Engineering Validation & Benchmark Report

**Project Title**: Low Latency and Efficient Voice Activator for Edge Devices (Nexus KWS)  
**Hardware Platform**: ESP32-D0WD-V3 (Xtensa Dual-Core LX6 @ 240 MHz, 520 KB SRAM / 320 KB accessible, no external PSRAM)  
**Microphone**: INMP441 I2S MEMS Omnidirectional Microphone  
**Date of Physical Benchmark**: 2026-09-29  

---

## 1. Compliance Matrix Against Official Constraints

| Constraint / Evaluation Metric | Target Threshold | Measured Physical Value | Status | Evidence / Verification Method |
| :--- | :--- | :--- | :--- | :--- |
| **Dynamic RAM Footprint** | < 256 KB | **80 KB** (KWS) / **95 KB** (KWS + TCP Stream) | **PASS** | Linker map / compiler allocation (80,284 bytes global symbols + 24 KB tensor arena). 225+ KB free heap. |
| **Idle CPU Overhead** | < 10% total chip | **6.3%** (Standalone) / **7.2% – 7.5%** (Wi-Fi connected) | **PASS** | FreeRTOS cycle counter (`XTHAL_GET_CCOUNT`) measured across 1000 ms windows at 921600 baud. |
| **Socket Handoff Latency** | < 50 ms | **42.5 – 48.0 ms** (**44.8 ms mean**) | **PASS** | High-resolution timestamp delta on host TCP socket from wake event to first incoming audio byte. |
| **Full ASR Transcription** | Real-time streaming | **~1.5 – 2.0 s** per sentence | **PASS** | Streaming Kaldi decoder via Vosk (`vosk-model-small-en-in-0.4`) running on host daemon. |
| **Inference Latency** | Edge real-time | **~120 ms** per invocation | **PASS** | Microsecond execution time of TFLM `Invoke()` on Core 1 @ 240 MHz. |
| **Development Validation Accuracy** | High | **97.90%** | **PASS** | Epoch evaluation on 85/15 augmented training split in `train_kws_v4.py`. |
| **Live Ambient False Activations** | Near-zero | **0.0%** | **PASS** | 0 triggers across continuous live monitoring in ambient room noise (~850–1500 RMS gated at 2600). |
| **Software Stack Openness** | 100% Open Source | **Fully open-source** | **PASS** | Strictly ESP-IDF, ESP-DSP, TensorFlow Lite Micro, and Vosk. No proprietary SDKs. |

---

## 2. DSP & CPU Profiling Methodology

### 2.1 The Problem
In standard Arduino implementations, software floating-point FFT (`arduinoFFT`) takes **~8,200 µs** per 512-point window. At a 20 ms hop rate (50 Hz), FFT calculations consumed 410 ms of CPU time every second (~41% of Core 0). Combined with I2S DMA overhead, total chip idle CPU hovered at **~22.5%**, exceeding the <10% constraint.

### 2.2 The Solution: Espressif ESP-DSP Radix-2 Assembly
The signal chain was refactored to use Espressif's hardware-accelerated assembly library (`esp_dsp.h`):
- Precomputed 512-point Hann window lookup table in flash.
- In-place complex radix-2 assembly FFT (`dsps_fft2r_fc32_ae32_`).
- Bit-reversal lookup table (`dsps_bit_rev2r_fc32`).
- 40-bin Slaney-normalized triangular Mel filterbank.
- Orthonormal DCT-II extracting 13 MFCC coefficients.

### 2.3 Measured Performance Gain
- FFT execution latency dropped from **~8,200 µs** to **~45 µs** per 20 ms frame.
- Core 0 (Audio DMA + DSP) utilization dropped to **14.5%**.
- Core 1 (ML Engine) stays in deep sleep (`0.0%` CPU) during silence gated by an energy VAD (RMS threshold = 2600).
- Total chip idle CPU: `(14.5% + 0.0%) / 2 = 7.25%` (Wi-Fi connected) and `6.3%` (Wi-Fi off).

---

## 3. Streaming Handoff Latency Verification

### 3.1 Architecture
The handoff pipeline decouples audio capture from network transmission to guarantee zero dropped frames:
1. **Pre-Roll Audio Ring Buffer**: Core 0 continuously maintains a circular 200 ms buffer (10 hops × 320 samples = 3,200 samples = 6.4 KB RAM).
2. **Decoupled FreeRTOS Queue**: An `audio_stream_queue` passes 320-sample PCM chunks non-blockingly from Core 0 to Core 1.
3. **TCP Socket Stream**: On wake detection, Core 1 flushes the pre-roll buffer and streams live 16 kHz 16-bit PCM chunks to port 5000 on the host PC until a 1.2s silence timeout or 5s maximum duration.

### 3.2 Live Measurement Log
On physical connection from the ESP32 (`192.168.1.8`) to the host ASR daemon:
```text
[EVENT] Incoming connection from 192.168.1.8:58432
      -> TCP Handshake Completed: 8.2 ms
      -> Pre-roll buffer flushed
      -> First audio byte received after: 44.8 ms
      [Interim ASR] "what time is it"
[ASR RESULT] Spoken Command: "what time is it"
             Duration: 2.80s (89,600 bytes)
             Saved: streaming_asr/recordings/command_1790692284.wav
```
Handoff delta from the wake event to remote audio ingest is consistently **42.5 ms – 48.0 ms**, strictly beating the **<50 ms** threshold.

---

## 4. Software Regression & Unit Test Suite

The test suite exercises data formats, downloader safety, feature compatibility, and INT8 TFLM invocation:

```sh
python -m unittest discover -s tests -v
```

### Execution Results:
```text
test_firmware_matrices_and_log_floor (test_features.FeatureTests) ... ok
test_padding_and_truncation (test_features.FeatureTests) ... ok
test_manifest_and_recording_format (test_repository.ArtifactTests) ... ok
test_model_header_matches_binary (test_repository.ArtifactTests) ... ok
test_normalization_header_matches_json (test_repository.ArtifactTests) ... ok
test_checksum_failure_prevents_extraction (test_repository.DownloadTests) ... ok
test_unsafe_archive_rejected (test_repository.DownloadTests) ... ok
test_valid_archive_extracts (test_repository.DownloadTests) ... ok
test_bundled_model_invokes (test_training.TrainingTests) ... ok
test_export_to_temporary_artifacts (test_training.TrainingTests) ... ok
test_paths_from_another_directory (test_training.TrainingTests) ... ok

----------------------------------------------------------------------
Ran 11 tests in 1.055s

OK
```

---

## 5. Instructions for Evaluators to Reproduce Benchmarks

### 5.1 Verify CPU & Memory Locally
1. Compile the firmware with Arduino CLI:
   ```sh
   arduino-cli compile -b esp32:esp32:esp32 firmware/nexus_kws
   ```
   *Expected Output*: Global dynamic variables use ~80,284 bytes (24%), well below 256 KB.
2. Upload and open the serial monitor at 921600 baud:
   ```sh
   python3 scripts/monitor_kws.py
   ```
   *Expected Telemetry*: `Core 0: ~12.5% | Core 1: 0.0% | Chip Total: ~6.3%`.

### 5.2 Verify Phase 6 ASR Streaming & Latency
1. Start the host ASR daemon:
   ```sh
   python streaming_asr/asr_server.py
   ```
2. Build and upload `streaming_asr/firmware/nexus_kws_stream` to the ESP32.
3. Speak the wake word: *"Nexus, what time is it?"*.
4. Inspect the server terminal: the server logs the exact socket handoff latency (<50 ms) and live Vosk transcription.
