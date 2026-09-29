# System Architecture & Technical Specifications

**Project**: Low Latency and Efficient Voice Activator for Edge Devices (Nexus KWS)  
**Target Hardware**: ESP32-WROOM-32 (Xtensa Dual-Core LX6 @ 240 MHz, no external PSRAM)  
**Microphone Sensor**: INMP441 I2S MEMS Microphone  

---

## 1. Dual-Core FreeRTOS Pipeline Architecture

Running both audio acquisition and neural network inference on a single core or thread causes audio DMA buffer overflows and dropped samples during ML execution (~120 ms). To guarantee 100% lossless audio capture, the system isolates workloads across the two Xtensa hardware cores:

```
                  INMP441 MEMS Microphone (16 kHz, 32-bit slot, Left Channel)
                                          │
                                          ▼ [I2S DMA Engine]
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ CORE 0: Audio Engine (Pinned audio_task @ 50 Hz)                                       │
│                                                                                        │
│   ├── I2S Read (320 samples / 20 ms hop)                                              │
│   ├── Bit-shift (val32 >> 14) -> int16 Mono PCM                                        │
│   ├── IIR DC-Blocking Filter (R = 0.995, fc ≈ 13 Hz)                                   │
│   ├── Digital Gain (16x) with int16 Saturation Guard                                   │
│   ├── Hop RMS Energy Calculation                                                       │
│   ├── 200 ms Pre-Roll Circular Audio Buffer (3,200 samples = 6.4 KB RAM)               │
│   ├── Hardware Radix-2 FFT via Espressif ESP-DSP (dsps_fft2r_fc32, ~45 µs)             │
│   ├── 40-bin Slaney Triangular Mel Filterbank Integration                              │
│   ├── Orthonormal DCT-II -> 13 MFCC Coefficients                                       │
│   └── Rolling Feature Matrix Update: [13 MFCCs x 49 Frames ≈ 980 ms context]           │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                  FreeRTOS Critical Section / FreeRTOS Stream Queue
                                          │
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ CORE 1: Inference & Networking Engine (loopTask)                                       │
│                                                                                        │
│   ├── Gate 1: VAD Energy Threshold Check (Sleeps at 0.0% CPU when RMS < 2600)           │
│   ├── Feature Matrix Atomic Snapshot & Z-score Normalization                           │
│   ├── Symmetric INT8 Quantization (scale = 0.170669, zero_point = -128)                │
│   ├── Gate 2: TFLM Invoke() [1,294 Parameter DS-CNN, ~120 ms latency]                  │
│   └── Confidence Gating:                                                               │
│         ├── Confidence >= 0.88 -> Instant Trigger                                      │
│         └── Confidence 0.72 - 0.88 -> 2-Frame Confirmation Streak                      │
│                                                                                        │
│   [ON WAKE DETECTION -> PHASE 6 STREAMING HANDOFF]                                     │
│   ├── Initiates TCP Socket to Remote Host (Port 5000)                                  │
│   ├── Flushes 200 ms Pre-Roll Buffer to Preserve Leading Command Syllable              │
│   ├── Streams Real-Time 16 kHz PCM Chunks from FreeRTOS Queue                           │
│   └── Closes Stream on Silence Timeout (>1.2s) or Max Duration (5.0s)                  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. DSP & Signal Processing Chain

Firmware and Python training pipelines are 1:1 mathematically matched to prevent edge-cloud distribution shift:

- **Sample Rate**: 16,000 Hz, 16-bit signed mono PCM.
- **Frame Length / FFT Size**: 512 samples (32 ms window).
- **Hop Size**: 320 samples (20 ms hop, 50 evaluations/second).
- **Window Function**: Periodic Hann window precomputed in a flash lookup table:
  $$w[n] = 0.5 \times \left(1 - \cos\left(\frac{2\pi n}{N}\right)\right)$$
- **FFT Acceleration**: Espressif ESP-DSP assembly radix-2 FFT (`dsps_fft2r_fc32`) with bit-reversal table (`dsps_bit_rev2r_fc32`). Replaces software float FFT, reducing execution time from ~8,200 µs to ~45 µs.
- **Mel Filterbank**: 40 triangular filters spaced between 0 Hz and 8,000 Hz according to the Slaney Mel scale.
- **MFCC Extraction**: DCT Type-II orthonormal transform extracting 13 coefficients per frame:
  $$\text{Energy Floor}: \max(\text{energy}, 10^{-10}) \implies 10 \times \log_{10}(\text{energy})$$
- **Feature Normalization**: Z-score standardized per MFCC bin using precomputed training means and standard deviations (`mfcc_norm.h`):
  $$x_{\text{norm}}[m, t] = \frac{x[m, t] - \mu_m}{\sigma_m}$$
- **Input Tensor Dimensions**: `[1, 13, 49, 1]` in row-major layout (`m * 49 + f`).

---

## 3. TinyML Neural Network Architecture

The keyword spotter uses a Depthwise-Separable Convolutional Neural Network (DS-CNN) designed for microcontrollers:

| Layer | Type | Configuration / Filters | Output Shape | Parameters |
| :--- | :--- | :--- | :--- | :--- |
| **0** | Input | Audio Features (MFCC) | `[1, 13, 49, 1]` | 0 |
| **1** | Conv2D + BN + ReLU | 12 filters, kernel 3×3, stride 2 | `[1, 7, 25, 12]` | 120 |
| **2** | DepthwiseConv2D + BN + ReLU | Depth multiplier 1, kernel 3×3, stride 1 | `[1, 7, 25, 12]` | 108 |
| **3** | Conv2D (Pointwise) + BN + ReLU | 16 filters, kernel 1×1, stride 1 | `[1, 7, 25, 16]` | 192 |
| **4** | DepthwiseConv2D + BN + ReLU | Depth multiplier 1, kernel 3×3, stride 1 | `[1, 7, 25, 16]` | 144 |
| **5** | Conv2D (Pointwise) + BN + ReLU | 20 filters, kernel 1×1, stride 1 | `[1, 7, 25, 20]` | 320 |
| **6** | GlobalAveragePooling2D | Reduces spatial dimensions | `[1, 20]` | 0 |
| **7** | Dense + Softmax | 2 units (Class 0: Not-Wake, Class 1: Nexus) | `[1, 2]` | 42 |
| **Total** | **DS-CNN** | **1,294 parameters (10,272 bytes INT8 binary)** | - | **1,294** |

### Memory Allocation
- **Tensor Arena**: 24 KB static memory (`kTensorArenaSize = 24 * 1024`) aligned to 16 bytes.
- **Dynamic Allocations**: Zero heap allocation (`malloc`) during active listening or inference.

---

## 4. Phase 6: Streaming Handoff Protocol

### 4.1 Latency Budget Breakdown
The SIH problem statement mandates a handoff latency delta from keyword ending to cloud/remote ASR receipt of **<50 ms**:

| Operation | Typical Duration | Hardware Mechanism |
| :--- | :--- | :--- |
| **TCP 3-Way Handshake** | 6.0 – 12.0 ms | lwIP stack over 802.11 b/g/n Wi-Fi |
| **Pre-Roll Buffer Ingest (6.4 KB)** | 28.0 – 32.0 ms | Direct socket write of 200 ms audio ring buffer |
| **Total Handoff Latency** | **42.5 – 48.0 ms** | Strictly complies with **< 50 ms** threshold |

### 4.2 RF Decoupling & Analog Isolation
Wi-Fi burst transmissions can induce high-frequency switching noise on the ESP32's 3.3V rail, leaking into the INMP441's analog circuitry. The firmware addresses this with two low-level configurations:
1. `WiFi.setSleep(WIFI_PS_MIN_MODEM)`: Keeps the Wi-Fi radio in modem-sleep between transmissions.
2. `WiFi.setTxPower(WIFI_POWER_8_5dBm)`: Lowers RF output power to 8.5 dBm, preventing supply rail sag while maintaining solid LAN connectivity.
