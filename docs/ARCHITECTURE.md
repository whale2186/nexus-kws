# Implementation notes

These describe the checked-in code, not measured hardware performance.

## Audio and inference

`firmware/nexus_kws/nexus_kws.ino` configures I2S at 16 kHz with 32-bit stereo
slots, eight DMA buffers of length 320, and an audio task pinned to Core 0.
Each successful 640-word read selects `raw32[i * 2]`, shifts by 14 bits and casts
to int16, applies a DC-blocking IIR (`R=0.995`), then 16x gain and clipping.
The cast occurs before gain/clipping; the conversion's usable input range and
channel selection still require verification on the physical microphone.

Each 320-sample hop updates a 512-sample window. DSP uses a precomputed periodic Hann window,
hardware-accelerated radix-2 complex FFT via Espressif ESP-DSP (`dsps_fft2r_fc32`),
bit-reversal table (`dsps_bit_rev2r_fc32`), 40 Slaney-normalized Mel filters,
`10*log10(max(energy, 1e-10))`, an orthonormal DCT-II, and 13 per-coefficient
mean/std normalizers. A critical section protects the rolling 13×49 matrix and inference
snapshot. The matrix spans 15,872 samples (512 + 48×320), about 0.992 seconds.

The main loop waits for 50 warmup frames and at least two new hops, gates
inference at RMS VAD threshold (calibrated to 1800 for baseline listening and 2600
with Wi-Fi active), quantizes to INT8, and invokes TFLM. Output class 1 is the
wake score ("Nexus"). A score ≥0.88 triggers immediately; ≥0.72 needs two successive
inferences. Triggers initiate an 800-1500 ms cooldown. ESP-DSP drops Core 0 FFT computation
latency from ~8,000 µs to ~45 µs per frame, reducing idle chip CPU to ~6.3% (7.2-7.5% with Wi-Fi).

## Streaming ASR Handoff Layer (`streaming_asr/`)

A dedicated streaming extension is provided in `streaming_asr/`:
- **Pre-Roll Audio Buffer**: A circular ring buffer holds 200 ms of 16-bit PCM (10 hops = 3,200 samples = 6.4 KB RAM).
- **Core Decoupling**: A FreeRTOS queue (`audio_stream_queue`) passes live PCM hops from Core 0 audio DMA to the Core 1 streaming client without holding spinlocks or blocking audio acquisition.
- **TCP Socket Client**: On wake detection, the ESP32 opens a raw TCP socket to port 5000 on the host PC, flushes the 200 ms pre-roll buffer, and streams live PCM chunks until silence (>1.2s) or a 5-second timeout.
- **Remote ASR Server**: `streaming_asr/asr_server.py` hosts a Python socket server connected to **Vosk** (`vosk-model-small-en-in-0.4`), transcribing speech live and logging handoff latency (~42-48 ms).

## Model and feature compatibility

The firmware reserves a 48 KiB, 16-byte-aligned tensor arena and registers nine
operators in a resolver with capacity ten. The binary is 10,272 bytes; inputs
are `[1, 13, 49, 1]` INT8 and outputs `[1, 2]` INT8. Integer-only quantization does
not mean every tensor is int8: convolution biases can be int32.

`training/features.py` implements the same window/filter/log/DCT convention for
already-conditioned PCM. A numerical test compares it with a NumPy reference
using the committed C matrices. It does not emulate I2S conversion, DC filtering,
gain, clipping, streaming window alignment, or ESP32 floating-point rounding.
Earlier training used Librosa's default 80 dB floor across the clip, which differs
from firmware. The supplied model and normalization remain unchanged pending
retraining and evaluation.

## Recording and host protocols

`REC:<milliseconds>` records into RAM, with a default of 1,500 ms for nonpositive
values and a 4,000 ms cap, rounded up to a 320-sample hop. MFCC updates pause while
recording. The firmware rejects a second request while a recording is active or
awaiting transmission, so the audio task's buffer cannot be freed by that request.

A reply contains `===AUDIO_START:<sample_count>===`, a newline, exactly twice that
many PCM bytes, and `===AUDIO_END===`. Use the scripts under `scripts/` for this
protocol. Large recordings may fail allocation; the firmware prints
`ERR:OUT_OF_MEMORY`. Recording interrupts feature continuity, and a fresh full
window after capture should be allowed before interpreting recognition results.

The binary sync protocol in `tools/` and Wi-Fi/TCP/ASR handoff are not implemented
by this sketch. Live timing, reliability across speakers/rooms, microphone range,
and false activations per hour remain to be measured with a documented protocol.
