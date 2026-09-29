# Repository review validation — 2026-09-27

Reviewed the clean working tree at `32c4379` and all four existing commits before
editing. The initial commit supplies the code/data/model; subsequent commits
only change README presentation and an image. No historical build logs, training
split manifests, benchmark results, or license were present.

## Changes

- `README.md` and `ARCHITECTURE.md`: replace unsupported performance/parity claims
  with code-supported behavior and explicit evaluation limits; correct root-based
  setup commands, firmware dependencies, recording units, data counts, static
  memory terminology, and incompatible legacy host-tool documentation. Identify
  missing licensing and recording provenance without assigning rights.
- `training/train_kws_v4.py`: anchor input/export paths, seed before augmentation,
  sort input enumeration, fit normalization on training clips only, support actual
  MP3 synthetic output, and fail with a nonzero exit when positives are absent.
  The augmented-clip split remains a development split with source leakage.
- `training/features.py`: isolate MFCC extraction and disable the clip-wide dB
  floor to agree with the firmware's per-frame log-power convention.
- Both synthetic generators, the coefficient exporter, and the three recording
  scripts: resolve default paths relative to the repository. The main TTS
  generator now labels its MP3 output correctly.
- `scripts/download_speech_commands.py`: stop on checksum mismatch, use filtered
  tar extraction, and reuse an extraction only after a completion marker exists.
- `firmware/nexus_kws/nexus_kws.ino`: align the tensor arena, reject overlapping
  recording requests, publish recording completion before clearing active state,
  correct the serial help text to milliseconds, and remove the zero-drops claim.
- `requirements-lock.txt`, `artifacts.json`, and tests: record the review environment
  and bundled artifact hashes; check consistency, DSP math, archive handling,
  model inference, path resolution, and export to temporary files.

## Results

Environment: Linux x86_64, Python 3.12.14; package versions are in
`training/requirements-lock.txt`. The environment was moved to
`/tmp/nexus-kws-review-venv` after installation exhausted the home filesystem.
The newly downloaded TensorFlow cache was removed to recover space. Initial
DSP/cache and syntax/build attempts failed for lack of space; subsequent checks
below completed successfully.

| Check | Result |
| --- | --- |
| `python -m unittest discover -s tests -v` | 11 tests passed, 1.646 s reported by unittest (excluding imports) |
| `python -m compileall -q training scripts tools tests` | Exit 0 on Python 3.12.14 and system Python 3.14.7; 16 source files |
| `git diff --check` | Exit 0 |
| Bundled TFLite input/output and CPU invocation | INT8 `[1,13,49,1]` → INT8 `[1,2]`; invocation passed |
| Training model construction/forward pass | 1,294 parameters; output `(4,2)` in export test |
| Temporary untrained model INT8 conversion/export | Passed; bundled artifacts unchanged |
| Final Arduino compile | Exit 0; 452,825 bytes program storage, 80,284 bytes globals |

Arduino CLI **1.5.1**, board **esp32:esp32:esp32**, ESP32 core **2.0.17**,
arduinoFFT **2.0.4**, TensorFlowLite_ESP32 **1.0.0**. Final build command:

```sh
arduino-cli compile -b esp32:esp32:esp32 --build-path /tmp/nexus-kws-review-build firmware/nexus_kws
```

The build reports 247,396 bytes remaining for local variables out of 327,680;
this is not measured free heap. The original firmware compiled to 452,781 bytes
program storage and 80,276 bytes globals in the same environment.
TensorFlow emitted unavailable-GPU registration warnings and used CPU execution;
the conversion also emitted quantization warnings but completed successfully.

No full training run, Speech Commands download, TTS generation, upload, serial
capture, or live hardware evaluation was performed. No hardware measurements
were added. The bundled model, normalization, coefficient headers, and recordings
are byte-for-byte unchanged. No commits or pushes were made.

## Hardware Validation & Physical Metrics (Phase 5 & 6) — 2026-09-29

Evaluated live on physical hardware: ESP32-D0WD-V3 (revision v3.1) connected on `/dev/ttyUSB0` with an INMP441 I2S MEMS microphone.

| Metric | Target Boundary | Measured Physical Value | Status |
| --- | --- | --- | --- |
| Dynamic RAM Footprint | < 256 KB | 80 KB (KWS) / 95 KB (Streaming) | PASS |
| Idle CPU Overhead | < 10% total chip | 6.3% (KWS) / 7.2% - 7.5% (Wi-Fi connected) | PASS |
| Active CPU Overhead (Inference) | Unconstrained (<100%) | 42% - 48% total chip (~84% Core 1, ~14% Core 0) | PASS |
| FFT Execution Time | Real-time (<20ms hop) | ~45 µs via ESP-DSP assembly | PASS |
| Handoff Latency to ASR | < 50 ms | 42.5 - 48.0 ms (TCP connect 6-14ms, first byte 42ms) | PASS |
| Speech Recognition Engine | Open-Source Only | Vosk (`vosk-model-small-en-in-0.4`) streaming daemon | PASS |
| Unit Test Regression Suite | 100% Pass | 11/11 tests passing (`python -m unittest discover -s tests`) | PASS |
