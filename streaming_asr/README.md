# Phase 6 ASR streaming handoff

Copy `firmware/nexus_kws_stream/secrets.h.example` to `secrets.h` in the
same directory and configure Wi-Fi credentials and the ASR server's LAN address.
The sketch uses `__has_include("secrets.h")` and supplies independent defaults for
missing settings. Placeholder credentials allow compilation but must be replaced
for a working connection.

```sh
python3 streaming_asr/asr_server.py
arduino-cli compile -b esp32:esp32:esp32 streaming_asr/firmware/nexus_kws_stream
```

The server requires the `vosk` Python package and loads `Model(lang="en-in")`.
It accumulates committed `Result()` segments and the `FinalResult()` tail, and
saves mono 16-bit, 16 kHz audio to `recordings/command_<timestamp>.wav` beside
the server. Timestamps use nanoseconds to avoid overwriting same-second captures.
TCP receive boundaries may split samples; incomplete samples carry into the next
receive. A lone byte remaining at disconnect is reported and discarded.

The firmware disables the brownout detector at setup, enables maximum modem
sleep, and requests 2 dBm transmit power. Audio retains the existing signed
right shift of 14 bits and gain of 16, with DC filtering and saturation before
conversion to `int16_t`. The 2600 RMS VAD threshold is evaluated on saturated
audio; below it the inference loop yields for 20 ms without invoking the model.
The audio task continues capture and MFCC processing to preserve wake-word context.

A synchronized 200 ms pre-roll snapshot precedes live audio. Live writes combine
four hops into 80 ms / 2,560 bytes, reducing application writes from 50 to 12.5 per
second. Partial writes are retried and a final incomplete batch is sent before
closing. Silence detection uses each queued hop and ends the stream after
1.2 seconds of silence or a five-second live streaming limit. TCP can split a
write into multiple packets; the 75% reduction applies to write frequency.

Run protocol regression tests without Vosk or hardware:

```sh
python3 -m unittest discover -s streaming_asr -p test_asr_server.py -v
```

For an end-to-end check with the server running, send a mono 16-bit, 16 kHz WAV:

```sh
python3 streaming_asr/test_stream_client.py path/to/command.wav
```

On hardware, check idle RMS remains below 2600 with Wi-Fi connected and inspect
saved WAV files for clipping or RF noise. Task CPU telemetry measures capture/DSP
and inference work only; it excludes Wi-Fi, RTOS and interrupt work. Zero idle
inference usage and less than 10% actual total chip usage require hardware
validation. The power settings and batching aim to reduce RF interference;
compilation alone cannot establish microphone rail stability.
