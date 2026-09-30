#!/usr/bin/env python3
"""
Test client to simulate ESP32 TCP audio streaming to asr_server.py.
Sends a 16kHz 16-bit PCM audio stream in 80ms chunks (2,560 bytes)
and measures end-to-end timing.
"""

import socket
import time
import sys
import wave

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5000

def test_client(wav_path):
    print(f"[TEST CLIENT] Opening WAV file: {wav_path}")
    with wave.open(wav_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)

    if (n_channels, sampwidth, framerate) != (1, 2, 16000):
        raise ValueError("Expected mono, 16-bit, 16 kHz PCM WAV")

    print(f"             Channels: {n_channels}, Rate: {framerate} Hz, Width: {sampwidth*8}-bit")
    print(f"             Duration: {n_frames / framerate:.2f}s ({len(audio_data)} bytes)")

    print(f"[TEST CLIENT] Connecting to {SERVER_HOST}:{SERVER_PORT} ...")
    t_start = time.time()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER_HOST, SERVER_PORT))
    t_connected = time.time()
    print(f"[TEST CLIENT] Connected in {(t_connected - t_start)*1000:.1f} ms! Streaming audio...")

    # Match firmware: four 20 ms hops per TCP write.
    chunk_size = 2560
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i + chunk_size]
        sock.sendall(chunk)
        time.sleep(len(chunk) / (16000 * 2))

    print("[TEST CLIENT] Audio streaming complete. Closing socket.")
    sock.close()
    print(f"[TEST CLIENT] Total elapsed: {time.time() - t_start:.2f}s")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        wav = sys.argv[1]
    else:
        # Default test WAV
        wav = "dataset/speech_commands_v2/yes/0a7c4000_nohash_0.wav"
    test_client(wav)
