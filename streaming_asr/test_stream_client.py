#!/usr/bin/env python3
"""
Test client to simulate ESP32 TCP audio streaming to asr_server.py.
Sends a 16kHz 16-bit PCM audio stream in 20ms hops (640 bytes)
and measures end-to-end timing.
"""

import socket
import time
import sys
import wave
import numpy as np

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

    print(f"             Channels: {n_channels}, Rate: {framerate} Hz, Width: {sampwidth*8}-bit")
    print(f"             Duration: {n_frames / framerate:.2f}s ({len(audio_data)} bytes)")

    print(f"[TEST CLIENT] Connecting to {SERVER_HOST}:{SERVER_PORT} ...")
    t_start = time.time()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER_HOST, SERVER_PORT))
    t_connected = time.time()
    print(f"[TEST CLIENT] Connected in {(t_connected - t_start)*1000:.1f} ms! Streaming audio...")

    # Stream in 20ms chunks (640 bytes per chunk at 16kHz 16-bit mono)
    chunk_size = 640
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i + chunk_size]
        sock.sendall(chunk)
        time.sleep(0.019) # real-time pace (~20ms per hop)

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
