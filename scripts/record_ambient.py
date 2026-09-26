#!/usr/bin/env python3
"""Automated ambient noise collector for ESP32 INMP441 microphone."""

import os
import sys
import time
import wave
import serial

PORT = "/dev/ttyUSB0"
BAUD = 921600
OUT_DIR = "dataset/real_negative"
SAMPLE_RATE = 16000
SAMPLE_COUNT = 40
DURATION_S = 1.5
START_MARKER = b"===AUDIO_START:"
END_MARKER = b"===AUDIO_END==="

def record_sample(ser, duration_ms):
    ser.reset_input_buffer()
    time.sleep(0.05)
    ser.write(f"REC:{duration_ms}\n".encode())
    
    t0 = time.time()
    raw = b""
    expected = 0
    found_start = False
    
    while time.time() - t0 < (DURATION_S + 4):
        chunk = ser.read(ser.in_waiting or 1)
        if not chunk:
            continue
        raw += chunk
        
        if not found_start:
            idx = raw.find(START_MARKER)
            if idx >= 0:
                end_hdr = raw.find(b"===\n", idx + len(START_MARKER))
                if end_hdr < 0:
                    end_hdr = raw.find(b"===\r\n", idx + len(START_MARKER))
                if end_hdr >= 0:
                    cnt_str = raw[idx + len(START_MARKER):end_hdr].decode(errors="ignore")
                    expected = int(cnt_str)
                    raw = raw[end_hdr + 3:]
                    if raw.startswith(b"\n"):
                        raw = raw[1:]
                    elif raw.startswith(b"\r\n"):
                        raw = raw[2:]
                    found_start = True
        
        if found_start:
            end_idx = raw.find(END_MARKER)
            if end_idx >= 0:
                return raw[:end_idx][:expected * 2]
            if len(raw) >= expected * 2 + 100:
                return raw[:expected * 2]
    return None

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print(f"Connecting to ESP32 on {PORT} at {BAUD} baud...")
    ser = serial.Serial(PORT, BAUD, timeout=2)
    time.sleep(1.0)
    ser.reset_input_buffer()
    
    print(f"\nTarget: {SAMPLE_COUNT} ambient noise samples ({DURATION_S}s each).")
    print("Room exit countdown: 10 seconds to step out and close the door...")
    for i in range(10, 0, -1):
        print(f"  {i}...")
        time.sleep(1.0)
    print("Starting capture now!\n")
    
    success = 0
    for idx in range(SAMPLE_COUNT):
        data = record_sample(ser, int(DURATION_S * 1000))
        if data and len(data) >= int(DURATION_S * SAMPLE_RATE) * 2:
            out_file = os.path.join(OUT_DIR, f"ambient_real_{idx:03d}.wav")
            with wave.open(out_file, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(data[:int(DURATION_S * SAMPLE_RATE) * 2])
            success += 1
            print(f"[{success}/{SAMPLE_COUNT}] Captured {out_file}")
        else:
            print(f"[{idx+1}/{SAMPLE_COUNT}] Failed to capture sample")
        time.sleep(0.2)
    
    ser.close()
    print(f"\nFinished: Successfully recorded {success}/{SAMPLE_COUNT} ambient noise samples.")

if __name__ == "__main__":
    main()
