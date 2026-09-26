import time
import subprocess
import soundfile as sf
import numpy as np
import serial
import asyncio
import edge_tts

def main():
    print("=" * 60)
    print("ESP32 Live KWS Verification")
    print("=" * 60)

    ser = serial.Serial('/dev/ttyUSB0', 921600, timeout=0.1)
    
    # Drain initial buffer
    print("Draining serial buffer...")
    t0 = time.time()
    while time.time() - t0 < 2.0:
        ser.read(4096)

    # 1. Baseline Idle Room Check
    print("\n1. Monitoring IDLE room noise floor for 3 seconds...")
    start = time.time()
    idle_lines = []
    while time.time() - start < 3.0:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line:
            idle_lines.append(line)
            print("  [SERIAL]", line)

    idle_triggers = any("WAKE WORD DETECTED" in l for l in idle_lines)
    print(f"Idle false trigger: {idle_triggers} (Expected: False)")

    # 2. Synthetic Clap Test
    print("\n2. Testing CLAP playback...")
    t = np.linspace(0, 0.25, int(16000 * 0.25))
    clap = np.random.randn(len(t)) * np.exp(-t * 60)
    sf.write('/tmp/clap_test.wav', (clap * 0.95).astype(np.float32), 16000)

    ser.reset_input_buffer()
    subprocess.run(['paplay', '--volume=65536', '/tmp/clap_test.wav'], check=False)

    start = time.time()
    clap_lines = []
    while time.time() - start < 2.0:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line:
            clap_lines.append(line)
            print("  [SERIAL]", line)

    clap_triggers = any("WAKE WORD DETECTED" in l for l in clap_lines)
    print(f"Clap false trigger: {clap_triggers} (Expected: False)")

    # 3. Speech Negative Test
    print("\n3. Testing NEGATIVE SPEECH playback ('Hello, how are you doing today?')...")
    asyncio.run(edge_tts.Communicate('Hello, how are you doing today?', 'en-US-GuyNeural').save('/tmp/speech_neg.mp3'))
    ser.reset_input_buffer()
    subprocess.run(['mpv', '--no-video', '--volume=150', '/tmp/speech_neg.mp3'], check=False)

    start = time.time()
    speech_lines = []
    while time.time() - start < 2.0:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line:
            speech_lines.append(line)
            print("  [SERIAL]", line)

    speech_triggers = any("WAKE WORD DETECTED" in l for l in speech_lines)
    print(f"Negative speech false trigger: {speech_triggers} (Expected: False)")

    # 4. Hard Negative: Texas
    print("\n4. Testing HARD PHONETIC NEGATIVE ('Texas')...")
    asyncio.run(edge_tts.Communicate('Texas', 'en-US-AvaNeural').save('/tmp/texas_neg.mp3'))
    ser.reset_input_buffer()
    subprocess.run(['mpv', '--no-video', '--volume=150', '/tmp/texas_neg.mp3'], check=False)

    start = time.time()
    texas_lines = []
    while time.time() - start < 2.0:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line:
            texas_lines.append(line)
            print("  [SERIAL]", line)

    texas_triggers = any("WAKE WORD DETECTED" in l for l in texas_lines)
    print(f"Texas false trigger: {texas_triggers} (Expected: False)")

    # 5. Positive Wake Word Test
    print("\n5. Testing POSITIVE WAKE WORD ('Nexus')...")
    asyncio.run(edge_tts.Communicate('Nexus', 'en-US-AvaNeural').save('/tmp/nexus_pos.mp3'))
    ser.reset_input_buffer()
    subprocess.run(['mpv', '--no-video', '--volume=160', '/tmp/nexus_pos.mp3'], check=False)

    start = time.time()
    nexus_lines = []
    while time.time() - start < 2.5:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line:
            nexus_lines.append(line)
            print("  [SERIAL]", line)

    nexus_triggers = any("WAKE WORD DETECTED" in l for l in nexus_lines)
    print(f"Nexus triggered wake word: {nexus_triggers} (Expected: True if mic volume reached gate)")

    ser.close()

if __name__ == '__main__':
    main()
