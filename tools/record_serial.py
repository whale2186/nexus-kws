import serial
import wave
import time
import struct
import argparse
import sys
import os

SYNC_AUDIO  = b'\xAA\x55\xAA\x55'
SYNC_STATUS = b'\x55\xAA\x55\xAA'

def main():
    parser = argparse.ArgumentParser(description="Record 16kHz PCM audio from ESP32 via Serial")
    parser.add_argument('--port', type=str, default='/dev/ttyUSB0', help='Serial port')
    parser.add_argument('--baud', type=int, default=921600, help='Baud rate')
    parser.add_argument('--duration', type=int, default=5, help='Record duration in seconds')
    parser.add_argument('--out', type=str, default='baseline_test.wav', help='Output WAV file')
    args = parser.parse_args()

    # Calculate expected number of samples (16000 samples/sec)
    target_samples = 16000 * args.duration

    print(f"Opening {args.port} at {args.baud} baud...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as e:
        print(f"Error opening port: {e}")
        print("Note: If the port is busy, ensure Arduino Serial Monitor is closed.")
        sys.exit(1)

    print("Syncing with ESP32 stream...")
    # Removed the SERIAL_AUDIO_READY wait loop because binary data is already flowing
    
    frames = []
    collected_samples = 0
    start_time = time.time()

    print(f"Recording {args.duration} seconds to {args.out}...")
    
    try:
        while collected_samples < target_samples:
            # Read until we find either sync marker
            header = ser.read(4)
            if not header or len(header) < 4:
                continue
                
            if header == SYNC_AUDIO:
                # Read sample count (uint16_t)
                s_count_bytes = ser.read(2)
                if len(s_count_bytes) < 2:
                    continue
                count = struct.unpack('<H', s_count_bytes)[0]
                
                # Read PCM payload
                pcm_bytes = ser.read(count * 2)
                if len(pcm_bytes) == count * 2:
                    frames.append(pcm_bytes)
                    collected_samples += count
                    
            elif header == SYNC_STATUS:
                # Read type and length
                t_l = ser.read(2)
                if len(t_l) == 2:
                    msg_type = chr(t_l[0])
                    msg_len = t_l[1]
                    msg = ser.read(msg_len).decode('utf-8', errors='ignore')
                    print(f"\n[ESP32 STATUS] {msg}")

            # Progress bar
            if collected_samples % 16000 < 500: # print roughly once a second
               elapsed = time.time() - start_time
               print(f"Captured {collected_samples}/{target_samples} samples ({elapsed:.1f}s)")
               
    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")

    print(f"\nFinished. Saving {collected_samples} samples to {args.out}...")
    
    with wave.open(args.out, 'wb') as f:
        f.setnchannels(1)      # Mono
        f.setsampwidth(2)      # 16-bit
        f.setframerate(16000)  # 16 kHz
        for chunk in frames:
            f.writeframes(chunk)
            
    print(f"Saved successfully to {os.path.abspath(args.out)}. Play it to verify mic quality.")

if __name__ == '__main__':
    main()
