#!/usr/bin/env python3
"""Live serial monitor for ESP32 Nexus KWS."""

import sys
import time
import serial

PORT = "/dev/ttyUSB0"
BAUD = 921600

def main():
    print("=" * 60)
    print("Nexus KWS Live Monitor (Press Ctrl+C to exit)")
    print(f"Connecting to {PORT} at {BAUD} baud...")
    print("=" * 60)
    
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
    except Exception as e:
        print(f"Error opening port {PORT}: {e}")
        sys.exit(1)
        
    ser.reset_input_buffer()
    print("Listening... Say 'Nexus' or other words:\n")
    
    try:
        while True:
            line = ser.readline()
            if line:
                txt = line.decode('utf-8', errors='ignore').strip()
                if txt:
                    if "WAKE WORD DETECTED" in txt:
                        print(f"\033[92m{txt}\033[0m")
                    elif txt.startswith("Prob:"):
                        print(f"  {txt}")
                    else:
                        print(f"[STATUS] {txt}")
    except KeyboardInterrupt:
        print("\nExiting monitor.")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
