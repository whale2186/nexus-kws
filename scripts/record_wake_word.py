#!/usr/bin/env python3
"""
record_wake_word.py — Capture real "Nexus" samples from ESP32 INMP441 mic via UART.

Usage:
    python3 record_wake_word.py                    # Interactive mode, saves to dataset/real_positive/
    python3 record_wake_word.py --count 50         # Record 50 samples
    python3 record_wake_word.py --port /dev/ttyACM0 --duration 1.5

The ESP32 must be running nexus_kws.ino (or serial_audio.ino) firmware.
This script sends "REC:<ms>" over UART, captures the raw PCM response,
and saves each sample as a 16kHz mono WAV.

Tips for good recordings:
  - Vary your distance (10cm, 30cm, 1m)
  - Vary your speed and emphasis
  - Record in different rooms / noise conditions
  - Include some whispered and some loud samples
  - Have other people record if possible
"""

import serial
import struct
import wave
import time
import os
import sys
import argparse


SAMPLE_RATE = 16000
OUTPUT_DIR = "dataset/real_positive"

# Markers from nexus_kws.ino UART recording mode
START_MARKER = b"===AUDIO_START:"
END_MARKER = b"===AUDIO_END==="


def find_serial_port():
    """Try common ESP32 serial ports."""
    for port in ["/dev/ttyUSB0", "/dev/ttyACM0", "/dev/ttyUSB1"]:
        if os.path.exists(port):
            return port
    return None


def record_one_sample(ser, duration_s, sample_num, output_dir):
    """Send REC command and capture the PCM response."""
    duration_ms = int(duration_s * 1000)

    # Flush input buffer
    ser.reset_input_buffer()
    time.sleep(0.1)

    # Send record command
    cmd = f"REC:{duration_ms}\n"
    ser.write(cmd.encode())
    print(f"  Sent: REC:{duration_ms} — Say 'Nexus' now!")

    # Wait for start marker
    timeout_start = time.time()
    raw_data = b""
    expected_samples = 0
    found_start = False

    while time.time() - timeout_start < (duration_s + 5):
        chunk = ser.read(ser.in_waiting or 1)
        if not chunk:
            continue
        raw_data += chunk

        if not found_start:
            idx = raw_data.find(START_MARKER)
            if idx >= 0:
                # Parse sample count: ===AUDIO_START:<samples>===
                end_of_header = raw_data.find(b"===\n", idx + len(START_MARKER))
                if end_of_header < 0:
                    end_of_header = raw_data.find(b"===\r\n", idx + len(START_MARKER))
                if end_of_header >= 0:
                    count_str = raw_data[idx + len(START_MARKER):end_of_header].decode(errors="ignore")
                    expected_samples = int(count_str)
                    # Keep only data after the header
                    raw_data = raw_data[end_of_header + 3:]  # skip "===\n"
                    if raw_data.startswith(b"\n"):
                        raw_data = raw_data[1:]
                    elif raw_data.startswith(b"\r\n"):
                        raw_data = raw_data[2:]
                    found_start = True
                    print(f"  Recording {expected_samples} samples ({expected_samples/SAMPLE_RATE:.2f}s)...")

        if found_start:
            end_idx = raw_data.find(END_MARKER)
            if end_idx >= 0:
                pcm_data = raw_data[:end_idx]
                break

            # Check if we have enough data
            if len(raw_data) >= expected_samples * 2 + 100:
                # Probably missed end marker, trim to expected size
                pcm_data = raw_data[:expected_samples * 2]
                break
    else:
        print("  ERROR: Timeout waiting for audio data!")
        return False

    # Parse PCM samples (little-endian int16)
    num_samples = len(pcm_data) // 2
    if num_samples < SAMPLE_RATE // 4:  # Less than 250ms
        print(f"  WARNING: Only got {num_samples} samples, skipping (too short)")
        return False

    samples = struct.unpack(f"<{num_samples}h", pcm_data[:num_samples * 2])

    # Save as WAV
    filename = os.path.join(output_dir, f"nexus_real_{sample_num:03d}.wav")
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data[:num_samples * 2])

    # Quick stats
    import numpy as np
    arr = np.array(samples, dtype=np.float32)
    rms = np.sqrt(np.mean(arr ** 2))
    peak = np.max(np.abs(arr))
    print(f"  Saved: {filename} ({num_samples} samples, RMS={rms:.0f}, Peak={peak:.0f})")
    return True


def interactive_record(ser, duration_s, output_dir):
    """Interactive recording loop — press Enter for each sample, 'q' to quit."""
    sample_num = 0

    # Find next available sample number
    existing = [f for f in os.listdir(output_dir) if f.startswith("nexus_real_") and f.endswith(".wav")]
    if existing:
        nums = [int(f.split("_")[2].split(".")[0]) for f in existing]
        sample_num = max(nums) + 1
        print(f"Found {len(existing)} existing samples, starting at #{sample_num}")

    print(f"\nRecording {duration_s}s samples to {output_dir}/")
    print("Press Enter to record, 'q' to quit, 's' to skip last\n")

    while True:
        try:
            user_input = input(f"[Sample #{sample_num}] Press Enter to record (q=quit): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input == "q":
            break

        success = record_one_sample(ser, duration_s, sample_num, output_dir)
        if success:
            sample_num += 1

    total = len([f for f in os.listdir(output_dir) if f.endswith(".wav")])
    print(f"\nDone! Total samples in {output_dir}: {total}")
    return total


def batch_record(ser, duration_s, count, output_dir):
    """Record a fixed number of samples with countdown prompts."""
    sample_num = 0
    existing = [f for f in os.listdir(output_dir) if f.startswith("nexus_real_") and f.endswith(".wav")]
    if existing:
        nums = [int(f.split("_")[2].split(".")[0]) for f in existing]
        sample_num = max(nums) + 1

    successful = 0
    for i in range(count):
        print(f"\n--- Sample {i+1}/{count} (#{sample_num}) ---")
        print("  Get ready... ", end="", flush=True)
        time.sleep(0.5)
        print("3... ", end="", flush=True)
        time.sleep(1)
        print("2... ", end="", flush=True)
        time.sleep(1)
        print("1... GO!")

        if record_one_sample(ser, duration_s, sample_num, output_dir):
            sample_num += 1
            successful += 1

        time.sleep(0.5)

    print(f"\nRecorded {successful}/{count} samples successfully.")
    return successful


def main():
    parser = argparse.ArgumentParser(
        description="Record real 'Nexus' wake word samples from ESP32 INMP441 mic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 record_wake_word.py                       # Interactive mode
  python3 record_wake_word.py --count 50            # Batch 50 samples
  python3 record_wake_word.py --count 20 --duration 2.0
  python3 record_wake_word.py --port /dev/ttyACM0
        """,
    )
    parser.add_argument("--port", type=str, default=None, help="Serial port (auto-detect if omitted)")
    parser.add_argument("--baud", type=int, default=921600, help="Baud rate (default: 921600)")
    parser.add_argument("--duration", type=float, default=1.5, help="Recording duration in seconds (default: 1.5)")
    parser.add_argument("--count", type=int, default=0, help="Batch mode: record N samples (0=interactive)")
    parser.add_argument("--out-dir", type=str, default=OUTPUT_DIR, help=f"Output directory (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    # Find port
    port = args.port or find_serial_port()
    if not port:
        print("ERROR: No serial port found. Specify with --port /dev/ttyUSB0")
        sys.exit(1)

    # Create output dir
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Opening {port} at {args.baud} baud...")
    try:
        ser = serial.Serial(port, args.baud, timeout=2)
    except serial.SerialException as e:
        print(f"ERROR: {e}")
        print("Ensure Arduino Serial Monitor is closed and ESP32 is connected.")
        sys.exit(1)

    # Wait for ESP32 to be ready
    time.sleep(2)
    ser.reset_input_buffer()

    try:
        if args.count > 0:
            batch_record(ser, args.duration, args.count, args.out_dir)
        else:
            interactive_record(ser, args.duration, args.out_dir)
    finally:
        ser.close()


if __name__ == "__main__":
    main()
