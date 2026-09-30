#!/usr/bin/env python3
"""
Streaming ASR TCP Server for Nexus Voice Activator.
Listens on TCP port 5000 for incoming audio streams from ESP32.
Transcribes audio in real-time using Vosk.
Measures and reports end-to-end handoff latency.
"""

import socket
import json
import time
import os
import re
import wave
from vosk import Model, KaldiRecognizer

HOST = "0.0.0.0"
PORT = 5000
SAMPLE_RATE = 16000

# Constrained Vocabulary Grammar (High-accuracy edge actions)
COMMAND_GRAMMAR = json.dumps([
    "turn on the light", "turn off the light",
    "turn on the lights", "turn off the lights",
    "turn on all lights", "turn off all lights",
    "turn on light", "turn off light",
    "turn on the fan", "turn off the fan",
    "turn on the ac", "turn off the ac",
    "what is the time", "what is the date", "what is the weather", "what time is it",
    "play music", "stop music", "pause music", "resume music",
    "volume up", "volume down", "mute",
    "open the door", "close the door",
    "hello", "hello nexus", "nexus", "stop", "cancel",
    "[unk]"
])

def dispatch_action(command_text):
    """Parse recognized command and display corresponding edge IoT action."""
    cmd = command_text.lower().strip()
    if not cmd:
        return "No command detected"
    
    if "turn on" in cmd and ("light" in cmd or "lights" in cmd):
        return "💡 [SMART RELAY 1] -> Power ON (Room Lighting Activated)"
    elif "turn off" in cmd and ("light" in cmd or "lights" in cmd):
        return "🌑 [SMART RELAY 1] -> Power OFF (Room Lighting Deactivated)"
    elif "turn on" in cmd and "fan" in cmd:
        return "🌀 [SMART RELAY 2] -> Ceiling Fan ON (Speed: MAX)"
    elif "turn off" in cmd and "fan" in cmd:
        return "🛑 [SMART RELAY 2] -> Ceiling Fan OFF"
    elif "turn on" in cmd and "ac" in cmd:
        return "❄️  [HVAC CONTROLLER] -> AC ON (Target: 24°C)"
    elif "turn off" in cmd and "ac" in cmd:
        return "🛑 [HVAC CONTROLLER] -> AC OFF"
    elif "time" in cmd:
        now_time = time.strftime("%I:%M:%S %p")
        return f"⏰ [SYSTEM CLOCK] -> Current Time is {now_time}"
    elif "date" in cmd:
        now_date = time.strftime("%A, %B %d, %Y")
        return f"📅 [SYSTEM CALENDAR] -> Today is {now_date}"
    elif "music" in cmd or "play" in cmd:
        return "🎵 [MEDIA HUB] -> Audio Stream Playback Started"
    elif "stop" in cmd or "pause" in cmd or "cancel" in cmd:
        return "⏹️  [SYSTEM] -> Action Halted / Cancelled"
    elif "door" in cmd and "open" in cmd:
        return "🚪 [SERVO LOCK] -> Main Door Unlocked (GPIO 22 HIGH)"
    elif "door" in cmd and "close" in cmd:
        return "🔒 [SERVO LOCK] -> Main Door Locked (GPIO 22 LOW)"
    else:
        return f"[COMMAND DISPATCHED] -> Executing: \"{cmd}\""

def run_server():
    print("=" * 60)
    print("Nexus Voice Activator - Streaming ASR Server")
    print("=" * 60)
    
    print("[1/2] Loading Vosk Speech Recognition Model (en-in)...")
    t0 = time.time()
    model = Model(lang="en-in")
    print(f"      Model loaded in {time.time() - t0:.2f}s")

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((HOST, PORT))
    server_sock.listen(5)
    
    print(f"[2/2] Listening for ESP32 stream on {HOST}:{PORT} ...")
    print("      Ready for wake word handoff!\n")

    try:
        while True:
            client_sock, client_addr = server_sock.accept()
            t_connect = time.time()
            print(f"\n[EVENT] Incoming connection from {client_addr[0]}:{client_addr[1]}")
            
            recognizer = KaldiRecognizer(model, SAMPLE_RATE, COMMAND_GRAMMAR)
            recognizer.SetWords(True)

            first_chunk = True
            total_bytes = 0
            t_first_byte = None
            transcript_segments = []

            client_sock.settimeout(4.0)
            audio_frames = bytearray()
            pending_pcm = b""

            try:
                while True:
                    data = client_sock.recv(2560)
                    if not data:
                        break
                    
                    if first_chunk:
                        t_first_byte = time.time()
                        handoff_latency_ms = (t_first_byte - t_connect) * 1000.0
                        print(f"      -> First audio byte received! Connection-to-audio latency: {handoff_latency_ms:.1f} ms")
                        first_chunk = False

                    total_bytes += len(data)
                    # TCP is a byte stream: recv() can split a 16-bit sample.
                    pending_pcm += data
                    aligned_bytes = len(pending_pcm) & ~1
                    pcm = pending_pcm[:aligned_bytes]
                    pending_pcm = pending_pcm[aligned_bytes:]
                    audio_frames.extend(pcm)
                    if not pcm:
                        continue

                    if recognizer.AcceptWaveform(pcm):
                        res = json.loads(recognizer.Result())
                        text = res.get("text", "").strip()
                        if text:
                            transcript_segments.append(text)
                            print(f"      [Interim ASR] \"{text}\"")

            except socket.timeout:
                print("      [Stream] Socket timeout (end of speech transmission)")
            except Exception as e:
                print(f"      [Stream Error] {e}")
            finally:
                client_sock.close()

            if pending_pcm:
                print("      [Stream Warning] Dropped incomplete trailing PCM sample")

            # Result() segments are committed; append only the FinalResult() tail.
            # PartialResult() hypotheses must not be appended (they repeat/revise).
            # Final leftover transcription
            res = json.loads(recognizer.FinalResult())
            final_leftover = res.get("text", "").strip()
            if final_leftover:
                transcript_segments.append(final_leftover)

            full_text = " ".join(transcript_segments).strip()

            # Clean wake-word fragments if present at the start (due to pre-roll)
            cleaned_command = re.sub(r"^(nexus|access|the excess|excess|success)\b[\s,.:;!?-]*", "", full_text, flags=re.IGNORECASE)
            duration_s = (len(audio_frames) / 2) / SAMPLE_RATE

            # Save incoming audio stream to WAV for verification
            wav_dir = os.path.join(os.path.dirname(__file__), "recordings")
            os.makedirs(wav_dir, exist_ok=True)
            wav_path = os.path.join(wav_dir, f"command_{time.time_ns()}.wav")
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_frames)

            print("-" * 50)
            print(f"[ASR RESULT] Spoken Command: \"{cleaned_command}\"")
            print(f"             Edge Action:    {dispatch_action(cleaned_command)}")
            if full_text != cleaned_command:
                print(f"             Raw Transcript: \"{full_text}\"")
            print(f"             Duration:       {duration_s:.2f}s ({total_bytes} bytes)")
            print(f"             Saved WAV:      {wav_path}")
            print(f"             Status:         Voice activation complete")
            print("-" * 50)

    except KeyboardInterrupt:
        print("\nStopping ASR Server.")
    finally:
        server_sock.close()

if __name__ == "__main__":
    run_server()
