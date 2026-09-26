#!/usr/bin/env python3
"""
Serial Audio Receiver — reads framed PCM from ESP32, serves web UI.

Usage:
    pip install pyserial flask
    python3 server.py              # default /dev/ttyUSB0
    python3 server.py /dev/ttyACM0 # specify port

Then open http://localhost:8080 in a browser.
"""

import sys
import struct
import threading
import time
import queue
from pathlib import Path

import serial
from flask import Flask, Response, send_from_directory, request, jsonify

# ─── config ──────────────────────────────────────────────────────────────────
SERIAL_PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
BAUD = 921600
SAMPLE_RATE = 16000
WEB_PORT = 8080
SYNC = b"\xAA\x55\xAA\x55"
STATUS_SYNC = b"\x55\xAA\x55\xAA"

# ─── shared state ───────────────────────────────────────────────────────────
ser_lock = threading.Lock()
ser_inst = None
wifi_status = {"state": "OFF"}
wifi_lock = threading.Lock()

# per-client audio queues
client_queues = []
client_lock = threading.Lock()

# per-client status queues (for SSE)
status_queues = []
status_lock = threading.Lock()

def push_status(msg):
    with wifi_lock:
        if msg.startswith("WIFI:ON:"):
            wifi_status["state"] = "ON"
            wifi_status["ip"] = msg[8:]
        elif msg == "WIFI:OFF":
            wifi_status["state"] = "OFF"
            wifi_status.pop("ip", None)
        elif msg == "WIFI:CONNECTING":
            wifi_status["state"] = "CONNECTING"
            wifi_status.pop("ip", None)
        elif msg == "WIFI:DISCONNECTED":
            wifi_status["state"] = "DISCONNECTED"
            wifi_status.pop("ip", None)
    with status_lock:
        for q in status_queues:
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass
    print(f"[esp32] {msg}")

# ─── serial reader thread ───────────────────────────────────────────────────
def serial_reader():
    global ser_inst
    while True:
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
            with ser_lock:
                ser_inst = ser
            print(f"[serial] connected to {SERIAL_PORT} at {BAUD}")

            deadline = time.time() + 10
            while time.time() < deadline:
                line = ser.readline()
                if b"SERIAL_AUDIO_READY" in line:
                    print("[serial] ESP32 ready, receiving audio")
                    break

            buf = b""
            while True:
                avail = ser.in_waiting
                chunk = ser.read(avail if avail > 0 else 1)
                if not chunk:
                    continue
                buf += chunk

                # Process all complete frames (audio + status)
                while True:
                    # Find the earliest sync marker of either type
                    audio_idx = buf.find(SYNC)
                    status_idx = buf.find(STATUS_SYNC)

                    if audio_idx == -1 and status_idx == -1:
                        buf = buf[-3:] if len(buf) > 3 else buf
                        break

                    # Process whichever comes first
                    if status_idx != -1 and (audio_idx == -1 or status_idx < audio_idx):
                        buf = buf[status_idx:]
                        # STATUS_SYNC(4) + type(1) + len(1) + text(len)
                        if len(buf) < 6:
                            break
                        msg_len = buf[5]
                        if len(buf) < 6 + msg_len:
                            break
                        msg = buf[6:6 + msg_len].decode("utf-8", errors="replace")
                        buf = buf[6 + msg_len:]
                        push_status(msg)
                        continue

                    # Audio frame
                    buf = buf[audio_idx:]
                    if len(buf) < 6:
                        break
                    count = struct.unpack_from("<H", buf, 4)[0]
                    frame_size = 6 + count * 2
                    if len(buf) < frame_size:
                        break
                    pcm = buf[6:frame_size]
                    buf = buf[frame_size:]

                    with client_lock:
                        for q in client_queues:
                            try:
                                q.put_nowait(pcm)
                            except queue.Full:
                                try:
                                    q.get_nowait()
                                except queue.Empty:
                                    pass
                                try:
                                    q.put_nowait(pcm)
                                except queue.Full:
                                    pass

        except serial.SerialException as e:
            print(f"[serial] {e}, retrying in 2s...")
            time.sleep(2)
        except Exception as e:
            print(f"[serial] error: {e}, retrying in 2s...")
            time.sleep(2)

# ─── Flask app ───────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=None)
static_dir = Path(__file__).parent

@app.route("/")
def index():
    return send_from_directory(static_dir, "index.html")

@app.route("/stream")
def stream():
    def generate():
        q = queue.Queue(maxsize=10)
        with client_lock:
            client_queues.append(q)
        try:
            while True:
                try:
                    data = q.get(timeout=2.0)
                    yield data
                except queue.Empty:
                    continue
        finally:
            with client_lock:
                client_queues.remove(q)

    return Response(generate(),
                    mimetype="application/octet-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                        "Access-Control-Allow-Origin": "*",
                    })

@app.route("/wifi", methods=["POST"])
def wifi_toggle():
    """Send WiFi on/off command to ESP32."""
    action = request.json.get("action", "")
    with ser_lock:
        if ser_inst and ser_inst.is_open:
            if action == "on":
                ser_inst.write(b"W")
                return jsonify({"ok": True, "sent": "W"})
            elif action == "off":
                ser_inst.write(b"w")
                return jsonify({"ok": True, "sent": "w"})
    return jsonify({"ok": False, "error": "invalid action or serial not connected"}), 400

@app.route("/wifi/status")
def wifi_get_status():
    with wifi_lock:
        return jsonify(wifi_status)

@app.route("/events")
def events():
    """SSE stream for real-time status updates from ESP32."""
    def generate():
        q = queue.Queue(maxsize=20)
        with status_lock:
            status_queues.append(q)
        try:
            # Send current state immediately
            with wifi_lock:
                yield f"data: {wifi_status.get('state','OFF')}\n\n"
            while True:
                try:
                    msg = q.get(timeout=15)
                    yield f"data: {msg}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with status_lock:
                status_queues.remove(q)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ─── main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t = threading.Thread(target=serial_reader, daemon=True)
    t.start()
    print(f"[web] http://localhost:{WEB_PORT}")
    app.run(host="0.0.0.0", port=WEB_PORT, threaded=True)
