"""Local web frontend for VirtualTutor.

Serves a single-page UI and bridges the voice pipeline to the browser using
Server-Sent Events. Built on the Python standard library only, so the project
keeps its offline, no-extra-dependency property.

  GET  /                -> the UI
  GET  /api/personas    -> available tutor personas
  GET  /api/status      -> whether the pipeline is running
  GET  /api/events      -> SSE stream of pipeline events
  POST /api/start       -> start listening   (body: {"persona": "tutor"})
  POST /api/stop        -> stop listening

Audio stays on this machine: the microphone and speakers are driven by the
Python pipeline exactly as in the CLI, and the browser is only the control
surface and transcript view.

SECURITY: binds to 127.0.0.1 and has NO authentication. It is intended for
local single-user use. Do not expose it to a network or bind it to 0.0.0.0
without adding access control -- anyone who can reach it could start your
microphone.

Run: ./.venv/bin/python src/server.py     then open http://127.0.0.1:8800
"""
import json
import mimetypes
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import config
import voice_agent

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
HOST = os.environ.get("VT_WEB_HOST", "127.0.0.1")
PORT = int(os.environ.get("VT_WEB_PORT", "8800"))


# ---------------------------------------------------------------------------
# Event fan-out: one pipeline event queue -> many browser tabs
# ---------------------------------------------------------------------------
class Broker:
    def __init__(self):
        self._clients: list[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=256)
        with self._lock:
            self._clients.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)

    def publish(self, event: dict):
        with self._lock:
            clients = list(self._clients)
        for q in clients:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass  # slow tab: drop rather than stall the pipeline


broker = Broker()


def pump_events():
    """Forward pipeline events to all connected browsers."""
    while True:
        try:
            event = voice_agent.event_q.get(timeout=0.5)
        except queue.Empty:
            continue
        broker.publish(event)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "VirtualTutor"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the console focused on the pipeline
        pass

    # -- helpers ----------------------------------------------------------
    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path):
        if not path.is_file():
            self._send_json({"error": "not found"}, 404)
            return
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    # -- routes -----------------------------------------------------------
    def do_GET(self):
        route = self.path.split("?", 1)[0]
        if route == "/":
            self._send_file(WEB_DIR / "index.html")
        elif route == "/api/personas":
            self._send_json({
                "default": config.DEFAULT_PERSONA,
                "personas": [
                    {"key": k, "name": v["name"], "blurb": v["blurb"]}
                    for k, v in config.PERSONAS.items()
                ],
            })
        elif route == "/api/status":
            self._send_json({"running": voice_agent.is_running()})
        elif route == "/api/events":
            self._stream_events()
        else:
            # Static assets, restricted to the web directory.
            candidate = (WEB_DIR / route.lstrip("/")).resolve()
            if WEB_DIR in candidate.parents:
                self._send_file(candidate)
            else:
                self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        route = self.path.split("?", 1)[0]
        if route == "/api/start":
            persona = str(self._read_json().get("persona", config.DEFAULT_PERSONA))
            if persona not in config.PERSONAS:
                persona = config.DEFAULT_PERSONA
            try:
                # Model load on first start can take a few seconds; do it inline
                # so the UI's "starting" state reflects reality.
                voice_agent.start_pipeline(persona)
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 500)
                return
            self._send_json({"ok": True, "running": True, "persona": persona})
        elif route == "/api/stop":
            try:
                voice_agent.stop_pipeline()
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 500)
                return
            self._send_json({"ok": True, "running": False})
        else:
            self._send_json({"error": "not found"}, 404)

    # -- SSE --------------------------------------------------------------
    def _stream_events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        q = broker.subscribe()
        # Tell the newly connected tab where things stand.
        initial = {"type": "state", "state": "listening" if voice_agent.is_running() else "idle"}
        try:
            self._write_event(initial)
            last_ping = time.monotonic()
            while True:
                try:
                    event = q.get(timeout=0.5)
                    self._write_event(event)
                except queue.Empty:
                    # Comment frame keeps proxies and the browser from timing out.
                    if time.monotonic() - last_ping > 15:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        last_ping = time.monotonic()
        except (BrokenPipeError, ConnectionResetError):
            pass  # tab closed
        finally:
            broker.unsubscribe(q)

    def _write_event(self, event: dict):
        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.flush()


def main():
    threading.Thread(target=pump_events, name="events", daemon=True).start()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    print(f"VirtualTutor web UI  ->  http://{HOST}:{PORT}")
    print("Local only, no authentication. Ctrl+C to quit.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down ...")
    finally:
        voice_agent.stop_pipeline()
        server.shutdown()


if __name__ == "__main__":
    main()
