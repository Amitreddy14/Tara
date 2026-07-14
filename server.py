"""
server.py — FastAPI + WebSocket bridge between Tara pipeline and the UI.

Run alongside main.py:
    python server.py

The UI connects to ws://localhost:8000/ws and receives real-time events:
  {"type": "state",   "state": "listening"}
  {"type": "message", "role": "user",  "text": "..."}
  {"type": "message", "role": "tara",  "text": "..."}
  {"type": "memory",  "facts": {...}, "tasks": [...]}
  {"type": "status",  "ollama": true, "memory": true}
"""

import asyncio
import json
import logging
import threading
from datetime import datetime
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

log = logging.getLogger("tara.server")

app = FastAPI(title="Tara UI Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Connection manager ────────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self._connections: list[WebSocket] = []
        self._lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.append(ws)
        log.info("UI client connected. Total: %d", len(self._connections))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            if ws in self._connections:
                self._connections.remove(ws)
        log.info("UI client disconnected. Total: %d", len(self._connections))

    async def broadcast(self, data: dict) -> None:
        if not self._connections:
            return
        message = json.dumps(data)
        dead = []
        for ws in self._connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)

    def emit(self, data: dict) -> None:
        """Thread-safe emit — call from non-async Tara pipeline threads."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast(data), self._loop)


manager = ConnectionManager()


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    # Send initial state on connect
    await ws.send_text(json.dumps({"type": "connected", "message": "Tara UI connected"}))
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            # Handle UI → server commands
            if msg.get("type") == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        await manager.disconnect(ws)


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/api/memory")
async def get_memory():
    """Return current semantic memory and tasks."""
    try:
        from memory.semantic import SemanticMemory
        from memory.tasks    import TaskMemory
        sm = SemanticMemory()
        tm = TaskMemory()
        return {
            "facts":   sm.get_all(),
            "tasks":   tm.get_pending(),
            "time":    datetime.now().strftime("%I:%M %p"),
            "date":    datetime.now().strftime("%A, %B %d, %Y"),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/status")
async def get_status():
    return {
        "status": "online",
        "time":   datetime.now().strftime("%I:%M %p"),
        "date":   datetime.now().strftime("%A, %B %d"),
    }


# ── Serve the UI ──────────────────────────────────────────────────────────────

@app.get("/")
async def serve_ui():
    return FileResponse("ui/index.html")


# ── Server startup ────────────────────────────────────────────────────────────

def run_server():
    """Run in a background thread from main.py."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    manager.set_loop(loop)
    config_uv = uvicorn.Config(app, host="0.0.0.0", port=8000, loop="none", log_level="warning")
    server = uvicorn.Server(config_uv)
    loop.run_until_complete(server.serve())


def start_server_thread():
    t = threading.Thread(target=run_server, daemon=True, name="ui-server")
    t.start()
    log.info("UI server started at http://localhost:8000")
    return t


# ── Emit helpers (called from pipeline) ──────────────────────────────────────

def emit_state(state: str) -> None:
    manager.emit({"type": "state", "state": state})

def emit_message(role: str, text: str) -> None:
    manager.emit({"type": "message", "role": role, "text": text,
                  "time": datetime.now().strftime("%H:%M:%S")})

def emit_memory(facts: dict, tasks: list) -> None:
    manager.emit({"type": "memory", "facts": facts, "tasks": tasks})

def emit_status(data: dict) -> None:
    manager.emit({"type": "status", **data})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)