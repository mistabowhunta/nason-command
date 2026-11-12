import asyncio
import time
from datetime import datetime, timezone
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()
START_TIME = time.time()


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_json(self, payload: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(payload)
            except Exception:
                self.disconnect(connection)


manager = ConnectionManager()


async def telemetry_and_cv_broadcaster(cv_engine):
    from app.services.discord_bot import discord_bot_state

    while True:
        now = time.time()
        uptime = int(now - START_TIME)
        away_mode = bool(getattr(cv_engine, "away_mode", False))
        away_armed_at = float(getattr(cv_engine, "away_armed_at", 0.0) or 0.0)
        mute_until = float(getattr(cv_engine, "mute_until", 0.0) or 0.0)
        payload = {
            "type": "HEARTBEAT",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": uptime,
            "active_streams": cv_engine.streams,
            "agent_mode": "🛫 AWAY // 24/7 ARMED" if away_mode else "YOLOv8s // RTX 3060 Ti",
            "away_mode": away_mode,
            "away_grace_remaining": max(0, int(round(away_armed_at - now))) if away_mode else 0,
            "mute_remaining": max(0, int(round(mute_until - now))),
            "bot_connected": bool(discord_bot_state.get("connected", False)),
            "detections": cv_engine.latest_detections,
        }
        await manager.broadcast_json(payload)
        await asyncio.sleep(0.15)


@router.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "PING":
                await websocket.send_json({"type": "PONG"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
