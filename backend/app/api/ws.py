import asyncio
import time
from datetime import datetime, timezone
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

router = APIRouter()
START_TIME = time.time()


class TelemetryHeartbeat(BaseModel):
    type: str = "HEARTBEAT"
    timestamp: str
    uptime_seconds: int
    active_streams: list[str]
    agent_mode: str
    detections: dict[str, list[dict]] = {}


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_json(self, data: dict):
        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception:
                dead.append(connection)
        for d in dead:
            self.disconnect(d)


manager = ConnectionManager()


async def telemetry_and_cv_broadcaster(cv_engine):
    while True:
        if manager.active_connections:
            payload = TelemetryHeartbeat(
                timestamp=datetime.now(timezone.utc).isoformat(),
                uptime_seconds=int(time.time() - START_TIME),
                active_streams=cv_engine.streams,
                agent_mode="CV ENGINE ACTIVE (YOLOv8n)",
                detections=cv_engine.latest_detections,
            )
            await manager.broadcast_json(payload.model_dump())
        await asyncio.sleep(0.15)


@router.websocket("/ws/telemetry")
async def telemetry_ws_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "PONG"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
