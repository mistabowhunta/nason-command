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
    while True:
        uptime = int(time.time() - START_TIME)
        payload = {
            "type": "HEARTBEAT",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": uptime,
            "active_streams": cv_engine.streams,
            "agent_mode": "YOLOv8s // RTX 3060 Ti",
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
