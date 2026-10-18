import asyncio
import json
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import select, delete
from app.api.ws import router as ws_router, telemetry_and_cv_broadcaster
from app.db.session import init_db, AsyncSessionLocal
from app.db.models import DetectionEvent, CameraZone
from app.services.cv_pipeline import EdgeCVEngine, db_persistence_worker, SNAPSHOT_DIR

STREAMS = ["annke_1", "tapo_1", "tapo_2"]
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


class PointDTO(BaseModel):
    x: float
    y: float


class CreateZoneDTO(BaseModel):
    camera_id: str
    name: str
    zone_type: str  # 'exclusion' or 'tripwire'
    points: list[PointDTO]


async def sync_zones_to_engine(cv_engine: EdgeCVEngine):
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(CameraZone))
        rows = result.scalars().all()
        grouped: dict[str, list[dict]] = {s: [] for s in STREAMS}
        for r in rows:
            grouped.setdefault(r.camera_id, []).append({
                "id": r.id,
                "camera_id": r.camera_id,
                "name": r.name,
                "zone_type": r.zone_type,
                "points": json.loads(r.points_json),
            })
        cv_engine.set_zones(grouped)
        return grouped


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    loop = asyncio.get_running_loop()
    event_queue: asyncio.Queue = asyncio.Queue()

    cv_engine = EdgeCVEngine(streams=STREAMS, event_queue=event_queue, loop=loop)
    app.state.cv_engine = cv_engine
    await sync_zones_to_engine(cv_engine)
    cv_engine.start()

    ws_task = asyncio.create_task(telemetry_and_cv_broadcaster(cv_engine))
    db_task = asyncio.create_task(db_persistence_worker(event_queue))
    yield
    cv_engine.stop()
    ws_task.cancel()
    db_task.cancel()


app = FastAPI(title="NasonNation | Command API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/snapshots", StaticFiles(directory=SNAPSHOT_DIR), name="snapshots")
app.include_router(ws_router)


@app.get("/api/health")
async def health_check():
    return {"status": "ONLINE", "system": "NasonNation | Command"}


@app.get("/api/zones")
async def get_zones():
    return await sync_zones_to_engine(app.state.cv_engine)


@app.post("/api/zones")
async def create_zone(payload: CreateZoneDTO):
    async with AsyncSessionLocal() as session:
        row = CameraZone(
            camera_id=payload.camera_id,
            name=payload.name,
            zone_type=payload.zone_type,
            points_json=json.dumps([p.model_dump() for p in payload.points]),
        )
        session.add(row)
        await session.commit()
    return await sync_zones_to_engine(app.state.cv_engine)


@app.delete("/api/zones/camera/{camera_id}")
async def clear_camera_zones(camera_id: str):
    async with AsyncSessionLocal() as session:
        await session.execute(delete(CameraZone).where(CameraZone.camera_id == camera_id))
        await session.commit()
    return await sync_zones_to_engine(app.state.cv_engine)


@app.get("/api/detections")
async def get_recent_detections(limit: int = 30):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(DetectionEvent).order_by(DetectionEvent.timestamp.desc()).limit(limit)
        )
        rows = result.scalars().all()
        return [
            {
                "id": r.id,
                "timestamp": r.timestamp.isoformat(),
                "camera_id": r.camera_id,
                "category": r.category,
                "sub_type": r.sub_type,
                "full_label": r.full_label,
                "confidence": r.confidence,
                "crop_url": f"/snapshots/crop_{r.attributes}" if r.attributes else None,
                "full_url": f"/snapshots/full_{r.attributes}" if r.attributes else None,
            }
            for r in rows
        ]
