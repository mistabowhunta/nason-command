import asyncio
import glob
import json
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import select, delete, text
from app.api.ws import router as ws_router, telemetry_and_cv_broadcaster
from app.db.session import init_db, AsyncSessionLocal
from app.db.models import DetectionEvent, CameraZone, CameraSchedule
from app.services.cv_pipeline import EdgeCVEngine, db_persistence_worker, SNAPSHOT_DIR
from app.agent.orchestrator import deep_inspect_detection, process_command_prompt

STREAMS = ["annke_1", "tapo_1", "tapo_2"]
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


class PointDTO(BaseModel):
    x: float
    y: float


class CreateZoneDTO(BaseModel):
    camera_id: str
    name: str
    zone_type: str
    points: list[PointDTO]


class ScheduleDTO(BaseModel):
    mode: str        # 'always_record' | 'paused' | 'schedule'
    start_time: str  # 'HH:MM'
    end_time: str    # 'HH:MM'


class AgentCommandDTO(BaseModel):
    prompt: str


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


async def sync_schedules_to_engine(cv_engine: EdgeCVEngine):
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(CameraSchedule))
        rows = result.scalars().all()
        sched_map = {
            s: {"camera_id": s, "mode": "always_record", "start_time": "21:00", "end_time": "06:00"}
            for s in STREAMS
        }
        for r in rows:
            sched_map[r.camera_id] = {
                "camera_id": r.camera_id,
                "mode": r.mode,
                "start_time": r.start_time,
                "end_time": r.end_time,
            }
        cv_engine.set_schedules(sched_map)
        return cv_engine.get_schedules_status()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    loop = asyncio.get_running_loop()
    event_queue: asyncio.Queue = asyncio.Queue()

    cv_engine = EdgeCVEngine(streams=STREAMS, event_queue=event_queue, loop=loop)
    app.state.cv_engine = cv_engine
    await sync_zones_to_engine(cv_engine)
    await sync_schedules_to_engine(cv_engine)
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


@app.get("/api/cameras/schedules")
async def get_camera_schedules():
    return await sync_schedules_to_engine(app.state.cv_engine)


@app.post("/api/cameras/{camera_id}/schedule")
async def update_camera_schedule(camera_id: str, payload: ScheduleDTO):
    async with AsyncSessionLocal() as session:
        existing = await session.get(CameraSchedule, camera_id)
        if existing:
            existing.mode = payload.mode
            existing.start_time = payload.start_time
            existing.end_time = payload.end_time
        else:
            session.add(
                CameraSchedule(
                    camera_id=camera_id,
                    mode=payload.mode,
                    start_time=payload.start_time,
                    end_time=payload.end_time,
                )
            )
        await session.commit()
    return await sync_schedules_to_engine(app.state.cv_engine)


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
async def get_recent_detections(limit: int = 35):
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
                "bbox_x": r.bbox_x,
                "bbox_y": r.bbox_y,
                "bbox_w": r.bbox_w,
                "bbox_h": r.bbox_h,
                "crop_url": f"/snapshots/crop_{r.attributes}" if r.attributes else None,
                "full_url": f"/snapshots/full_{r.attributes}" if r.attributes else None,
                "ai_notes": r.ai_notes,
            }
            for r in rows
        ]


@app.delete("/api/detections")
async def purge_all_detections():
    for f in glob.glob(f"{SNAPSHOT_DIR}/*.jpg"):
        try:
            os.remove(f)
        except OSError:
            pass
    async with AsyncSessionLocal() as session:
        await session.execute(text("TRUNCATE TABLE detection_events RESTART IDENTITY;"))
        await session.commit()
    return {"status": "PURGED"}


@app.post("/api/detections/{event_id}/inspect")
async def inspect_detection_endpoint(event_id: int):
    return await deep_inspect_detection(event_id)


@app.post("/api/agent/command")
async def agent_command_endpoint(payload: AgentCommandDTO):
    return await process_command_prompt(payload.prompt)
