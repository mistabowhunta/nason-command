import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from app.api.ws import router as ws_router, telemetry_and_cv_broadcaster
from app.db.session import init_db, AsyncSessionLocal
from app.db.models import DetectionEvent
from app.services.cv_pipeline import EdgeCVEngine, db_persistence_worker

STREAMS = ["annke_1", "tapo_1", "tapo_2"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    loop = asyncio.get_running_loop()
    event_queue: asyncio.Queue = asyncio.Queue()

    cv_engine = EdgeCVEngine(streams=STREAMS, event_queue=event_queue, loop=loop)
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

app.include_router(ws_router)


@app.get("/api/health")
async def health_check():
    return {"status": "ONLINE", "system": "NasonNation | Command"}


@app.get("/api/detections")
async def get_recent_detections(limit: int = 25):
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
                "full_label": r.full_label,
                "confidence": r.confidence,
                "bbox": [r.bbox_x, r.bbox_y, r.bbox_w, r.bbox_h],
            }
            for r in rows
        ]
