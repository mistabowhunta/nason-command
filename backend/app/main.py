import asyncio
import base64
import glob
import json
import os
import time
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
import cv2
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import select, delete, text
from app.api.ws import router as ws_router, telemetry_and_cv_broadcaster
from app.db.session import init_db, AsyncSessionLocal
from app.db.models import DetectionEvent, CameraZone, CameraSchedule, KnownFace, PushSettings
from app.services.cv_pipeline import EdgeCVEngine, db_persistence_worker, SNAPSHOT_DIR
from app.services.face_engine import face_engine, FACE_CROPS_DIR
from app.services.watchlist_scraper import periodic_watchlist_worker, sync_public_mugshot_feeds
from app.services.push_notifier import get_push_config, send_tactical_push
from app.agent.orchestrator import deep_inspect_detection, process_command_prompt

STREAMS = ["annke_1", "tapo_1", "tapo_2"]
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(FACE_CROPS_DIR, exist_ok=True)


class PointDTO(BaseModel):
    x: float
    y: float


class CreateZoneDTO(BaseModel):
    camera_id: str
    name: str
    zone_type: str
    points: list[PointDTO]


class ScheduleDTO(BaseModel):
    mode: str
    start_time: str
    end_time: str


class EnrollFaceDTO(BaseModel):
    name: str
    role: str = "whitelist"


class UpdateFaceDTO(BaseModel):
    name: str
    role: str


class UploadFaceImageDTO(BaseModel):
    name: str
    role: str = "watchlist"
    image_base64: str


class ScrapeFaceUrlDTO(BaseModel):
    name: str
    role: str = "watchlist"
    image_url: str


class PushConfigDTO(BaseModel):
    enabled: bool
    ntfy_topic: str
    discord_webhook: str
    notify_breach: bool
    notify_watchlist: bool
    notify_delivery: bool


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
    await face_engine.sync_gallery_from_db_and_disk()
    loop = asyncio.get_running_loop()
    event_queue: asyncio.Queue = asyncio.Queue()

    cv_engine = EdgeCVEngine(streams=STREAMS, event_queue=event_queue, loop=loop)
    app.state.cv_engine = cv_engine
    await sync_zones_to_engine(cv_engine)
    await sync_schedules_to_engine(cv_engine)
    cv_engine.start()

    ws_task = asyncio.create_task(telemetry_and_cv_broadcaster(cv_engine))
    db_task = asyncio.create_task(db_persistence_worker(event_queue))
    scraper_task = asyncio.create_task(periodic_watchlist_worker())
    yield
    cv_engine.stop()
    ws_task.cancel()
    db_task.cancel()
    scraper_task.cancel()


app = FastAPI(title="NasonNation | Command API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/snapshots", StaticFiles(directory=SNAPSHOT_DIR), name="snapshots")
app.mount("/face_crops", StaticFiles(directory=FACE_CROPS_DIR), name="face_crops")
app.include_router(ws_router)


@app.get("/api/health")
async def health_check():
    return {
        "status": "ONLINE",
        "system": "NasonNation | Command",
        "enrolled_faces": len(face_engine.gallery),
    }


@app.get("/api/push/settings")
async def read_push_settings():
    cfg = await get_push_config()
    return {
        "enabled": cfg.enabled,
        "ntfy_topic": cfg.ntfy_topic,
        "discord_webhook": cfg.discord_webhook,
        "notify_breach": cfg.notify_breach,
        "notify_watchlist": cfg.notify_watchlist,
        "notify_delivery": cfg.notify_delivery,
    }


@app.post("/api/push/settings")
async def save_push_settings(payload: PushConfigDTO):
    async with AsyncSessionLocal() as session:
        cfg = await session.get(PushSettings, 1)
        if not cfg:
            cfg = PushSettings(id=1)
            session.add(cfg)
        cfg.enabled = payload.enabled
        cfg.ntfy_topic = payload.ntfy_topic.strip()
        cfg.discord_webhook = payload.discord_webhook.strip()
        cfg.notify_breach = payload.notify_breach
        cfg.notify_watchlist = payload.notify_watchlist
        cfg.notify_delivery = payload.notify_delivery
        await session.commit()
    return await read_push_settings()


@app.post("/api/push/test")
async def trigger_test_push():
    return await send_tactical_push(
        title="🛡️ NASONNATION // COMMAND TEST",
        message="Tactical Push Alert link verified from node BOSSMAN (192.168.1.27).",
        priority="high",
        tags="white_check_mark,shield",
    )


@app.get("/api/faces")
async def list_enrolled_faces(include_auto: bool = False):
    items = await face_engine.sync_gallery_from_db_and_disk()
    visible = [
        {
            "id": f["id"],
            "name": f["name"],
            "role": "watchlist" if f["role"] == "watchlist_auto" else f["role"],
            "raw_role": f["role"],
            "sample_crop": f["sample_crop"] if str(f["sample_crop"] or "").startswith("/") else None,
            "sightings": f["sightings"],
            "last_seen": f["last_seen"],
            "created_at": f["created_at"],
        }
        for f in items
        if include_auto or f["role"] != "watchlist_auto" or int(f.get("sightings") or 0) > 0
    ]
    return visible


@app.get("/api/faces/watchlist_status")
async def get_watchlist_status():
    auto_count = sum(1 for f in face_engine.gallery if f["role"] == "watchlist_auto")
    return {"auto_watchlist_vectors": auto_count}


@app.post("/api/faces/sync_watchlist")
async def trigger_watchlist_sync():
    return await sync_public_mugshot_feeds(max_new_per_pass=50)


@app.patch("/api/faces/{face_id}")
async def update_face_entry(face_id: int, payload: UpdateFaceDTO):
    clean_name = payload.name.strip().upper()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")
    async with AsyncSessionLocal() as session:
        row = await session.get(KnownFace, face_id)
        if not row:
            raise HTTPException(status_code=404, detail="Face vector not found")
        row.name = clean_name
        row.role = payload.role
        await session.commit()
    return await list_enrolled_faces()


@app.delete("/api/faces/{face_id}")
async def delete_face_entry(face_id: int):
    async with AsyncSessionLocal() as session:
        row = await session.get(KnownFace, face_id)
        if not row:
            raise HTTPException(status_code=404, detail="Face vector not found")
        await session.execute(delete(KnownFace).where(KnownFace.id == face_id))
        await session.commit()
    return await list_enrolled_faces()


@app.post("/api/faces/enroll_image")
async def enroll_face_from_uploaded_image(payload: UploadFaceImageDTO):
    clean_name = payload.name.strip().upper()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Name is required")

    b64_str = payload.image_base64
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(b64_str)
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    vec, _, face_crop = face_engine.extract_face_data(img)
    if vec is None:
        raise HTTPException(status_code=422, detail="YuNet could not detect a clear face in the uploaded photo.")

    fname = f"upload_{clean_name}_{int(time.time() * 1000)}.jpg"
    cv2.imwrite(f"{FACE_CROPS_DIR}/{fname}", face_crop if face_crop is not None else img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    async with AsyncSessionLocal() as session:
        row = KnownFace(
            name=clean_name,
            role=payload.role,
            embedding_json=json.dumps(vec.tolist()),
            sample_crop=f"/face_crops/{fname}",
            sightings=1,
            last_seen=datetime.now(timezone.utc),
        )
        session.add(row)
        await session.commit()

    return await list_enrolled_faces()


@app.post("/api/faces/enroll_url")
async def enroll_face_from_url(payload: ScrapeFaceUrlDTO):
    clean_name = payload.name.strip().upper()
    if not clean_name or not payload.image_url.strip():
        raise HTTPException(status_code=400, detail="Name and Image URL are required")

    try:
        async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
            resp = await client.get(payload.image_url.strip(), headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                raise HTTPException(status_code=400, detail=f"HTTP {resp.status_code} fetching URL")
            np_arr = np.frombuffer(resp.content, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image URL: {e}")

    vec, _, face_crop = face_engine.extract_face_data(img)
    if vec is None:
        raise HTTPException(status_code=422, detail="YuNet could not isolate a clear face from that URL.")

    fname = f"url_{clean_name}_{int(time.time() * 1000)}.jpg"
    cv2.imwrite(f"{FACE_CROPS_DIR}/{fname}", face_crop if face_crop is not None else img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    async with AsyncSessionLocal() as session:
        row = KnownFace(
            name=clean_name,
            role=payload.role,
            embedding_json=json.dumps(vec.tolist()),
            sample_crop=f"/face_crops/{fname}",
            sightings=1,
            last_seen=datetime.now(timezone.utc),
        )
        session.add(row)
        await session.commit()

    return await list_enrolled_faces()


@app.post("/api/detections/{event_id}/enroll_face")
async def enroll_face_from_detection(event_id: int, payload: EnrollFaceDTO):
    clean_name = payload.name.strip().upper()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(DetectionEvent).where(DetectionEvent.id == event_id))
        ev = result.scalar_one_or_none()
        if not ev or not ev.attributes:
            raise HTTPException(status_code=404, detail="Detection snapshot not found")

        crop_path = f"{SNAPSHOT_DIR}/crop_{ev.attributes}"
        full_path = f"{SNAPSHOT_DIR}/full_{ev.attributes}"

        vec, _, face_crop = (None, 0.0, None)
        if os.path.exists(crop_path):
            vec, _, face_crop = face_engine.extract_face_data(cv2.imread(crop_path))
        if vec is None and os.path.exists(full_path):
            vec, _, face_crop = face_engine.extract_face_data(cv2.imread(full_path))

        if vec is None:
            raise HTTPException(
                status_code=422,
                detail="YuNet could not isolate a clear face in this snapshot. Try a closer front/angled frame.",
            )

        persistent_fname = f"enrolled_{clean_name}_{int(time.time() * 1000)}.jpg"
        if face_crop is not None:
            cv2.imwrite(f"{FACE_CROPS_DIR}/{persistent_fname}", face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        elif os.path.exists(crop_path):
            cv2.imwrite(f"{FACE_CROPS_DIR}/{persistent_fname}", cv2.imread(crop_path))

        row = KnownFace(
            name=clean_name,
            role=payload.role,
            embedding_json=json.dumps(vec.tolist()),
            sample_crop=f"/face_crops/{persistent_fname}",
            sightings=1,
            last_seen=datetime.now(timezone.utc),
        )
        session.add(row)

        ev.sub_type = f"⚠️ WATCHLIST: {clean_name}" if payload.role == "watchlist" else f"ID: {clean_name}"
        ev.full_label = f"{ev.motion_slot}_{clean_name}"
        ev.ai_notes = f"🧬 [ENROLLED {payload.role.upper()}: {clean_name}] {ev.ai_notes or ''}"
        await session.commit()

    await face_engine.sync_gallery_from_db_and_disk()
    return {
        "status": "ENROLLED",
        "name": clean_name,
        "sub_type": ev.sub_type,
        "full_label": ev.full_label,
        "ai_notes": ev.ai_notes,
        "total_faces": len(face_engine.gallery),
    }


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


@app.get("/api/detections/heatmap/{camera_id}")
async def get_camera_heatmap(camera_id: str):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    points = []
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(DetectionEvent)
            .where(DetectionEvent.camera_id == camera_id, DetectionEvent.timestamp >= cutoff)
            .limit(400)
        )
        for r in result.scalars().all():
            cx = round(r.bbox_x + r.bbox_w * 0.5, 4)
            cy = round(r.bbox_y + (r.bbox_h * 0.88 if r.category in ("person", "animal") else r.bbox_h * 0.5), 4)
            points.append({
                "x": cx,
                "y": cy,
                "category": r.category,
                "breach": "BREACH" in (r.full_label or ""),
            })

    cv_engine = getattr(app.state, "cv_engine", None)
    if cv_engine and hasattr(cv_engine, "heatmap_traces"):
        now_ts = time.time()
        for pt in list(cv_engine.heatmap_traces.get(camera_id, [])):
            if now_ts - pt.get("ts", now_ts) <= 86400:
                points.append({
                    "x": pt["x"],
                    "y": pt["y"],
                    "category": pt["category"],
                    "breach": pt.get("breach", False),
                })

    return {"camera_id": camera_id, "count": len(points), "points": points[-1200:]}


@app.get("/api/detections")
async def get_recent_detections(limit: int = 35):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(DetectionEvent).order_by(DetectionEvent.timestamp.desc()).limit(limit)
        )
        rows = result.scalars().all()
        out = []
        for r in rows:
            clip_exists = (
                r.attributes
                and r.category in ("person", "animal")
                and os.path.exists(f"{SNAPSHOT_DIR}/clip_{r.attributes}.mp4")
            )
            out.append({
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
                "clip_url": f"/snapshots/clip_{r.attributes}.mp4" if clip_exists else None,
                "ai_notes": r.ai_notes,
            })
        return out


@app.delete("/api/detections")
async def purge_all_detections():
    for ext in ("*.jpg", "*.mp4"):
        for f in glob.glob(f"{SNAPSHOT_DIR}/{ext}"):
            try:
                os.remove(f)
            except OSError:
                pass
    cv_engine = getattr(app.state, "cv_engine", None)
    if cv_engine and hasattr(cv_engine, "heatmap_traces"):
        for k in cv_engine.heatmap_traces:
            cv_engine.heatmap_traces[k].clear()
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
