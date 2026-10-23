import base64
import json
import os
import cv2
import httpx
import numpy as np
from sqlalchemy import select, func, delete
from app.db.session import AsyncSessionLocal
from app.db.models import DetectionEvent, CameraZone

WORKSTATION_GPU_URL = os.getenv("WORKSTATION_OLLAMA_URL", "http://192.168.1.223:11434")
SNAPSHOT_DIR = "/tmp/command_snapshots"


async def probe_workstation_ollama() -> tuple[bool, list[str]]:
    try:
        async with httpx.AsyncClient(timeout=0.8) as client:
            res = await client.get(f"{WORKSTATION_GPU_URL}/api/tags")
            if res.status_code == 200:
                models = [m["name"] for m in res.json().get("models", [])]
                return True, models
    except Exception:
        pass
    return False, []


def pick_best_vlm(models: list[str]) -> str | None:
    """Prioritizes qwen2.5vl over legacy llava models."""
    for preferred in ("qwen2.5vl", "qwen2-vl", "minicpm-v", "llama3.2-vision", "llava"):
        for m in models:
            if preferred in m.lower():
                return m
    return None


async def deep_inspect_detection(event_id: int) -> dict:
    """Runs VLM on RTX 3060 Ti to extract accurate paint/clothing color, type, and purge false positives."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(DetectionEvent).where(DetectionEvent.id == event_id))
        ev = result.scalar_one_or_none()
        if not ev:
            return {"error": "Event not found"}

        crop_path = f"{SNAPSHOT_DIR}/crop_{ev.attributes}" if ev.attributes else ""
        gpu_online, models = await probe_workstation_ollama()
        chosen_vlm = pick_best_vlm(models) if gpu_online else None

        if chosen_vlm and os.path.exists(crop_path):
            try:
                with open(crop_path, "rb") as f:
                    b64_img = base64.b64encode(f.read()).decode("utf-8")

                prompt = (
                    f"Analyze this security camera crop from '{ev.camera_id}' (initial detector guess: {ev.category}). "
                    "Ignore window screen mesh and tree shadows. "
                    "Respond ONLY with valid JSON in this exact format: "
                    '{"valid_target": true, "category": "person|vehicle|animal|false_alarm", '
                    '"summary": "Concise 10-18 word description of true body/clothing color, vehicle type, and action"}'
                )

                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.post(
                        f"{WORKSTATION_GPU_URL}/api/generate",
                        json={
                            "model": chosen_vlm,
                            "prompt": prompt,
                            "images": [b64_img],
                            "format": "json",
                            "stream": False,
                            "options": {"temperature": 0.1},
                        },
                    )
                    if resp.status_code == 200:
                        raw_text = resp.json().get("response", "").strip()
                        parsed = json.loads(raw_text)
                        valid = bool(parsed.get("valid_target", True))
                        vlm_cat = str(parsed.get("category", ev.category)).lower()
                        summary = str(parsed.get("summary", "")).strip()

                        if not valid or vlm_cat == "false_alarm":
                            await session.execute(delete(DetectionEvent).where(DetectionEvent.id == ev.id))
                            await session.commit()
                            return {"id": ev.id, "purged": True, "ai_notes": "Purged false positive via RTX 3060 Ti VLM"}

                        if vlm_cat in ("person", "vehicle", "animal") and vlm_cat != ev.category:
                            ev.category = vlm_cat
                            ev.full_label = f"{ev.motion_slot}_{vlm_cat}"

                        ev.ai_notes = f"[RTX 3060 Ti // {chosen_vlm}] {summary}"
                        await session.commit()
                        return {"id": ev.id, "ai_notes": ev.ai_notes, "engine": f"RTX 3060 Ti // {chosen_vlm}"}
            except Exception as e:
                print(f"[VLM ERROR] {e}")

        ev.ai_notes = (
            f"[UNVERIFIED EDGE CROP] {ev.sub_type.upper()} ({int(ev.confidence * 100)}% conf) — "
            "Start Ollama (qwen2.5vl:7b) on 192.168.1.223 for GPU color/object verification."
        )
        await session.commit()
        return {"id": ev.id, "ai_notes": ev.ai_notes, "engine": "EDGE_FALLBACK"}


async def process_command_prompt(prompt: str) -> dict:
    clean = prompt.strip()
    low = clean.lower()

    if any(k in low for k in ("grid", "all cam", "matrix", "reset view", "unfocus")):
        return {
            "reply": "Restored 3-camera Tactical Matrix view.",
            "engine": "DETERMINISTIC_TOOL",
            "ui_action": {"type": "SET_FOCUS", "stream": None},
        }

    for alias, stream in [
        ("garage", "annke_1"), ("driveway", "annke_1"), ("annke", "annke_1"),
        ("exterior 1", "tapo_1"), ("ext 1", "tapo_1"), ("tapo_1", "tapo_1"), ("tapo 1", "tapo_1"),
        ("exterior 2", "tapo_2"), ("ext 2", "tapo_2"), ("tapo_2", "tapo_2"), ("tapo 2", "tapo_2"),
    ]:
        if alias in low and any(v in low for v in ("focus", "maximize", "pull up", "show", "theater", "zoom", "open")):
            return {
                "reply": f"Promoted {stream.upper()} to primary Theater Mode viewport.",
                "engine": "DETERMINISTIC_TOOL",
                "ui_action": {"type": "SET_FOCUS", "stream": stream},
            }

    if "inspect" in low:
        async with AsyncSessionLocal() as session:
            res = await session.execute(
                select(DetectionEvent).order_by(DetectionEvent.timestamp.desc()).limit(1)
            )
            latest = res.scalar_one_or_none()
        if not latest:
            return {"reply": "No logged detection events available to inspect yet.", "engine": "DETERMINISTIC_TOOL"}
        insp = await deep_inspect_detection(latest.id)
        return {
            "reply": f"Inspected Event #{latest.id}: {insp.get('ai_notes')}",
            "engine": insp.get("engine", "RTX_3060_TI"),
            "ui_action": {"type": "REFRESH_RAIL"},
        }

    async with AsyncSessionLocal() as session:
        total_res = await session.execute(
            select(DetectionEvent.category, func.count(DetectionEvent.id)).group_by(DetectionEvent.category)
        )
        counts = {row[0]: row[1] for row in total_res.all()}

        breach_res = await session.execute(
            select(func.count(DetectionEvent.id)).where(DetectionEvent.full_label.like("%BREACH%"))
        )
        breach_count = breach_res.scalar() or 0

        recent_res = await session.execute(
            select(DetectionEvent).order_by(DetectionEvent.timestamp.desc()).limit(5)
        )
        recent_rows = recent_res.scalars().all()
        zone_res = await session.execute(select(CameraZone))
        zones = zone_res.scalars().all()

    gpu_online, models = await probe_workstation_ollama()
    text_models = [m for m in models if "qwen" in m.lower() or "llama" in m.lower()]

    recent_summary = "; ".join(
        [
            f"#{r.id} {r.camera_id} {r.full_label} ({int(r.confidence*100)}%) at {r.timestamp.strftime('%H:%M:%S')} {r.ai_notes or ''}"
            for r in recent_rows
        ]
    ) or "None"

    if gpu_online and text_models:
        chosen_llm = text_models[0]
        sys_context = (
            "You are COMMAND, the tactical AI operating NasonNation's security & robotics node. "
            f"Database totals: {counts}. Tripwire breaches: {breach_count}. Active zones: {len(zones)}. "
            f"Last 5 detections: {recent_summary}. "
            "Answer the operator's query directly and concisely in 1-2 tactical sentences."
        )
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.post(
                    f"{WORKSTATION_GPU_URL}/api/generate",
                    json={"model": chosen_llm, "system": sys_context, "prompt": clean, "stream": False},
                )
                if resp.status_code == 200:
                    ans = resp.json().get("response", "").strip()
                    if ans:
                        return {"reply": ans, "engine": f"RTX 3060 Ti // {chosen_llm}"}
        except Exception:
            pass

    total_all = sum(counts.values())
    return {
        "reply": (
            f"TELEMETRY SUMMARY: {total_all} verified events "
            f"(Person: {counts.get('person', 0)} | Vehicle: {counts.get('vehicle', 0)} | Animal: {counts.get('animal', 0)} | Breaches: {breach_count}). "
            f"Recent: {recent_summary}."
        ),
        "engine": "SQL_TELEMETRY_ENGINE",
    }
