import base64
import os
import cv2
import httpx
import numpy as np
from sqlalchemy import select, func
from app.db.session import AsyncSessionLocal
from app.db.models import DetectionEvent, CameraZone

WORKSTATION_GPU_URL = os.getenv("WORKSTATION_OLLAMA_URL", "http://192.168.1.223:11434")
SNAPSHOT_DIR = "/tmp/command_snapshots"


def _classify_hsv_region(bgr_region: np.ndarray) -> str:
    if bgr_region is None or bgr_region.size == 0:
        return "indeterminate"
    hsv = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.mean(hsv)[:3]
    if v < 50:
        return "black/dark-charcoal"
    if s < 38:
        if v > 185:
            return "white/bright-silver"
        return "metallic-grey/slate"
    if h < 10 or h > 165:
        return "red/crimson"
    if h < 23:
        return "orange/amber"
    if h < 35:
        return "yellow/gold"
    if h < 85:
        return "olive/green"
    if h < 135:
        return "navy/blue"
    return "brown/tan"


def run_edge_forensic_crop_analysis(ev: DetectionEvent, crop_path: str) -> str:
    """Instant OpenCV chromatic + spatial forensic fallback when RTX 3060 Ti Ollama is offline."""
    pos_x = "left" if ev.bbox_x < 0.35 else ("right" if ev.bbox_x > 0.65 else "center")
    pos_y = "far-field" if ev.bbox_y < 0.35 else ("near-field" if ev.bbox_y > 0.65 else "mid-ground")
    breach_tag = " [PERIMETER BREACH]" if "BREACH" in ev.full_label else ""

    if not os.path.exists(crop_path):
        return (
            f"[EDGE FORENSICS]{breach_tag} {ev.sub_type.upper()} tracked in {pos_y} ({pos_x} sector), "
            f"conf {int(ev.confidence * 100)}%."
        )

    img = cv2.imread(crop_path)
    if img is None or img.size == 0:
        return f"[EDGE FORENSICS]{breach_tag} Target {ev.sub_type} verified at {int(ev.confidence * 100)}% conf."

    ih, iw, _ = img.shape
    upper = img[int(ih * 0.15): int(ih * 0.50), int(iw * 0.25): int(iw * 0.75)]
    lower = img[int(ih * 0.50): int(ih * 0.85), int(iw * 0.25): int(iw * 0.75)]
    c_up = _classify_hsv_region(upper)
    c_low = _classify_hsv_region(lower)

    if ev.category == "person":
        posture = "upright pedestrian" if ih > iw * 1.35 else "compact/seated or partial profile"
        return (
            f"[EDGE FORENSICS]{breach_tag} {posture.capitalize()} in {pos_y} ({pos_x}); "
            f"upper garment: {c_up}, lower: {c_low} ({iw}x{ih}px crop)."
        )
    elif ev.category == "vehicle":
        profile = "SUV/truck/van profile" if (ih / max(1, iw)) > 0.68 else "sedan/coupe lateral profile"
        return (
            f"[EDGE FORENSICS]{breach_tag} {c_low.capitalize()} {ev.sub_type} ({profile}) "
            f"in {pos_y} {pos_x} sector; roof/cabin tone: {c_up}."
        )
    else:
        return (
            f"[EDGE FORENSICS]{breach_tag} {c_up.capitalize()} {ev.sub_type} detected in {pos_y} ({pos_x})."
        )


async def probe_workstation_ollama() -> tuple[bool, list[str]]:
    """Checks if Windows 11 RTX 3060 Ti Ollama server is reachable on LAN."""
    try:
        async with httpx.AsyncClient(timeout=0.75) as client:
            res = await client.get(f"{WORKSTATION_GPU_URL}/api/tags")
            if res.status_code == 200:
                models = [m["name"] for m in res.json().get("models", [])]
                return True, models
    except Exception:
        pass
    return False, []


async def deep_inspect_detection(event_id: int) -> dict:
    """Runs VLM on RTX 3060 Ti if online; otherwise runs local Edge Forensics."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(DetectionEvent).where(DetectionEvent.id == event_id))
        ev = result.scalar_one_or_none()
        if not ev:
            return {"error": "Event not found"}

        crop_path = f"{SNAPSHOT_DIR}/crop_{ev.attributes}" if ev.attributes else ""
        gpu_online, models = await probe_workstation_ollama()

        vlm_models = [m for m in models if any(k in m.lower() for k in ("vision", "llava", "moondream", "vl", "minicpm"))]
        notes = None

        if gpu_online and vlm_models and os.path.exists(crop_path):
            chosen_vlm = vlm_models[0]
            try:
                with open(crop_path, "rb") as f:
                    b64_img = base64.b64encode(f.read()).decode("utf-8")
                prompt = (
                    f"You are a tactical security VLM analyzing a camera crop from '{ev.camera_id}'. "
                    f"Detected class: {ev.category} ({ev.sub_type}). "
                    "In 1 concise sentence (max 22 words), describe the subject's color, type/clothing, and orientation."
                )
                async with httpx.AsyncClient(timeout=12.0) as client:
                    resp = await client.post(
                        f"{WORKSTATION_GPU_URL}/api/generate",
                        json={
                            "model": chosen_vlm,
                            "prompt": prompt,
                            "images": [b64_img],
                            "stream": False,
                        },
                    )
                    if resp.status_code == 200:
                        raw_text = resp.json().get("response", "").strip()
                        if raw_text:
                            notes = f"[RTX 3060 Ti // {chosen_vlm}] {raw_text}"
            except Exception as e:
                print(f"[VLM FALLBACK] Ollama vision call failed: {e}")

        if not notes:
            notes = run_edge_forensic_crop_analysis(ev, crop_path)

        ev.ai_notes = notes
        await session.commit()
        return {
            "id": ev.id,
            "ai_notes": notes,
            "engine": "RTX_3060_TI_OLLAMA" if (gpu_online and vlm_models) else "EDGE_OPENCV_FORENSICS",
        }


async def process_command_prompt(prompt: str) -> dict:
    """Deterministic tool-calling & natural language query engine for Command."""
    clean = prompt.strip()
    low = clean.lower()
    ui_action = None

    # 1. Check direct UI layout / camera switching intents
    if any(k in low for k in ("grid", "all cam", "matrix", "reset view", "unfocus")):
        ui_action = {"type": "SET_FOCUS", "stream": None}
        return {
            "reply": "Restored 3-camera Tactical Matrix view.",
            "engine": "DETERMINISTIC_TOOL",
            "ui_action": ui_action,
        }

    for alias, stream in [
        ("garage", "annke_1"), ("annke", "annke_1"),
        ("exterior 1", "tapo_1"), ("ext 1", "tapo_1"), ("tapo_1", "tapo_1"), ("tapo 1", "tapo_1"),
        ("exterior 2", "tapo_2"), ("ext 2", "tapo_2"), ("tapo_2", "tapo_2"), ("tapo 2", "tapo_2"),
    ]:
        if alias in low and any(v in low for v in ("focus", "maximize", "pull up", "show", "theater", "zoom", "open")):
            ui_action = {"type": "SET_FOCUS", "stream": stream}
            return {
                "reply": f"Promoted {stream.upper()} to primary Theater Mode viewport.",
                "engine": "DETERMINISTIC_TOOL",
                "ui_action": ui_action,
            }

    # 2. Check "inspect latest" intent
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
            "reply": f"Inspected Event #{latest.id} ({latest.camera_id} - {latest.full_label}): {insp.get('ai_notes')}",
            "engine": insp.get("engine", "EDGE_FORENSICS"),
            "ui_action": {"type": "REFRESH_RAIL"},
        }

    # 3. Gather live PostgreSQL telemetry context for analytical questions
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
    text_models = [m for m in models if not any(k in m.lower() for k in ("llava", "moondream"))]

    recent_summary = "; ".join(
        [
            f"#{r.id} {r.camera_id} {r.full_label} ({int(r.confidence*100)}%) at {r.timestamp.strftime('%H:%M:%S')} {r.ai_notes or ''}"
            for r in recent_rows
        ]
    ) or "None"

    # 4. If RTX 3060 Ti Ollama has an LLM loaded, route natural language with live DB context
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
                    json={
                        "model": chosen_llm,
                        "system": sys_context,
                        "prompt": clean,
                        "stream": False,
                    },
                )
                if resp.status_code == 200:
                    ans = resp.json().get("response", "").strip()
                    if ans:
                        return {"reply": ans, "engine": f"RTX 3060 Ti // {chosen_llm}"}
        except Exception:
            pass

    # 5. Built-in SQL Telemetry Responder (Instant 0ms fallback)
    total_all = sum(counts.values())
    p_cnt = counts.get("person", 0)
    v_cnt = counts.get("vehicle", 0)
    a_cnt = counts.get("animal", 0)

    if "breach" in low or "tripwire" in low or "zone" in low:
        return {
            "reply": (
                f"PERIMETER STATUS: {len(zones)} active polygon zones configured across cameras. "
                f"{breach_count} tripwire breach events recorded in PostgreSQL."
            ),
            "engine": "SQL_TELEMETRY_ENGINE",
        }

    return {
        "reply": (
            f"TELEMETRY LOG SUMMARY: {total_all} total targets persisted "
            f"(Person: {p_cnt} | Vehicle: {v_cnt} | Animal: {a_cnt} | Breaches: {breach_count}). "
            f"Latest activity: {recent_summary}. "
            f"[Workstation RTX 3060 Ti Ollama ({WORKSTATION_GPU_URL}): {'ONLINE' if gpu_online else 'STANDBY - Using Local Edge Engine'}]"
        ),
        "engine": "SQL_TELEMETRY_ENGINE",
    }
