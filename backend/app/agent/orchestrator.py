import asyncio
import base64
import json
import os
import httpx
from sqlalchemy import select, func, delete
from app.db.session import AsyncSessionLocal
from app.db.models import DetectionEvent, CameraZone
from app.services.face_engine import face_engine
from app.services.push_notifier import get_push_config, send_tactical_push, CAM_NAMES

WORKSTATION_GPU_URL = os.getenv("WORKSTATION_OLLAMA_URL", "http://192.168.1.223:11434")
SNAPSHOT_DIR = "/tmp/command_snapshots"
GPU_SEMAPHORE = asyncio.Semaphore(1)


async def probe_workstation_ollama() -> tuple[bool, list[str]]:
    try:
        async with httpx.AsyncClient(timeout=2.5) as client:
            res = await client.get(f"{WORKSTATION_GPU_URL}/api/tags")
            if res.status_code == 200:
                models = [m["name"] for m in res.json().get("models", [])]
                return True, models
    except Exception:
        pass
    return False, []


def pick_best_vlm(models: list[str]) -> str | None:
    for preferred in ("qwen2.5vl", "qwen2-vl", "minicpm-v", "llama3.2-vision", "llava"):
        for m in models:
            if preferred in m.lower():
                return m
    return None


def extract_json_dict(raw_text: str) -> dict:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(raw_text[start : end + 1])
    return json.loads(raw_text)


async def deep_inspect_detection(event_id: int) -> dict:
    async with GPU_SEMAPHORE:
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(DetectionEvent).where(DetectionEvent.id == event_id))
            ev = result.scalar_one_or_none()
            if not ev:
                return {"error": "Event not found"}

            crop_path = f"{SNAPSHOT_DIR}/crop_{ev.attributes}" if ev.attributes else ""
            recognized_face = None
            face_role = None

            if ev.category == "person" and os.path.exists(crop_path):
                face_info = await face_engine.process_snapshot_biometrics(crop_path)
                if face_info:
                    fname = face_info["name"]
                    face_role = face_info["role"]
                    fsightings = face_info["sightings"]
                    if face_role == "visitor":
                        recognized_face = f"{fname} (Seen {fsightings}x)"
                        ev.sub_type = f"ID: {fname} ({fsightings}x)"
                    elif face_role in ("watchlist", "watchlist_auto"):
                        recognized_face = f"⚠️ WATCHLIST: {fname} (Seen {fsightings}x)"
                        ev.sub_type = f"⚠️ WATCHLIST: {fname}"
                        ev.full_label = f"{ev.motion_slot}_{fname}_BREACH"
                    else:
                        recognized_face = f"{fname} (Seen {fsightings}x)"
                        ev.sub_type = f"ID: {fname}"
                        ev.full_label = f"{ev.motion_slot}_{fname}"

            gpu_online, models = await probe_workstation_ollama()
            chosen_vlm = pick_best_vlm(models) if gpu_online else None
            is_breach = "BREACH" in (ev.full_label or "")
            is_watchlist = face_role in ("watchlist", "watchlist_auto") or "WATCHLIST" in (ev.sub_type or "")
            breach_prefix = "🚨 [BREACH] " if is_breach else ""
            face_prefix = f"🧬 [{recognized_face}] " if recognized_face else ""
            detected_carrier = "NONE"
            detected_plate = "NONE"

            if chosen_vlm and os.path.exists(crop_path):
                try:
                    with open(crop_path, "rb") as f:
                        b64_img = base64.b64encode(f.read()).decode("utf-8")

                    prompt = (
                        f"Analyze this security camera crop from '{ev.camera_id}' (detector guess: {ev.category}). "
                        "Note: grayscale images are infrared night vision on dry pavement (never water/flooding). "
                        "Cool-blue or bright white headlights are normal aftermarket LED headlights—NEVER call a car an emergency/police vehicle unless roof lightbars and agency decals are clearly visible. "
                        "If this is a vehicle, read any legible license plate characters into 'plate' (or 'NONE') and check for delivery logos (AMAZON, UPS, FEDEX, USPS, DHL, or NONE) in 'carrier'. "
                        "If this is an inanimate object (trash bin, mailbox, shadow), set valid_target to false and category to false_alarm. "
                        "Respond ONLY with valid JSON: "
                        '{"valid_target": true, "category": "person|vehicle|animal|false_alarm", '
                        '"sub_type": "short 1-3 word label like white suv, sedan, or person", '
                        '"plate": "ABC1234 or NONE", '
                        '"carrier": "AMAZON|UPS|FEDEX|USPS|NONE", '
                        '"summary": "Concise 12-20 word description of clothing/paint color, vehicle type, and action"}'
                    )

                    async with httpx.AsyncClient(timeout=35.0) as client:
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
                            parsed = extract_json_dict(raw_text)
                            valid = bool(parsed.get("valid_target", True))
                            vlm_cat = str(parsed.get("category", ev.category)).lower()
                            vlm_sub = str(parsed.get("sub_type", ev.sub_type)).strip().lower()
                            detected_plate = str(parsed.get("plate", "NONE")).strip().upper()
                            detected_carrier = str(parsed.get("carrier", "NONE")).strip().upper()
                            summary = str(parsed.get("summary", "")).strip()

                            if (not valid or vlm_cat == "false_alarm") and not recognized_face:
                                await session.execute(delete(DetectionEvent).where(DetectionEvent.id == ev.id))
                                await session.commit()
                                return {"id": ev.id, "purged": True, "ai_notes": "Purged inanimate false positive"}

                            if vlm_cat in ("person", "vehicle", "animal") and not recognized_face:
                                ev.category = vlm_cat
                                suffix = "_BREACH" if is_breach else ""
                                ev.full_label = f"{ev.motion_slot}_{vlm_cat}{suffix}"

                            if not recognized_face:
                                alpr_tags = []
                                if detected_carrier not in ("", "NONE", "NULL", "N/A"):
                                    alpr_tags.append(f"📦 {detected_carrier}")
                                if detected_plate not in ("", "NONE", "NULL", "N/A", "UNKNOWN") and len(detected_plate) >= 3:
                                    alpr_tags.append(f"🔢 {detected_plate}")

                                if alpr_tags:
                                    ev.sub_type = " · ".join(alpr_tags)[:48]
                                elif vlm_sub and len(vlm_sub) <= 32:
                                    ev.sub_type = vlm_sub

                            alpr_note = ""
                            if detected_carrier not in ("", "NONE", "NULL"):
                                alpr_note += f"[CARRIER: {detected_carrier}] "
                            if detected_plate not in ("", "NONE", "NULL", "UNKNOWN") and len(detected_plate) >= 3:
                                alpr_note += f"[PLATE: {detected_plate}] "

                            ev.ai_notes = f"{breach_prefix}{face_prefix}{alpr_note}[RTX 3060 Ti // {chosen_vlm}] {summary}"
                            await session.commit()
                except Exception as e:
                    print(f"[VLM ERROR] {e}")

            if not ev.ai_notes:
                ev.ai_notes = (
                    f"{breach_prefix}{face_prefix}[YOLOv8s EDGE] {ev.sub_type.upper()} ({int(ev.confidence * 100)}% conf)"
                )
                await session.commit()

            # Trigger Instant Mobile Push Notification if criteria met (NEVER push for Whitelisted family!)
            if face_role != "whitelist":
                push_cfg = await get_push_config()
                cam_friendly = CAM_NAMES.get(ev.camera_id, ev.camera_id.upper())
                is_delivery = detected_carrier not in ("", "NONE", "NULL", "N/A")

                if is_watchlist and push_cfg.notify_watchlist:
                    await send_tactical_push(
                        title=f"🚨 WATCHLIST MATCH // {cam_friendly}",
                        message=ev.ai_notes,
                        crop_path=crop_path,
                        priority="urgent",
                        tags="rotating_light,skull",
                    )
                elif is_breach and push_cfg.notify_breach:
                    await send_tactical_push(
                        title=f"⚠️ TRIPWIRE BREACH // {cam_friendly}",
                        message=ev.ai_notes,
                        crop_path=crop_path,
                        priority="high",
                        tags="warning,shield",
                    )
                elif is_delivery and push_cfg.notify_delivery:
                    await send_tactical_push(
                        title=f"📦 {detected_carrier} DELIVERY // {cam_friendly}",
                        message=ev.ai_notes,
                        crop_path=crop_path,
                        priority="default",
                        tags="package,truck",
                    )

            return {
                "id": ev.id,
                "sub_type": ev.sub_type,
                "full_label": ev.full_label,
                "ai_notes": ev.ai_notes,
                "engine": f"RTX 3060 Ti // {chosen_vlm}" if chosen_vlm else "EDGE_FALLBACK",
            }


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
        ("driveway", "annke_1"), ("garage", "annke_1"), ("annke", "annke_1"),
        ("loft", "tapo_1"), ("tapo_1", "tapo_1"), ("tapo 1", "tapo_1"),
        ("living", "tapo_2"), ("living_room", "tapo_2"), ("tapo_2", "tapo_2"), ("tapo 2", "tapo_2"),
    ]:
        if alias in low and any(v in low for v in ("focus", "maximize", "pull up", "show", "theater", "zoom", "open")):
            return {
                "reply": f"Promoted {alias.upper()} ({stream}) to primary Theater Mode viewport.",
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
            "Cameras: DRIVEWAY (annke_1), LOFT (tapo_1), LIVING_ROOM (tapo_2). "
            f"Enrolled Face Vectors: {len(face_engine.gallery)}. "
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
            f"TELEMETRY SUMMARY: {total_all} verified events | Enrolled Faces: {len(face_engine.gallery)} "
            f"(Person: {counts.get('person', 0)} | Vehicle: {counts.get('vehicle', 0)} | Animal: {counts.get('animal', 0)} | Breaches: {breach_count}). "
            f"Recent: {recent_summary}."
        ),
        "engine": "SQL_TELEMETRY_ENGINE",
    }

# --- v7 Override: Delivery/Service Logo Rules + Silent Landscapers + Driveway LPR ---
import base64 as _b64
from datetime import datetime as _dt, timezone as _tz
import json as _json
import os as _os
import re as _re
import httpx as _httpx
from sqlalchemy import delete as _delete, select as _select, update as _update
from app.db.session import AsyncSessionLocal as _AsyncSessionLocal
from app.db.models import DetectionEvent as _DetectionEvent, KnownFace as _KnownFace
from app.services.face_engine import face_engine as _face_engine
from app.services.delivery_rules import get_delivery_rules as _get_delivery_rules, record_delivery_sighting as _record_delivery_sighting

_V7_OLLAMA_URL = _os.getenv("OLLAMA_BASE_URL", "http://192.168.1.223:11434")
_V7_OLLAMA_MODEL = _os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")
_V7_SNAP_DIR = "/tmp/command_snapshots"

def _v7_encode_b64(path: str):
    if not path or not _os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return _b64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None

async def deep_inspect_detection(event_id: int):
    async with _AsyncSessionLocal() as session:
        res = await session.execute(_select(_DetectionEvent).where(_DetectionEvent.id == event_id))
        ev = res.scalar_one_or_none()
        if not ev:
            return
        snap_id = (ev.attributes or "").split("|")[0].strip()
        crop_path = f"{_V7_SNAP_DIR}/crop_{snap_id}" if snap_id else ""
        full_path = f"{_V7_SNAP_DIR}/full_{snap_id}" if snap_id else ""
        plate_path = f"{_V7_SNAP_DIR}/plate_{snap_id}" if snap_id else ""
        initial_category = ev.category
        initial_label = ev.full_label or ""
        initial_sub = ev.sub_type or ""

    bio_result = None
    if initial_category == "person" and _os.path.exists(crop_path):
        try:
            bio_result = await _face_engine.process_snapshot_biometrics(crop_path)
        except Exception as e:
            print(f"[ORCHESTRATOR BIO ERROR] {e}")

    is_whitelisted_person = bio_result and bio_result.get("role") == "whitelist"
    rules = _get_delivery_rules()
    active_rule_names = [r["id"] for r in rules]
    rules_prompt_lines = "\n".join([f"- {r['id']}: {r.get('cues', '')}" for r in rules])

    images_b64 = []
    for p in (crop_path, plate_path, full_path):
        b64 = _v7_encode_b64(p)
        if b64 and b64 not in images_b64:
            images_b64.append(b64)
        if len(images_b64) >= 2:
            break

    vlm_data = None
    if images_b64 and not is_whitelisted_person:
        prompt = f"""Analyze this security camera image and respond ONLY with a raw JSON object:
1. Check for delivery carriers or service crews matching:
{rules_prompt_lines}
2. If a vehicle is in the driveway, identify its Color + Make/Body Type and read its license plate characters if visible.
JSON schema:
{{"matched_service": "ONE OF {active_rule_names} OR NONE", "vehicle_desc": "Color Make/Type or NONE", "license_plate": "PLATE or NONE", "summary": "1-sentence tactical summary"}}"""
        try:
            async with _httpx.AsyncClient(timeout=18.0) as client:
                resp = await client.post(
                    f"{_V7_OLLAMA_URL}/api/generate",
                    json={"model": _V7_OLLAMA_MODEL, "prompt": prompt, "images": images_b64, "stream": False, "options": {"temperature": 0.1}},
                )
                if resp.status_code == 200:
                    raw_txt = resp.json().get("response", "").strip()
                    m = _re.search(r"\{.*\}", raw_txt, _re.DOTALL)
                    if m:
                        vlm_data = _json.loads(m.group(0))
        except Exception as e:
            print(f"[VLM INSPECT SKIPPED] {e}")

    new_category, new_sub, new_label = initial_category, initial_sub, initial_label
    matched_service = (vlm_data.get("matched_service") or "NONE").strip().upper() if vlm_data else "NONE"
    vehicle_desc = (vlm_data.get("vehicle_desc") or "NONE").strip() if vlm_data else "NONE"
    license_plate = (vlm_data.get("license_plate") or "NONE").strip().upper() if vlm_data else "NONE"
    vlm_summary = (vlm_data.get("summary") or "").strip() if vlm_data else ""

    if license_plate in ("NONE", "N/A", "UNKNOWN", "UNREADABLE", "NULL", ""):
        license_plate = ""
    if vehicle_desc in ("NONE", "N/A", "UNKNOWN", "NULL", ""):
        vehicle_desc = ""

    rule_map = {r["id"]: r for r in rules}
    if matched_service in rule_map and not is_whitelisted_person:
        rule = rule_map[matched_service]
        _record_delivery_sighting(matched_service, _dt.now(_tz.utc).isoformat())
        if bio_result and bio_result.get("auto_enrolled") and bio_result.get("name", "").startswith("VISITOR_"):
            async with _AsyncSessionLocal() as clean_sess:
                await clean_sess.execute(_delete(_KnownFace).where(_KnownFace.name == bio_result["name"]))
                await clean_sess.commit()
            await _face_engine.sync_gallery_from_db_and_disk()

        new_category = "delivery"
        plate_tag = f" [PLATE: {license_plate}]" if license_plate else ""
        veh_tag = f" ({vehicle_desc})" if vehicle_desc else ""
        base_slot = initial_label.split("_")[0] + "_" + initial_label.split("_")[1] if "_" in initial_label else "Motion_1"
        if rule.get("alert_mode") == "silent":
            new_label = f"{base_slot}_SERVICE_{matched_service}"
            new_sub = f"🔇 SERVICE: {matched_service}{veh_tag}{plate_tag}"
        else:
            breach_suffix = "_BREACH" if initial_label.endswith("_BREACH") else ""
            new_label = f"{base_slot}_DELIVERY_{matched_service}{breach_suffix}"
            new_sub = f"📦 DELIVERY: {matched_service}{veh_tag}{plate_tag}"
    elif initial_category == "vehicle":
        plate_tag = f" [PLATE: {license_plate}]" if license_plate else " [NO PLATE VIEW]"
        desc_label = vehicle_desc if vehicle_desc else initial_sub
        new_sub = f"🚗 {desc_label}{plate_tag}"
        if license_plate:
            base_slot = initial_label.replace("_BREACH", "")
            breach_suffix = "_BREACH" if initial_label.endswith("_BREACH") else ""
            new_label = f"{base_slot}_{license_plate.replace(' ', '')}{breach_suffix}"
    elif bio_result and bio_result.get("name"):
        bname, brole, bsightings = bio_result["name"], bio_result["role"], bio_result.get("sightings", 1)
        base_slot = initial_label.split("_")[0] + "_" + initial_label.split("_")[1] if "_" in initial_label else "Motion_1"
        breach_suffix = "_BREACH" if (initial_label.endswith("_BREACH") and brole != "whitelist") else ""
        new_label = f"{base_slot}_{bname}{breach_suffix}"
        new_sub = f"ID: {bname} ({bsightings}x)" if brole == "visitor" else (f"⚠️ WATCHLIST: {bname}" if brole in ("watchlist", "watchlist_auto") else f"ID: {bname}")

    attr_parts = [snap_id] if snap_id else []
    if vlm_summary:
        attr_parts.append(vlm_summary)

    async with _AsyncSessionLocal() as session:
        await session.execute(
            _update(_DetectionEvent)
            .where(_DetectionEvent.id == event_id)
            .values(category=new_category, sub_type=new_sub[:120], full_label=new_label[:120], attributes=" | ".join(attr_parts)[:250])
        )
        await session.commit()
