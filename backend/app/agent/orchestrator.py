import base64
from datetime import datetime, timezone
import json
import os
import re
import httpx
from sqlalchemy import delete, select, update
from app.db.session import AsyncSessionLocal
from app.db.models import DetectionEvent, KnownFace
from app.services.face_engine import face_engine
from app.services.delivery_rules import get_delivery_rules, record_delivery_sighting

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://192.168.1.223:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")
SNAPSHOT_DIR = "/tmp/command_snapshots"


def _encode_b64(path: str) -> str | None:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


async def deep_inspect_detection(event_id: int):
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(DetectionEvent).where(DetectionEvent.id == event_id))
        ev = res.scalar_one_or_none()
        if not ev:
            return

        snap_id = (ev.attributes or "").split("|")[0].strip()
        crop_path = f"{SNAPSHOT_DIR}/crop_{snap_id}" if snap_id else ""
        full_path = f"{SNAPSHOT_DIR}/full_{snap_id}" if snap_id else ""
        plate_path = f"{SNAPSHOT_DIR}/plate_{snap_id}" if snap_id else ""

        initial_category = ev.category
        initial_label = ev.full_label or ""
        initial_sub = ev.sub_type or ""

    # 1. Run Biometric Face Engine on person crops first (to check if it's a known Whitelist family member)
    bio_result = None
    if initial_category == "person" and os.path.exists(crop_path):
        try:
            bio_result = await face_engine.process_snapshot_biometrics(crop_path)
        except Exception as e:
            print(f"[ORCHESTRATOR BIO ERROR] {e}")

    is_whitelisted_person = bio_result and bio_result.get("role") == "whitelist"

    # 2. Query Qwen2.5-VL for Delivery/Service Logos, Landscaping Gear, Vehicle Make/Color, & License Plate OCR
    rules = get_delivery_rules()
    active_rule_names = [r["id"] for r in rules]
    rules_prompt_lines = "\n".join([f"- {r['id']}: {r.get('cues', '')}" for r in rules])

    images_b64 = []
    for p in (crop_path, plate_path, full_path):
        b64 = _encode_b64(p)
        if b64 and b64 not in images_b64:
            images_b64.append(b64)
        if len(images_b64) >= 2:
            break

    vlm_data = None
    if images_b64 and not is_whitelisted_person:
        prompt = f"""Analyze this security camera image carefully and respond ONLY with a single raw JSON object (no markdown fences).
Check for:
1. Delivery carriers or service workers matching these configured categories (by vehicle logo, uniform/vest, packages, OR landscaping tools like mowers/blowers/trimmers/trailers/work bandanas):
{rules_prompt_lines}
2. If a vehicle is visible in the driveway, identify its color + make/body type (e.g., "White Ford F-150 Pickup", "Blue Amazon Rivian Van", "Gray Toyota SUV") and read the license plate characters if legible.

Return this exact JSON schema:
{{
  "matched_service": "ONE OF {active_rule_names} OR NONE",
  "vehicle_desc": "Color Make/Type or NONE",
  "license_plate": "EXACT PLATE CHARACTERS or NONE",
  "summary": "Concise 1-sentence tactical summary of subject, clothing/gear, or vehicle"
}}"""
        try:
            async with httpx.AsyncClient(timeout=18.0) as client:
                resp = await client.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "images": images_b64,
                        "stream": False,
                        "options": {"temperature": 0.1},
                    },
                )
                if resp.status_code == 200:
                    raw_txt = resp.json().get("response", "").strip()
                    m = re.search(r"\{.*\}", raw_txt, re.DOTALL)
                    if m:
                        vlm_data = json.loads(m.group(0))
        except Exception as e:
            print(f"[VLM INSPECT SKIPPED] {e}")

    # 3. Reconcile Biometrics + Delivery/Service Rules + Driveway Vehicle LPR
    new_category = initial_category
    new_sub = initial_sub
    new_label = initial_label
    summary_note = ""

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
        now_iso = datetime.now(timezone.utc).isoformat()
        record_delivery_sighting(matched_service, now_iso)

        # If face_engine auto-enrolled a random VISITOR_XX for this delivery driver or landscaper, remove that clutter
        if bio_result and bio_result.get("auto_enrolled") and bio_result.get("name", "").startswith("VISITOR_"):
            vis_name = bio_result["name"]
            async with AsyncSessionLocal() as clean_sess:
                await clean_sess.execute(delete(KnownFace).where(KnownFace.name == vis_name))
                await clean_sess.commit()
            await face_engine.sync_gallery_from_db_and_disk()

        new_category = "delivery"
        plate_tag = f" [PLATE: {license_plate}]" if license_plate else ""
        veh_tag = f" ({vehicle_desc})" if vehicle_desc else ""

        if rule.get("alert_mode") == "silent":
            # SILENT SERVICE (e.g., LANDSCAPERS): Strip _BREACH and suppress alerts
            base_slot = initial_label.split("_")[0] + "_" + initial_label.split("_")[1] if "_" in initial_label else "Motion_1"
            new_label = f"{base_slot}_SERVICE_{matched_service}"
            new_sub = f"🔇 SERVICE: {matched_service}{veh_tag}{plate_tag}"
        else:
            # WATCHLIST DELIVERY ALERT (e.g., AMAZON, UPS, FEDEX)
            base_slot = initial_label.split("_")[0] + "_" + initial_label.split("_")[1] if "_" in initial_label else "Motion_1"
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
        bname = bio_result["name"]
        brole = bio_result["role"]
        bsightings = bio_result.get("sightings", 1)
        base_slot = initial_label.split("_")[0] + "_" + initial_label.split("_")[1] if "_" in initial_label else "Motion_1"
        breach_suffix = "_BREACH" if (initial_label.endswith("_BREACH") and brole != "whitelist") else ""
        new_label = f"{base_slot}_{bname}{breach_suffix}"
        if brole == "visitor":
            new_sub = f"ID: {bname} ({bsightings}x)"
        elif brole in ("watchlist", "watchlist_auto"):
            new_sub = f"⚠️ WATCHLIST: {bname}"
        else:
            new_sub = f"ID: {bname}"

    if vlm_summary:
        summary_note = vlm_summary

    attr_parts = [snap_id] if snap_id else []
    if summary_note:
        attr_parts.append(summary_note)
    new_attributes = " | ".join(attr_parts)

    async with AsyncSessionLocal() as session:
        await session.execute(
            update(DetectionEvent)
            .where(DetectionEvent.id == event_id)
            .values(
                category=new_category,
                sub_type=new_sub[:120],
                full_label=new_label[:120],
                attributes=new_attributes[:250],
            )
        )
        await session.commit()
