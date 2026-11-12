import asyncio
import base64
import json
import os
import time
from datetime import datetime as _dt, timedelta as _td, timezone as _tz
import httpx
from sqlalchemy import select, func, delete
from app.db.session import AsyncSessionLocal
from app.db.models import DetectionEvent, CameraZone, KnownFace as _KnownFace, CameraSchedule as _CameraSchedule
from app.services.face_engine import face_engine
from app.services.push_notifier import get_push_config, send_tactical_push, CAM_NAMES
from app.services.delivery_rules import get_delivery_rules as _get_delivery_rules, record_delivery_sighting as _record_delivery_sighting
from app.services.ai_lessons import get_ai_lessons as _get_ai_lessons

WORKSTATION_GPU_URL = os.getenv("WORKSTATION_OLLAMA_URL", "http://192.168.1.223:11434")
SNAPSHOT_DIR = "/tmp/command_snapshots"
GPU_SEMAPHORE = asyncio.Semaphore(1)

_GPU_CACHE: dict = {"checked_at": 0.0, "online": False, "models": []}


async def probe_workstation_ollama(force: bool = False) -> tuple[bool, list[str]]:
    now = time.time()
    ttl = 60.0 if _GPU_CACHE["online"] else 30.0
    if not force and (now - _GPU_CACHE["checked_at"]) < ttl:
        return _GPU_CACHE["online"], list(_GPU_CACHE["models"])

    try:
        async with httpx.AsyncClient(timeout=2.2) as client:
            res = await client.get(f"{WORKSTATION_GPU_URL}/api/tags")
            if res.status_code == 200:
                models = [m["name"] for m in res.json().get("models", [])]
                _GPU_CACHE.update({"checked_at": now, "online": True, "models": models})
                return True, models
    except Exception:
        pass

    _GPU_CACHE.update({"checked_at": now, "online": False, "models": []})
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


def _is_isaac_event(sub_type: str | None, full_label: str | None, ai_notes: str | None = None) -> bool:
    u_sub = (sub_type or "").upper()
    u_lbl = (full_label or "").upper()
    u_not = (ai_notes or "").upper()
    return (
        "ID: ISAAC" in u_sub
        or u_lbl.endswith("_ISAAC")
        or "_ISAAC_" in u_lbl
        or "[ISAAC (" in u_not
        or "MATCH: ISAAC" in u_not
    )


def _is_whitelist_family_event(sub_type: str | None, full_label: str | None, face_role: str | None = None) -> bool:
    u_sub = (sub_type or "").upper()
    if "WATCHLIST" in u_sub or "VISITOR_" in u_sub:
        return False
    if face_role == "whitelist" or u_sub.startswith("ID: "):
        return True
    return False


async def _evaluate_best_confidence_push(
    event_id: int,
    face_role: str | None = None,
    recognized_face: str | None = None,
):
    """
    Waits briefly (4.5s) so the 5-second MP4 replay clip finishes encoding and any sibling
    detections from the same person/vehicle walking closer are committed. Then selects ONLY
    the single highest-confidence detection in the encounter window (±30s) and enforces a 90s
    per-camera/category cooldown so the user only receives 1 alert with the best confidence.
    """
    try:
        await asyncio.sleep(4.5)
        wait_ticks = 0
        while GPU_SEMAPHORE.locked() and wait_ticks < 24:
            await asyncio.sleep(0.5)
            wait_ticks += 1

        async with AsyncSessionLocal() as session:
            ev = await session.get(DetectionEvent, event_id)
            if not ev:
                return

            cam_sched = await session.get(_CameraSchedule, ev.camera_id)
            cam_alert_all = bool(cam_sched and getattr(cam_sched, "alert_whitelist", False))
            push_cfg = await get_push_config()

            win_start = ev.timestamp - _td(seconds=30)
            win_end = ev.timestamp + _td(seconds=30)
            burst_res = await session.execute(
                select(DetectionEvent).where(
                    DetectionEvent.camera_id == ev.camera_id,
                    DetectionEvent.timestamp >= win_start,
                    DetectionEvent.timestamp <= win_end,
                )
            )
            burst_events = burst_res.scalars().all()

        now_ts = time.time()
        is_away = bool(getattr(push_cfg, "away_mode", False))
        away_armed_at = float(getattr(push_cfg, "away_armed_at", 0.0) or 0.0)
        in_away_grace = is_away and (now_ts < away_armed_at)
        away_active = is_away and not in_away_grace

        # 1. Permanent ISAAC & Silent Service exemption across ±30s encounter window
        for b in burst_events:
            if _is_isaac_event(b.sub_type, b.full_label, b.ai_notes) or (b.sub_type or "").startswith("🔇 SERVICE:"):
                return

        if _is_isaac_event(ev.sub_type, ev.full_label, ev.ai_notes) or (ev.sub_type or "").startswith("🔇 SERVICE:"):
            return

        # 2. Shield whitelisted family during normal Home Mode (when FAM·ALERT is OFF) OR during 3m Away exit grace
        if (not away_active) and (not cam_alert_all) and ev.category == "person":
            if any(_is_whitelist_family_event(b.sub_type, b.full_label) for b in burst_events):
                return

        def _cat_group(c: str | None) -> str:
            return "person_or_delivery" if c in ("person", "delivery") else (c or "other")

        my_group = _cat_group(ev.category)
        is_indoor_cam = ev.camera_id in ("tapo_1", "tapo_2")

        def _qualifies_and_score(cand: DetectionEvent) -> tuple[bool, tuple]:
            c_sub = cand.sub_type or ""
            c_lbl = cand.full_label or ""
            if _is_isaac_event(c_sub, c_lbl, cand.ai_notes) or c_sub.startswith("🔇 SERVICE:"):
                return (False, (0, 0, 0, 0.0, 0))

            c_watch = "WATCHLIST" in c_sub or (
                "watchlist" if cand.id == ev.id and face_role in ("watchlist", "watchlist_auto") else ""
            ) != ""
            c_breach = "BREACH" in c_lbl
            c_deliv = cand.category == "delivery" and c_sub.startswith("📦 DELIVERY:")
            c_fam = _is_whitelist_family_event(c_sub, c_lbl, face_role if cand.id == ev.id else None)
            c_indoor_stranger = is_indoor_cam and cand.category == "person" and not c_fam

            qualifies = False
            if c_watch and (push_cfg.notify_watchlist or cam_alert_all or away_active):
                qualifies = True
            elif away_active and cand.category in ("person", "delivery", "vehicle"):
                # Whole household is away: alert on ANY person, delivery, or vehicle on ANY camera
                qualifies = True
            elif cam_alert_all:
                qualifies = True
            elif not c_fam:
                if (
                    ((c_breach or c_indoor_stranger) and push_cfg.notify_breach)
                    or (c_deliv and push_cfg.notify_delivery)
                ):
                    qualifies = True

            has_id_or_plate = 1 if (c_sub.startswith("ID: ") or "[PLATE:" in c_sub or c_deliv) else 0
            score = (
                1 if c_watch else 0,
                1 if (c_breach or c_indoor_stranger or c_deliv) else 0,
                has_id_or_plate,
                round(float(cand.confidence or 0.0), 4),
                cand.id,
            )
            return (qualifies, score)

        my_qualifies, my_score = _qualifies_and_score(ev)
        if not my_qualifies:
            return

        # 3. Only the single highest-scoring/highest-confidence event in the ±30s burst sends the alert
        for sibling in burst_events:
            if sibling.id == ev.id or _cat_group(sibling.category) != my_group:
                continue
            sib_qualifies, sib_score = _qualifies_and_score(sibling)
            if sib_qualifies and sib_score > my_score:
                return

        snap_id = (ev.attributes or "").split("|")[0].strip()
        crop_path = f"{SNAPSHOT_DIR}/crop_{snap_id}" if snap_id else None
        full_path = f"{SNAPSHOT_DIR}/full_{snap_id}" if snap_id else None
        plate_path = f"{SNAPSHOT_DIR}/plate_{snap_id}" if snap_id else None
        clip_path = f"{SNAPSHOT_DIR}/clip_{snap_id}.mp4" if snap_id else None
        ai_notes_out = ev.ai_notes or (ev.attributes.split("|", 1)[1].strip() if "|" in (ev.attributes or "") else "")

        cam_friendly = CAM_NAMES.get(ev.camera_id, ev.camera_id.upper())
        is_watchlist = face_role in ("watchlist", "watchlist_auto") or "WATCHLIST" in (ev.sub_type or "")
        is_breach = "BREACH" in (ev.full_label or "")
        is_delivery_alert = ev.category == "delivery" and (ev.sub_type or "").startswith("📦 DELIVERY:")
        is_family_whitelist = _is_whitelist_family_event(ev.sub_type, ev.full_label, face_role)
        is_indoor_stranger = is_indoor_cam and ev.category == "person" and not is_family_whitelist
        dedup_key = f"{ev.camera_id}:{my_group}"

        if is_watchlist and (push_cfg.notify_watchlist or cam_alert_all or away_active):
            await send_tactical_push(
                title=f"🚨 WATCHLIST MATCH // {cam_friendly}",
                message=ai_notes_out or ev.full_label,
                crop_path=crop_path,
                full_path=full_path,
                plate_path=plate_path,
                clip_path=clip_path,
                confidence=ev.confidence,
                priority="urgent",
                tags="rotating_light,skull",
                dedup_key=dedup_key,
            )
        elif away_active:
            if is_delivery_alert:
                title_str = f"🛫 {ev.sub_type} // {cam_friendly}"
                prio_str = "high"
                tag_str = "package,airplane"
            elif is_breach or is_indoor_cam:
                title_str = f"🚨 AWAY INTRUSION: {(ev.sub_type or ev.category).upper()} // {cam_friendly}"
                prio_str = "urgent"
                tag_str = "rotating_light,shield"
            else:
                title_str = f"🛫 AWAY ALERT: {(ev.sub_type or ev.category).upper()} // {cam_friendly}"
                prio_str = "high"
                tag_str = "shield,airplane"
            await send_tactical_push(
                title=title_str,
                message=ai_notes_out or ev.full_label,
                crop_path=crop_path,
                full_path=full_path,
                plate_path=plate_path,
                clip_path=clip_path,
                confidence=ev.confidence,
                priority=prio_str,
                tags=tag_str,
                dedup_key=dedup_key,
            )
        elif cam_alert_all:
            if is_family_whitelist:
                fam_name = (ev.sub_type or "").replace("ID:", "").strip() or (recognized_face or "FAMILY")
                await send_tactical_push(
                    title=f"🏠 FAMILY DETECTED: {fam_name} // {cam_friendly}",
                    message=ai_notes_out or ev.full_label,
                    crop_path=crop_path,
                    full_path=full_path,
                    plate_path=plate_path,
                    clip_path=clip_path,
                    confidence=ev.confidence,
                    priority="high",
                    tags="house,bell",
                    dedup_key=f"{ev.camera_id}:family:{fam_name}",
                )
            elif is_breach or is_indoor_stranger:
                await send_tactical_push(
                    title=f"⚠️ {'INDOOR INTRUSION' if is_indoor_stranger and not is_breach else 'TRIPWIRE BREACH'} // {cam_friendly}",
                    message=ai_notes_out or ev.full_label,
                    crop_path=crop_path,
                    full_path=full_path,
                    plate_path=plate_path,
                    clip_path=clip_path,
                    confidence=ev.confidence,
                    priority="high",
                    tags="warning,shield",
                    dedup_key=dedup_key,
                )
            elif is_delivery_alert:
                await send_tactical_push(
                    title=f"{ev.sub_type} // {cam_friendly}",
                    message=ai_notes_out or ev.full_label,
                    crop_path=crop_path,
                    full_path=full_path,
                    plate_path=plate_path,
                    clip_path=clip_path,
                    confidence=ev.confidence,
                    priority="default",
                    tags="package,truck",
                    dedup_key=dedup_key,
                )
            else:
                await send_tactical_push(
                    title=f"🔔 {(ev.sub_type or ev.category).upper()} // {cam_friendly}",
                    message=ai_notes_out or ev.full_label,
                    crop_path=crop_path,
                    full_path=full_path,
                    plate_path=plate_path,
                    clip_path=clip_path,
                    confidence=ev.confidence,
                    priority="default",
                    tags="bell,eye",
                    dedup_key=dedup_key,
                )
        elif not is_family_whitelist:
            if (is_breach or is_indoor_stranger) and push_cfg.notify_breach:
                await send_tactical_push(
                    title=f"⚠️ {'INDOOR INTRUSION' if is_indoor_stranger and not is_breach else 'TRIPWIRE BREACH'} // {cam_friendly}",
                    message=ai_notes_out or ev.full_label,
                    crop_path=crop_path,
                    full_path=full_path,
                    plate_path=plate_path,
                    clip_path=clip_path,
                    confidence=ev.confidence,
                    priority="high",
                    tags="warning,shield",
                    dedup_key=dedup_key,
                )
            elif is_delivery_alert and push_cfg.notify_delivery:
                await send_tactical_push(
                    title=f"{ev.sub_type} // {cam_friendly}",
                    message=ai_notes_out or ev.full_label,
                    crop_path=crop_path,
                    full_path=full_path,
                    plate_path=plate_path,
                    clip_path=clip_path,
                    confidence=ev.confidence,
                    priority="default",
                    tags="package,truck",
                    dedup_key=dedup_key,
                )
    except Exception as e:
        print(f"[PUSH DISPATCH ERROR] {e}")


async def deep_inspect_detection(event_id: int) -> dict:
    async with GPU_SEMAPHORE:
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(DetectionEvent).where(DetectionEvent.id == event_id))
            ev = result.scalar_one_or_none()
            if not ev:
                return {"error": "Event not found"}

            raw_attr = ev.attributes or ""
            snap_id = raw_attr.split("|")[0].strip()
            existing_note = raw_attr.split("|", 1)[1].strip() if "|" in raw_attr else ""

            crop_path = f"{SNAPSHOT_DIR}/crop_{snap_id}" if snap_id else ""
            full_path = f"{SNAPSHOT_DIR}/full_{snap_id}" if snap_id else ""
            plate_path = f"{SNAPSHOT_DIR}/plate_{snap_id}" if snap_id else ""
            clip_path = f"{SNAPSHOT_DIR}/clip_{snap_id}.mp4" if snap_id else ""

            recognized_face = None
            face_role = None
            face_info = None

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
                elif (ev.sub_type or "").startswith("ID: "):
                    pre_name = (ev.sub_type or "").replace("ID:", "").strip()
                    recognized_face = pre_name
                    face_role = "visitor" if pre_name.startswith("VISITOR_") else "whitelist"

            gpu_online, models = await probe_workstation_ollama()
            chosen_vlm = pick_best_vlm(models) if gpu_online else None

            rules = _get_delivery_rules()
            active_rule_names = [r["id"] for r in rules]
            rules_text = "; ".join([f"{r['id']} ({r.get('cues', '')})" for r in rules])

            lessons = _get_ai_lessons(ev.camera_id)
            lessons_text = "; ".join([
                f"If detector guessed '{L['wrong_sub_type']}', check if it is actually '{L['correct_sub_type']}' ({L['note']})"
                for L in lessons[-8:]
            ])

            img_to_send = (
                full_path
                if (ev.category == "animal" and os.path.exists(full_path))
                else (crop_path if os.path.exists(crop_path) else full_path)
            )
            ai_notes_out = existing_note

            if chosen_vlm and img_to_send and os.path.exists(img_to_send):
                try:
                    with open(img_to_send, "rb") as f:
                        images_b64 = [base64.b64encode(f.read()).decode("utf-8")]
                    if ev.category == "vehicle" and plate_path and os.path.exists(plate_path):
                        with open(plate_path, "rb") as pf:
                            images_b64.append(base64.b64encode(pf.read()).decode("utf-8"))

                    prompt = (
                        f"Analyze this security camera image from '{ev.camera_id}' (initial detector guess: category={ev.category}, sub_type={ev.sub_type}). "
                        f"USER TAUGHT RULES FOR THIS CAMERA: {lessons_text or 'None'}. "
                        f"CONFIGURED DELIVERY/SERVICE RULES: {rules_text}. "
                        "Note: grayscale images are infrared night vision on dry pavement. "
                        "If this is purely an inanimate false trigger (empty pavement, wind-blown tree/shadow, IR spiderweb, trash bin, or headlight glare with NO actual person, vehicle, or animal visible), set 'valid_target' to false and 'category' to 'false_alarm'. "
                        "If a person is walking an animal on a sidewalk/leash, classify sub_type as 'dog' and mention the owner/leash in summary. "
                        "If a small four-legged animal is alone on the driveway/pavement with NO person or leash visible, classify sub_type as 'cat'. "
                        "If a vehicle is visible, inspect both the vehicle and bumper crop to read any legible license plate characters into 'plate' (or 'NONE'). "
                        f"If a delivery carrier or service crew (like LANDSCAPERS with mowers/blowers/vests/bandanas or AMAZON/UPS/FEDEX/USPS) is visible, set 'carrier' to one of {active_rule_names} (or 'NONE') and note if a package was left. "
                        "Respond ONLY with valid JSON: "
                        '{"valid_target": true, "category": "person|delivery|vehicle|animal|false_alarm", '
                        '"sub_type": "short label like cat, dog, white suv, or person", '
                        '"plate": "ABC1234 or NONE", '
                        f'"carrier": "{"|".join(active_rule_names)}|NONE", '
                        '"summary": "Concise 12-20 word tactical description of subject, clothing/gear, package, or vehicle"}'
                    )

                    async with httpx.AsyncClient(timeout=35.0) as client:
                        resp = await client.post(
                            f"{WORKSTATION_GPU_URL}/api/generate",
                            json={
                                "model": chosen_vlm,
                                "prompt": prompt,
                                "images": images_b64,
                                "format": "json",
                                "stream": False,
                                "options": {"temperature": 0.1},
                            },
                        )
                        if resp.status_code == 200:
                            parsed = extract_json_dict(resp.json().get("response", "{}"))
                            valid_target = bool(parsed.get("valid_target", True))
                            vlm_cat = str(parsed.get("category") or ev.category).strip().lower()
                            summary = (parsed.get("summary") or "").strip()
                            carrier = (parsed.get("carrier") or "NONE").strip().upper()
                            plate = (parsed.get("plate") or "NONE").strip().upper()
                            vlm_sub = (parsed.get("sub_type") or "").strip().lower()

                            # Purge confirmed inanimate VLM false alarms before any alert can fire
                            if (not valid_target or vlm_cat == "false_alarm") and not recognized_face:
                                if face_info and face_info.get("auto_enrolled") and str(face_info.get("name", "")).startswith("VISITOR_"):
                                    await session.execute(delete(_KnownFace).where(_KnownFace.name == face_info["name"]))
                                await session.execute(delete(DetectionEvent).where(DetectionEvent.id == ev.id))
                                await session.commit()
                                for fp in (crop_path, full_path, plate_path, clip_path):
                                    if fp and os.path.exists(fp):
                                        try:
                                            os.remove(fp)
                                        except OSError:
                                            pass
                                return {"id": ev.id, "purged": True, "ai_notes": "Purged inanimate VLM false alarm"}

                            if vlm_cat in ("person", "vehicle", "animal") and vlm_cat != ev.category and not recognized_face:
                                ev.category = vlm_cat

                            if plate in ("NONE", "N/A", "UNKNOWN", "NULL", ""):
                                plate = ""

                            rule_map = {r["id"]: r for r in rules}
                            if carrier in rule_map and face_role != "whitelist":
                                rule = rule_map[carrier]
                                _record_delivery_sighting(carrier, _dt.now(_tz.utc).isoformat())
                                if face_info and face_info.get("auto_enrolled") and str(face_info.get("name", "")).startswith("VISITOR_"):
                                    await session.execute(delete(_KnownFace).where(_KnownFace.name == face_info["name"]))
                                ev.category = "delivery"
                                plate_str = f" [PLATE: {plate}]" if plate else ""
                                if rule.get("alert_mode") == "silent":
                                    ev.sub_type = f"🔇 SERVICE: {carrier}{plate_str}"
                                    ev.full_label = f"{ev.motion_slot}_SERVICE_{carrier}"
                                else:
                                    ev.sub_type = f"📦 DELIVERY: {carrier}{plate_str}"
                                    ev.full_label = f"{ev.motion_slot}_DELIVERY_{carrier}"
                            elif ev.category == "animal":
                                if "cat" in vlm_sub or "cat" in summary.lower() or "feline" in summary.lower():
                                    ev.sub_type = "cat"
                                    ev.full_label = f"{ev.motion_slot}_cat"
                                elif "dog" in vlm_sub or "dog" in summary.lower() or "leash" in summary.lower():
                                    ev.sub_type = "dog"
                                    ev.full_label = f"{ev.motion_slot}_dog"
                                elif vlm_sub and vlm_sub not in ("none", "animal", "unknown"):
                                    ev.sub_type = vlm_sub[:32]
                                    ev.full_label = f"{ev.motion_slot}_{vlm_sub.replace(' ', '_')[:24]}"
                            elif ev.category == "vehicle":
                                plate_str = f" [PLATE: {plate}]" if plate else " [NO PLATE VIEW]"
                                desc = vlm_sub.upper() if vlm_sub and vlm_sub != "none" else (ev.sub_type or "VEHICLE").upper()
                                ev.sub_type = f"🚗 {desc}{plate_str}"[:120]
                                if plate:
                                    ev.full_label = f"{ev.motion_slot}_{plate.replace(' ', '')}"[:120]

                            face_prefix = f"🧬 [{recognized_face}] " if recognized_face else ""
                            plate_prefix = f"[PLATE: {plate}] " if plate else ""
                            if summary:
                                ai_notes_out = f"{face_prefix}{plate_prefix}[RTX 3060 Ti // {chosen_vlm}] {summary}"
                except Exception as e:
                    print(f"[DEEP INSPECT VLM ERROR] {e}")

            upper_sub = (ev.sub_type or "").upper()
            upper_lbl = (ev.full_label or "").upper()
            upper_face = (recognized_face or "").upper()
            is_isaac = (
                "ISAAC" in upper_face
                or "ID: ISAAC" in upper_sub
                or upper_lbl.endswith("_ISAAC")
                or "_ISAAC_" in upper_lbl
            )
            if is_isaac and "_BREACH" in (ev.full_label or ""):
                ev.full_label = ev.full_label.replace("_BREACH", "")

            # Edge Fallback Note when Windows RTX 3060 Ti (192.168.1.223) is asleep/offline
            if not ai_notes_out:
                breach_prefix = "🚨 [BREACH] " if "BREACH" in (ev.full_label or "") else ""
                face_prefix = f"🧬 [{recognized_face}] " if recognized_face else ""
                conf_pct = int(round(float(ev.confidence or 0.0) * 100))
                ai_notes_out = f"{breach_prefix}{face_prefix}[YOLOv8s EDGE] {(ev.sub_type or ev.category).upper()} ({conf_pct}% conf)"

            ev.ai_notes = ai_notes_out or ev.ai_notes
            ev.attributes = f"{snap_id}|{ai_notes_out}"[:250] if (snap_id and ai_notes_out) else (snap_id or ai_notes_out)[:250]
            await session.commit()

            asyncio.create_task(_evaluate_best_confidence_push(ev.id, face_role, recognized_face))

            return {
                "id": ev.id,
                "category": ev.category,
                "sub_type": ev.sub_type,
                "full_label": ev.full_label,
                "attributes": ev.attributes,
                "ai_notes": ai_notes_out,
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
