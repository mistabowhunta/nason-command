import asyncio
import json
import os
import time
import httpx
from app.db.session import AsyncSessionLocal
from app.db.models import PushSettings

CAM_NAMES = {
    "annke_1": "DRIVEWAY",
    "tapo_1": "LOFT",
    "tapo_2": "LIVING_ROOM",
}

_PUSH_LOCK = asyncio.Lock()
_LAST_PUSH_BY_KEY: dict[str, tuple[float, str]] = {}


async def get_push_config() -> PushSettings:
    async with AsyncSessionLocal() as session:
        cfg = await session.get(PushSettings, 1)
        if not cfg:
            cfg = PushSettings(
                id=1,
                enabled=True,
                ntfy_topic="nason_command_97113",
                discord_webhook="",
                discord_bot_token="",
                notify_breach=True,
                notify_watchlist=True,
                notify_delivery=True,
                away_mode=False,
                away_armed_at=0.0,
                mute_until=0.0,
                daily_sitrep_enabled=True,
                daily_sitrep_time="05:00",
            )
            session.add(cfg)
            await session.commit()
            await session.refresh(cfg)
        return cfg


async def send_tactical_push(
    title: str,
    message: str,
    crop_path: str | None = None,
    full_path: str | None = None,
    plate_path: str | None = None,
    clip_path: str | None = None,
    confidence: float | None = None,
    priority: str = "high",
    tags: str = "rotating_light,shield",
    dedup_key: str | None = None,
    cooldown_sec: float = 90.0,
    silent: bool = False,
    bypass_mute: bool = False,
) -> dict:
    cfg = await get_push_config()
    if not cfg.enabled:
        return {"sent": False, "reason": "Push disabled"}

    now = time.time()
    if not bypass_mute and (cfg.mute_until or 0.0) > now:
        rem_min = max(1, int(round((cfg.mute_until - now) / 60.0)))
        return {"sent": False, "reason": f"Alerts muted ({rem_min}m remaining)"}

    async with _PUSH_LOCK:
        now = time.time()
        if dedup_key:
            prev = _LAST_PUSH_BY_KEY.get(dedup_key)
            if prev is not None:
                prev_ts, prev_prio = prev
                # Suppress duplicate alerts within cooldown_sec unless upgrading to urgent (Watchlist)
                if (now - prev_ts) < cooldown_sec and not (priority == "urgent" and prev_prio != "urgent"):
                    return {"sent": False, "reason": f"Suppressed duplicate ({dedup_key}) within {int(cooldown_sec)}s cooldown"}

    conf_badge = f" · {int(round(confidence * 100))}% CONF" if confidence is not None else ""
    full_title = f"{title}{conf_badge}"

    sent_channels = []
    async with httpx.AsyncClient(timeout=25.0) as client:
        # 1. ntfy.sh Instant Push (skip during silent sitreps so phone stays quiet at 5 AM)
        topic = (cfg.ntfy_topic or "").strip()
        if topic and not silent:
            try:
                headers = {
                    "Title": full_title,
                    "Priority": priority,
                    "Tags": tags,
                }
                attach_img = crop_path if (crop_path and os.path.exists(crop_path)) else full_path
                if attach_img and os.path.exists(attach_img):
                    headers["Message"] = message
                    headers["Filename"] = "target_crop.jpg"
                    with open(attach_img, "rb") as f:
                        img_bytes = f.read()
                    n_resp = await client.put(f"https://ntfy.sh/{topic}", content=img_bytes, headers=headers)
                else:
                    n_resp = await client.post(f"https://ntfy.sh/{topic}", content=message.encode("utf-8"), headers=headers)
                if n_resp.status_code < 400:
                    sent_channels.append(f"ntfy.sh/{topic}")
                else:
                    print(f"[PUSH WARN ntfy HTTP {n_resp.status_code}] {n_resp.text[:160]}")
            except Exception as e:
                print(f"[PUSH ERROR ntfy] {e}")

        # 2. Discord Webhook (attaches wide full_scene.jpg + close-up crop.jpg + optional plate_crop.jpg + 5s MP4 replay)
        webhook = (cfg.discord_webhook or "").strip()
        if webhook.startswith("http"):
            try:
                prefix = "" if silent else "@everyone "
                content_str = f"{prefix}**{full_title}**\n`{message}`"
                payload_obj: dict = {"content": content_str}
                if silent:
                    payload_obj["flags"] = 4096  # SUPPRESS_NOTIFICATIONS
                    payload_obj["allowed_mentions"] = {"parse": []}

                image_parts: list[tuple[str, bytes, str]] = []
                total_bytes = 0
                for fname, fpath in (
                    ("full_scene.jpg", full_path),
                    ("crop.jpg", crop_path),
                    ("plate_crop.jpg", plate_path),
                ):
                    if fpath and os.path.exists(fpath):
                        raw = open(fpath, "rb").read()
                        if total_bytes + len(raw) < 9_000_000:
                            image_parts.append((fname, raw, "image/jpeg"))
                            total_bytes += len(raw)

                all_parts = list(image_parts)
                has_clip = False
                if clip_path and os.path.exists(clip_path):
                    csize = os.path.getsize(clip_path)
                    if 1024 < csize and (total_bytes + csize) < 9_000_000:
                        with open(clip_path, "rb") as cf:
                            all_parts.append(("replay_5s.mp4", cf.read(), "video/mp4"))
                            has_clip = True

                if all_parts:
                    multipart_files = [(f"files[{i}]", p) for i, p in enumerate(all_parts)]
                    d_resp = await client.post(webhook, data={"payload_json": json.dumps(payload_obj)}, files=multipart_files)
                    # Fallback without MP4 if Discord rejects combined multipart size (e.g. HTTP 413)
                    if d_resp.status_code >= 400 and has_clip and image_parts:
                        print(f"[PUSH WARN discord HTTP {d_resp.status_code} with clip; retrying images only]")
                        fb_files = [(f"files[{i}]", p) for i, p in enumerate(image_parts)]
                        d_resp = await client.post(webhook, data={"payload_json": json.dumps(payload_obj)}, files=fb_files)
                else:
                    d_resp = await client.post(webhook, json=payload_obj)

                if d_resp.status_code in (200, 204):
                    sent_channels.append("discord")
                else:
                    print(f"[PUSH ERROR discord HTTP {d_resp.status_code}] {d_resp.text[:200]}")
            except Exception as e:
                print(f"[PUSH ERROR discord] {e}")

    if sent_channels and dedup_key:
        async with _PUSH_LOCK:
            _LAST_PUSH_BY_KEY[dedup_key] = (time.time(), priority)

    return {"sent": len(sent_channels) > 0, "channels": sent_channels}


async def send_discord_files(
    title: str,
    lines: list[str],
    files_data: list[tuple[str, bytes, str]],
    silent: bool = True,
) -> dict:
    """
    Posts a multi-file status/sitrep card to the configured Discord webhook.
    When silent=True, sets Discord's SUPPRESS_NOTIFICATIONS flag (4096) and omits @everyone.
    """
    cfg = await get_push_config()
    webhook = (cfg.discord_webhook or "").strip()
    if not webhook.startswith("http"):
        return {"sent": False, "reason": "No Discord webhook configured"}

    prefix = "" if silent else "@everyone "
    body_block = "\n".join(lines)
    content_str = f"{prefix}**{title}**\n```yaml\n{body_block}\n```"
    payload_obj: dict = {"content": content_str}
    if silent:
        payload_obj["flags"] = 4096
        payload_obj["allowed_mentions"] = {"parse": []}

    multipart = [
        (f"files[{idx}]", (fname, raw_bytes, mime))
        for idx, (fname, raw_bytes, mime) in enumerate(files_data)
    ]

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            if multipart:
                resp = await client.post(webhook, data={"payload_json": json.dumps(payload_obj)}, files=multipart)
            else:
                resp = await client.post(webhook, json=payload_obj)
            return {"sent": resp.status_code in (200, 204), "status_code": resp.status_code, "silent": silent}
    except Exception as e:
        print(f"[SITREP DISCORD ERROR] {e}")
        return {"sent": False, "reason": str(e)}
