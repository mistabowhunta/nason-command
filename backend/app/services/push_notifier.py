import asyncio
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
                notify_breach=True,
                notify_watchlist=True,
                notify_delivery=True,
            )
            session.add(cfg)
            await session.commit()
            await session.refresh(cfg)
        return cfg


async def send_tactical_push(
    title: str,
    message: str,
    crop_path: str | None = None,
    plate_path: str | None = None,
    clip_path: str | None = None,
    confidence: float | None = None,
    priority: str = "high",
    tags: str = "rotating_light,shield",
    dedup_key: str | None = None,
    cooldown_sec: float = 90.0,
) -> dict:
    cfg = await get_push_config()
    if not cfg.enabled:
        return {"sent": False, "reason": "Push disabled"}

    async with _PUSH_LOCK:
        now = time.time()
        if dedup_key:
            prev = _LAST_PUSH_BY_KEY.get(dedup_key)
            if prev is not None:
                prev_ts, prev_prio = prev
                # Suppress duplicate alerts within cooldown_sec unless upgrading to urgent (Watchlist)
                if (now - prev_ts) < cooldown_sec and not (priority == "urgent" and prev_prio != "urgent"):
                    return {"sent": False, "reason": f"Suppressed duplicate ({dedup_key}) within {int(cooldown_sec)}s cooldown"}
            _LAST_PUSH_BY_KEY[dedup_key] = (now, priority)

    conf_badge = f" · {int(round(confidence * 100))}% CONF" if confidence is not None else ""
    full_title = f"{title}{conf_badge}"

    sent_channels = []
    async with httpx.AsyncClient(timeout=20.0) as client:
        # 1. ntfy.sh Instant Push (supports raw JPEG image attachment + mobile push)
        topic = (cfg.ntfy_topic or "").strip()
        if topic:
            try:
                headers = {
                    "Title": full_title,
                    "Priority": priority,
                    "Tags": tags,
                }
                if crop_path and os.path.exists(crop_path):
                    headers["Message"] = message
                    headers["Filename"] = "target_crop.jpg"
                    with open(crop_path, "rb") as f:
                        img_bytes = f.read()
                    await client.put(f"https://ntfy.sh/{topic}", content=img_bytes, headers=headers)
                else:
                    await client.post(f"https://ntfy.sh/{topic}", content=message.encode("utf-8"), headers=headers)
                sent_channels.append(f"ntfy.sh/{topic}")
            except Exception as e:
                print(f"[PUSH ERROR ntfy] {e}")

        # 2. Discord Webhook (attaches close-up crop + optional plate crop + optional 5s MP4 replay clip)
        webhook = (cfg.discord_webhook or "").strip()
        if webhook.startswith("http"):
            try:
                content_str = f"@everyone **{full_title}**\n`{message}`"
                multipart_files = []
                idx = 0
                if crop_path and os.path.exists(crop_path):
                    with open(crop_path, "rb") as f:
                        multipart_files.append((f"files[{idx}]", ("crop.jpg", f.read(), "image/jpeg")))
                    idx += 1
                if plate_path and os.path.exists(plate_path):
                    with open(plate_path, "rb") as pf:
                        multipart_files.append((f"files[{idx}]", ("plate_crop.jpg", pf.read(), "image/jpeg")))
                    idx += 1
                if clip_path and os.path.exists(clip_path) and os.path.getsize(clip_path) < 8 * 1024 * 1024:
                    with open(clip_path, "rb") as cf:
                        multipart_files.append((f"files[{idx}]", ("replay_5s.mp4", cf.read(), "video/mp4")))
                    idx += 1

                if multipart_files:
                    await client.post(webhook, data={"content": content_str}, files=multipart_files)
                else:
                    await client.post(webhook, json={"content": content_str})
                sent_channels.append("discord")
            except Exception as e:
                print(f"[PUSH ERROR discord] {e}")

    return {"sent": len(sent_channels) > 0, "channels": sent_channels}
