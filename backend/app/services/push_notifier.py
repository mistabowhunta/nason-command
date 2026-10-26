import os
import httpx
from app.db.session import AsyncSessionLocal
from app.db.models import PushSettings

CAM_NAMES = {
    "annke_1": "DRIVEWAY",
    "tapo_1": "LOFT",
    "tapo_2": "LIVING_ROOM",
}


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
    priority: str = "high",
    tags: str = "rotating_light,shield",
) -> dict:
    cfg = await get_push_config()
    if not cfg.enabled:
        return {"sent": False, "reason": "Push disabled"}

    sent_channels = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        # 1. ntfy.sh Instant Push (supports raw JPEG image attachment + mobile push)
        topic = (cfg.ntfy_topic or "").strip()
        if topic:
            try:
                headers = {
                    "Title": title,
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

        # 2. Optional Discord Webhook (multipart image upload)
        webhook = (cfg.discord_webhook or "").strip()
        if webhook.startswith("http"):
            try:
                content_str = f"**{title}**\n`{message}`"
                if crop_path and os.path.exists(crop_path):
                    with open(crop_path, "rb") as f:
                        files = {"file": ("crop.jpg", f.read(), "image/jpeg")}
                        await client.post(webhook, data={"content": content_str}, files=files)
                else:
                    await client.post(webhook, json={"content": content_str})
                sent_channels.append("discord")
            except Exception as e:
                print(f"[PUSH ERROR discord] {e}")

    return {"sent": len(sent_channels) > 0, "channels": sent_channels}
