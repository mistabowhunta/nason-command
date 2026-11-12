import asyncio
from datetime import datetime
import hashlib
import json
import os
import re
import time
from zoneinfo import ZoneInfo
import httpx
import websockets
from sqlalchemy import select, func
from app.db.session import AsyncSessionLocal
from app.db.models import DetectionEvent, PushSettings
from app.services.push_notifier import CAM_NAMES, get_push_config
from app.services.cv_pipeline import (
    BACKUP_DIR,
    EdgeCVEngine,
    build_and_send_sitrep,
    get_snapshots_storage_mb,
)
from app.agent.orchestrator import probe_workstation_ollama, pick_best_vlm

DISCORD_API_BASE = "https://discord.com/api/v10"
GATEWAY_URL = "wss://gateway.discord.gg/?v=10&encoding=json"
LOCAL_TZ = ZoneInfo("America/Los_Angeles")
CHEATSHEET_STATE_PATH = os.path.join(BACKUP_DIR, ".discord_cheatsheet.json")

# Single canonical command registry: adding a command here automatically updates
# Discord Slash Commands, !text parsing, /help, AND the Pinned Channel Cheat Sheet!
COMMAND_REGISTRY: list[dict] = [
    {
        "name": "snap",
        "aliases": [],
        "section": "📸 Live Optics & Visuals",
        "slash_usage": "/snap [camera]",
        "text_usage": "!snap [all | driveway | loft | living]",
        "summary": "Grab live stamped HD JPEG frame(s) from all 3 cameras or a single feed right now.",
        "slash_def": {
            "name": "snap",
            "description": "Grab live HD snapshot(s) from your home cameras right now",
            "options": [
                {
                    "name": "camera",
                    "description": "Which camera to capture (defaults to all 3)",
                    "type": 3,
                    "required": False,
                    "choices": [
                        {"name": "All 3 Cameras", "value": "all"},
                        {"name": "Driveway (annke_1)", "value": "driveway"},
                        {"name": "Loft (tapo_1)", "value": "loft"},
                        {"name": "Living Room (tapo_2)", "value": "living_room"},
                    ],
                }
            ],
        },
    },
    {
        "name": "sitrep",
        "aliases": [],
        "section": "📸 Live Optics & Visuals",
        "slash_usage": "/sitrep",
        "text_usage": "!sitrep",
        "summary": "Post a full 3-camera visual Sitrep card + 24h detection telemetry on demand.",
        "slash_def": {
            "name": "sitrep",
            "description": "Post a full 3-camera visual Sitrep card",
        },
    },
    {
        "name": "away",
        "aliases": [],
        "section": "🛫 Posture & Alert Controls",
        "slash_usage": "/away state:[on | off]",
        "text_usage": "!away [on | off]",
        "summary": "Arm 24/7 Away Mode across all 3 cameras (or restore 23:30–05:00 PT Home schedules).",
        "slash_def": {
            "name": "away",
            "description": "Arm or disarm 24/7 Away Mode across all cameras",
            "options": [
                {
                    "name": "state",
                    "description": "Arm Away Mode (on) or return to Home Mode (off)",
                    "type": 3,
                    "required": True,
                    "choices": [
                        {"name": "ON — Arm 24/7 Away Mode", "value": "on"},
                        {"name": "OFF — Restore Home Schedules", "value": "off"},
                    ],
                }
            ],
        },
    },
    {
        "name": "mute",
        "aliases": ["snooze"],
        "section": "🛫 Posture & Alert Controls",
        "slash_usage": "/mute [minutes]",
        "text_usage": "!mute [30m | 1h | 2h | 4h]",
        "summary": "Temporarily snooze outbound push alerts (detections still save to PostgreSQL).",
        "slash_def": {
            "name": "mute",
            "description": "Temporarily snooze outbound push alerts",
            "options": [
                {
                    "name": "minutes",
                    "description": "Minutes to snooze push notifications (default 60)",
                    "type": 4,
                    "required": False,
                    "choices": [
                        {"name": "30 minutes", "value": 30},
                        {"name": "1 hour (60m)", "value": 60},
                        {"name": "2 hours (120m)", "value": 120},
                        {"name": "4 hours (240m)", "value": 240},
                    ],
                }
            ],
        },
    },
    {
        "name": "unmute",
        "aliases": [],
        "section": "🛫 Posture & Alert Controls",
        "slash_usage": "/unmute",
        "text_usage": "!unmute",
        "summary": "Clear any active snooze timer and resume outbound push alerts immediately.",
        "slash_def": {
            "name": "unmute",
            "description": "Resume outbound push alerts immediately",
        },
    },
    {
        "name": "status",
        "aliases": [],
        "section": "📊 Telemetry & Reference",
        "slash_usage": "/status",
        "text_usage": "!status",
        "summary": "Show live camera health, RTX 3060 Ti VLM reachability, HDD usage, and last 5 events.",
        "slash_def": {
            "name": "status",
            "description": "Show live camera health, GPU status, storage, and recent detections",
        },
    },
    {
        "name": "help",
        "aliases": ["commands", "cheatsheet"],
        "section": "📊 Telemetry & Reference",
        "slash_usage": "/help",
        "text_usage": "!help  (or !cheatsheet)",
        "summary": "Display this Command Reference Card (`!cheatsheet` refreshes the channel's pinned card).",
        "slash_def": {
            "name": "help",
            "description": "Show the NasonNation // Command reference cheat sheet",
        },
    },
]

VALID_COMMAND_TOKENS: set[str] = {
    tok
    for entry in COMMAND_REGISTRY
    for tok in ([entry["name"], *entry.get("aliases", [])])
}

discord_bot_state: dict = {
    "connected": False,
    "bot_username": None,
    "slash_commands_ready": False,
    "last_error": None,
}
_RELOAD_EVENT = asyncio.Event()


def notify_bot_config_changed():
    _RELOAD_EVENT.set()


def build_cheatsheet_embed() -> dict:
    sections: dict[str, list[str]] = {}
    for item in COMMAND_REGISTRY:
        sec = item["section"]
        line = f"• **`{item['slash_usage']}`**  ·  `{item['text_usage']}`\n  ↳ {item['summary']}"
        sections.setdefault(sec, []).append(line)

    fields = [
        {"name": sec_title, "value": "\n".join(lines), "inline": False}
        for sec_title, lines in sections.items()
    ]
    fields.append(
        {
            "name": "🤖 Automated Background Watchdogs",
            "value": (
                "• **`05:00 PT Daily`** — Silent 3-camera visual Sitrep (`SUPPRESS_NOTIFICATIONS`, zero buzz)\n"
                "• **`>90s RTSP Dropout`** — Instant `🔌 CAMERA OFFLINE` + silent `✅ RESTORED` snapshot\n"
                "• **`Permanent Shield`** — `ISAAC` & silent services (`LANDSCAPERS`) never trigger `@everyone`"
            ),
            "inline": False,
        }
    )

    return {
        "title": "📌 NasonNation // Command — Remote Cheat Sheet",
        "description": (
            "Control **BOSSMAN** (`192.168.1.27`) from anywhere using **Slash Menus** (type `/` on mobile) "
            "or **Text Commands** (`!cmd`).\n"
            "**Camera Aliases:** `all` · `driveway` (`annke_1`) · `loft` (`tapo_1`) · `living` (`tapo_2`)"
        ),
        "color": 0x10B981,
        "fields": fields,
        "footer": {
            "text": "Tip: Pin this message once — Command auto-updates it in-place whenever new commands are added!"
        },
    }


async def _try_pin_message(bot_token: str, channel_id: str | None, message_id: str | None) -> bool:
    if not bot_token or not channel_id or not message_id:
        return False
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.put(
                f"{DISCORD_API_BASE}/channels/{channel_id}/pins/{message_id}",
                headers={"Authorization": f"Bot {bot_token}"},
            )
            return resp.status_code in (200, 204)
    except Exception:
        return False


async def sync_discord_cheatsheet(force_new: bool = False) -> dict:
    """
    Posts or in-place updates the Reference Commands Cheat Sheet in the configured Discord channel.
    - When force_new=False (startup sync): updates the existing Cheat Sheet message in-place ONLY if
      COMMAND_REGISTRY changed (preserving the user's Pin without cluttering the channel), or posts
      it once if never posted before.
    - When force_new=True (UI button or !cheatsheet): deletes the old card if possible, posts a fresh
      Cheat Sheet at the bottom of the channel, and auto-pins it if the Bot has Manage Messages.
    """
    cfg = await get_push_config()
    webhook = (cfg.discord_webhook or "").strip()
    if not webhook.startswith("http"):
        return {"sent": False, "reason": "No Discord webhook configured"}

    bot_token = ((getattr(cfg, "discord_bot_token", "") or "").strip() or os.getenv("DISCORD_BOT_TOKEN", "").strip())
    embed = build_cheatsheet_embed()
    embed_hash = hashlib.sha256(json.dumps(embed, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    saved: dict = {}
    if os.path.exists(CHEATSHEET_STATE_PATH):
        try:
            with open(CHEATSHEET_STATE_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
        except Exception:
            saved = {}

    old_msg_id = saved.get("message_id")
    old_hash = saved.get("hash")
    clean_webhook = webhook.split("?")[0].rstrip("/")

    async with httpx.AsyncClient(timeout=15.0) as client:
        # 1. Startup auto-sync: if already posted and unchanged, verify nothing to do; if changed, edit in-place!
        if old_msg_id and not force_new:
            if old_hash == embed_hash:
                return {"sent": True, "action": "already_up_to_date", "message_id": old_msg_id}
            patch_resp = await client.patch(
                f"{clean_webhook}/messages/{old_msg_id}",
                json={"content": "", "embeds": [embed], "allowed_mentions": {"parse": []}},
            )
            if patch_resp.status_code == 200:
                data = patch_resp.json()
                new_state = {
                    "message_id": data.get("id", old_msg_id),
                    "channel_id": data.get("channel_id", saved.get("channel_id")),
                    "hash": embed_hash,
                    "updated_at": int(time.time()),
                }
                with open(CHEATSHEET_STATE_PATH, "w", encoding="utf-8") as f:
                    json.dump(new_state, f)
                return {"sent": True, "action": "updated_in_place", "message_id": new_state["message_id"]}
            # If 404 (message was deleted in Discord), fall through to post a fresh copy!

        # 2. If force_new=True and an older card exists, clean up the old one so we don't leave duplicates
        if old_msg_id and force_new:
            try:
                await client.delete(f"{clean_webhook}/messages/{old_msg_id}")
            except Exception:
                pass

        # 3. Post fresh silent Rich Embed Cheat Sheet (?wait=true returns message_id & channel_id)
        post_resp = await client.post(
            f"{clean_webhook}?wait=true",
            json={
                "embeds": [embed],
                "flags": 4096,  # SUPPRESS_NOTIFICATIONS (silent post)
                "allowed_mentions": {"parse": []},
            },
        )
        if post_resp.status_code in (200, 201):
            data = post_resp.json()
            msg_id = data.get("id")
            chan_id = data.get("channel_id")
            pinned = await _try_pin_message(bot_token, chan_id, msg_id)
            new_state = {
                "message_id": msg_id,
                "channel_id": chan_id,
                "hash": embed_hash,
                "pinned": pinned,
                "updated_at": int(time.time()),
            }
            with open(CHEATSHEET_STATE_PATH, "w", encoding="utf-8") as f:
                json.dump(new_state, f)
            return {"sent": True, "action": "posted_new", "message_id": msg_id, "pinned": pinned}

        return {"sent": False, "reason": f"Discord HTTP {post_resp.status_code}"}


async def _resolve_webhook_scope(webhook_url: str) -> tuple[str | None, str | None]:
    if not webhook_url or not webhook_url.startswith("http"):
        return None, None
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(webhook_url)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("guild_id"), data.get("channel_id")
    except Exception:
        pass
    return None, None


async def set_away_mode_db(cv_engine: EdgeCVEngine, away: bool, grace_seconds: int = 0) -> dict:
    now = time.time()
    armed_at = (now + grace_seconds) if (away and grace_seconds > 0) else (now if away else 0.0)
    async with AsyncSessionLocal() as session:
        cfg = await session.get(PushSettings, 1)
        if not cfg:
            cfg = PushSettings(id=1)
            session.add(cfg)
        cfg.away_mode = bool(away)
        cfg.away_armed_at = armed_at
        await session.commit()
        mute_until = float(cfg.mute_until or 0.0)
    cv_engine.set_away_and_mute_state(away, armed_at, mute_until)
    return {
        "away_mode": bool(away),
        "away_armed_at": armed_at,
        "grace_remaining": max(0, int(round(armed_at - now))),
    }


async def set_mute_minutes_db(cv_engine: EdgeCVEngine, minutes: int) -> dict:
    now = time.time()
    mute_until = (now + minutes * 60.0) if minutes > 0 else 0.0
    async with AsyncSessionLocal() as session:
        cfg = await session.get(PushSettings, 1)
        if not cfg:
            cfg = PushSettings(id=1)
            session.add(cfg)
        cfg.mute_until = mute_until
        await session.commit()
        away_mode = bool(cfg.away_mode)
        away_armed_at = float(cfg.away_armed_at or 0.0)
    cv_engine.set_away_and_mute_state(away_mode, away_armed_at, mute_until)
    return {
        "muted": mute_until > now,
        "mute_until": mute_until,
        "mute_remaining": max(0, int(round(mute_until - now))),
    }


def _parse_duration_minutes(arg: str) -> int:
    s = (arg or "").strip().lower()
    if not s:
        return 60
    if s in ("0", "off", "false", "none", "unmute", "clear"):
        return 0
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([hm]?)$", s)
    if not m:
        return 60
    val = float(m.group(1))
    unit = m.group(2)
    return max(1, int(round(val * 60 if unit == "h" else val)))


async def _execute_command(
    cv_engine: EdgeCVEngine, cmd: str, arg: str = ""
) -> tuple[str, list[tuple[str, bytes, str]], list[dict]]:
    cmd = cmd.lower().strip()
    arg = arg.strip().lower()
    now_ts = time.time()
    now_pt = datetime.now(LOCAL_TZ)

    if cmd == "cheatsheet":
        res = await sync_discord_cheatsheet(force_new=True)
        if res.get("sent"):
            pin_note = " and auto-pinned it" if res.get("pinned") else " (tap & hold -> Pin Message to keep it at the top)"
            return (f"📌 Posted fresh Command Cheat Sheet{pin_note}.", [], [])
        return (f"⚠️ Could not post cheat sheet: `{res.get('reason')}`", [], [])

    if cmd in ("help", "commands"):
        return ("", [], [build_cheatsheet_embed()])

    if cmd == "snap":
        alias_map = {
            "driveway": ["annke_1"],
            "annke_1": ["annke_1"],
            "annke": ["annke_1"],
            "loft": ["tapo_1"],
            "tapo_1": ["tapo_1"],
            "living": ["tapo_2"],
            "living_room": ["tapo_2"],
            "tapo_2": ["tapo_2"],
        }
        targets = alias_map.get(arg, list(cv_engine.streams))
        stamp = now_pt.strftime("%H:%M:%S PT")
        files: list[tuple[str, bytes, str]] = []
        offline: list[str] = []
        for s in targets:
            friendly = CAM_NAMES.get(s, s.upper())
            jpg = cv_engine.capture_live_jpeg_bytes(s, stamp_label=f"{friendly} // LIVE {stamp}")
            if jpg:
                files.append((f"{s}_snap.jpg", jpg, "image/jpeg"))
            else:
                offline.append(friendly)
        if not files:
            return (f"⚠️ Could not capture live frames ({', '.join(offline)} offline or warming up).", [], [])
        note = f" *(Offline: {', '.join(offline)})*" if offline else ""
        cam_list = ", ".join(CAM_NAMES.get(s, s.upper()) for s in targets)
        return (f"📸 **LIVE SNAPSHOT // {cam_list}** (`{stamp}`){note}", files, [])

    if cmd == "away":
        cfg = await get_push_config()
        if arg in ("on", "1", "true", "arm"):
            target = True
        elif arg in ("off", "0", "false", "home", "disarm"):
            target = False
        else:
            target = not bool(cfg.away_mode)

        await set_away_mode_db(cv_engine, away=target, grace_seconds=0)
        if target:
            return (
                "🛫 **AWAY MODE ARMED (IMMEDIATE)**\n"
                "`All 3 cameras (DRIVEWAY, LOFT, LIVING_ROOM) forced to 24/7 recording. "
                "Alerting on any person, vehicle, delivery, or watchlist match.`",
                [],
                [],
            )
        return (
            "🏠 **HOME MODE RESTORED**\n"
            "`Indoor cameras (LOFT, LIVING_ROOM) returned to saved 23:30–05:00 PT schedule.`",
            [],
            [],
        )

    if cmd in ("mute", "snooze"):
        mins = _parse_duration_minutes(arg)
        res = await set_mute_minutes_db(cv_engine, mins)
        if not res["muted"]:
            return ("🔔 **PUSH ALERTS RESUMED** — Outbound notifications are active.", [], [])
        until_str = datetime.fromtimestamp(res["mute_until"], LOCAL_TZ).strftime("%I:%M %p PT")
        return (f"🔇 **ALERTS SNOOZED FOR {mins} MIN** (until `{until_str}`). Events still log to Postgres.", [], [])

    if cmd == "unmute":
        await set_mute_minutes_db(cv_engine, 0)
        return ("🔔 **PUSH ALERTS RESUMED** — Outbound notifications are active.", [], [])

    if cmd == "sitrep":
        res = await build_and_send_sitrep(
            cv_engine,
            silent=True,
            header="📸 ON-DEMAND TACTICAL SITREP // BOSSMAN",
        )
        if res.get("sent"):
            return ("✅ Dispatched full 3-camera Sitrep card above.", [], [])
        return (f"⚠️ Sitrep failed: `{res.get('reason', 'unknown error')}`", [], [])

    if cmd == "status":
        from app.api.ws import START_TIME as _START_TIME
        from app.services.face_engine import face_engine

        cfg = await get_push_config()
        gpu_online, models = await probe_workstation_ollama()
        vlm = pick_best_vlm(models) if gpu_online else None
        up_s = max(0, int(now_ts - _START_TIME))
        up_str = f"{up_s // 86400}d {(up_s % 86400) // 3600}h {(up_s % 3600) // 60}m"

        cam_parts = []
        for s in cv_engine.streams:
            g = cv_engine.grabber_map.get(s)
            age = (now_ts - g.frame_time) if (g and g.frame_time > 0) else 9999
            cam_parts.append(f"{CAM_NAMES.get(s, s)}: {'LIVE' if age < 15 else f'STALE({int(age)}s)'}")

        async with AsyncSessionLocal() as session:
            rec_res = await session.execute(
                select(DetectionEvent).order_by(DetectionEvent.timestamp.desc()).limit(5)
            )
            rec_rows = rec_res.scalars().all()
            tot_res = await session.execute(select(func.count(DetectionEvent.id)))
            tot_cnt = tot_res.scalar() or 0

        recent_lines = [
            f"  #{r.id} [{r.timestamp.astimezone(LOCAL_TZ).strftime('%m/%d %H:%M')}] "
            f"{CAM_NAMES.get(r.camera_id, r.camera_id)} -> {r.sub_type} ({int(r.confidence * 100)}%)"
            for r in rec_rows
        ] or ["  (No events recorded)"]

        posture = "AWAY (24/7 ALL CAMS)" if cfg.away_mode else "HOME"
        mute_info = (
            f"MUTED ({max(1, int(round((cfg.mute_until - now_ts) / 60.0)))}m left)"
            if (cfg.mute_until or 0.0) > now_ts
            else "ACTIVE"
        )
        body = "\n".join([
            f"Posture:  {posture} | Push: {mute_info} | Daily Sitrep: {cfg.daily_sitrep_time} PT",
            f"Cameras:  {' | '.join(cam_parts)}",
            f"GPU Node: {'ONLINE (' + str(vlm) + ')' if gpu_online else 'OFFLINE / EDGE FALLBACK'}",
            f"Storage:  {get_snapshots_storage_mb()} MB / 30,720 MB | DB Events: {tot_cnt} | Faces: {len(face_engine.gallery)} | Uptime: {up_str}",
            "Recent 5:",
            *recent_lines,
        ])
        return (f"**📊 COMMAND STATUS // NODE BOSSMAN**\n```yaml\n{body}\n```", [], [])

    return ("Unknown command. Type `/help` or `!help` for the full Command Cheat Sheet.", [], [])


async def _register_guild_slash_commands(token: str, app_id: str, guild_id: str) -> bool:
    commands_payload = [entry["slash_def"] for entry in COMMAND_REGISTRY if "slash_def" in entry]
    url = f"{DISCORD_API_BASE}/applications/{app_id}/guilds/{guild_id}/commands"
    headers = {"Authorization": f"Bot {token}"}
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.put(url, headers=headers, json=commands_payload)
            return resp.status_code in (200, 201)
    except Exception as e:
        print(f"[DISCORD SLASH REG WARNING] {e}")
        return False


async def _post_channel_reply(
    token: str,
    channel_id: str,
    content: str,
    files: list[tuple[str, bytes, str]],
    embeds: list[dict] | None = None,
    reply_to_id: str | None = None,
):
    url = f"{DISCORD_API_BASE}/channels/{channel_id}/messages"
    headers = {"Authorization": f"Bot {token}"}
    payload_obj: dict = {"allowed_mentions": {"parse": []}}
    if content:
        payload_obj["content"] = content
    if embeds:
        payload_obj["embeds"] = embeds
    if reply_to_id:
        payload_obj["message_reference"] = {"message_id": reply_to_id, "fail_if_not_exists": False}

    multipart = [(f"files[{i}]", (fn, b, m)) for i, (fn, b, m) in enumerate(files)]
    async with httpx.AsyncClient(timeout=20.0) as client:
        if multipart:
            await client.post(url, headers=headers, data={"payload_json": json.dumps(payload_obj)}, files=multipart)
        else:
            await client.post(url, headers=headers, json=payload_obj)


async def _handle_slash_interaction(cv_engine: EdgeCVEngine, token: str, d: dict, allowed_guild_id: str | None):
    if allowed_guild_id and d.get("guild_id") != allowed_guild_id:
        return
    inter_id = d.get("id")
    inter_token = d.get("token")
    app_id = d.get("application_id")
    data = d.get("data") or {}
    cmd_name = (data.get("name") or "").lower()
    opts = data.get("options") or []
    first_arg = str(opts[0].get("value", "")) if opts else ""

    async with httpx.AsyncClient(timeout=20.0) as client:
        await client.post(
            f"{DISCORD_API_BASE}/interactions/{inter_id}/{inter_token}/callback",
            json={"type": 5},
        )
        reply_text, files, embeds = await _execute_command(cv_engine, cmd_name, first_arg)
        payload_obj: dict = {"allowed_mentions": {"parse": []}}
        if reply_text:
            payload_obj["content"] = reply_text
        if embeds:
            payload_obj["embeds"] = embeds
        multipart = [(f"files[{i}]", (fn, b, m)) for i, (fn, b, m) in enumerate(files)]
        edit_url = f"{DISCORD_API_BASE}/webhooks/{app_id}/{inter_token}/messages/@original"
        if multipart:
            await client.patch(edit_url, data={"payload_json": json.dumps(payload_obj)}, files=multipart)
        else:
            await client.patch(edit_url, json=payload_obj)


async def _handle_text_message(
    cv_engine: EdgeCVEngine,
    token: str,
    d: dict,
    bot_user_id: str | None,
    allowed_guild_id: str | None,
):
    author = d.get("author") or {}
    if author.get("bot"):
        return
    if allowed_guild_id and d.get("guild_id") != allowed_guild_id:
        return

    raw_content = (d.get("content") or "").strip()
    if not raw_content:
        return

    cleaned = raw_content
    if bot_user_id:
        cleaned = re.sub(rf"^<@!?{re.escape(bot_user_id)}>\s*", "!", cleaned)

    if not cleaned.startswith("!"):
        return

    parts = cleaned[1:].strip().split(None, 1)
    if not parts:
        return
    cmd_name = parts[0].lower()
    if cmd_name not in VALID_COMMAND_TOKENS:
        return
    arg_str = parts[1] if len(parts) > 1 else ""
    reply_text, files, embeds = await _execute_command(cv_engine, cmd_name, arg_str)
    await _post_channel_reply(
        token=token,
        channel_id=str(d.get("channel_id")),
        content=reply_text,
        files=files,
        embeds=embeds,
        reply_to_id=d.get("id"),
    )


async def discord_bot_worker(cv_engine: EdgeCVEngine):
    """
    Persistent outbound-only Discord Gateway v10 client (zero port-forwarding).
    Hot-reloads automatically whenever discord_bot_token is added/updated in PushSettings.
    """
    use_message_content_intent = True

    while True:
        _RELOAD_EVENT.clear()
        cfg = await get_push_config()
        token = ((getattr(cfg, "discord_bot_token", "") or "").strip() or os.getenv("DISCORD_BOT_TOKEN", "").strip())

        if not token:
            discord_bot_state.update({"connected": False, "bot_username": None, "slash_commands_ready": False})
            try:
                await asyncio.wait_for(_RELOAD_EVENT.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                pass
            continue

        allowed_guild_id, _ = await _resolve_webhook_scope(cfg.discord_webhook or "")
        intents = ((1 << 0) | (1 << 9) | (1 << 15)) if use_message_content_intent else ((1 << 0) | (1 << 9))
        seq: int | None = None
        bot_user_id: str | None = None

        try:
            async with websockets.connect(GATEWAY_URL, max_size=4 * 1024 * 1024, ping_interval=30) as ws:
                hello_raw = await ws.recv()
                hello = json.loads(hello_raw)
                hb_interval = hello.get("d", {}).get("heartbeat_interval", 41250) / 1000.0

                identify_payload = {
                    "op": 2,
                    "d": {
                        "token": token,
                        "intents": intents,
                        "properties": {"os": "linux", "browser": "nason_command", "device": "bossman"},
                    },
                }
                await ws.send(json.dumps(identify_payload))

                stop_hb = asyncio.Event()

                async def _heartbeat_loop():
                    while not stop_hb.is_set():
                        await asyncio.sleep(hb_interval * 0.9)
                        if stop_hb.is_set():
                            break
                        try:
                            await ws.send(json.dumps({"op": 1, "d": seq}))
                        except Exception:
                            break

                hb_task = asyncio.create_task(_heartbeat_loop())
                try:
                    while not _RELOAD_EVENT.is_set():
                        recv_task = asyncio.create_task(ws.recv())
                        reload_task = asyncio.create_task(_RELOAD_EVENT.wait())
                        done, pending = await await_first(recv_task, reload_task)
                        for p in pending:
                            p.cancel()
                        if reload_task in done:
                            break

                        raw_msg = recv_task.result()
                        pkt = json.loads(raw_msg)
                        if pkt.get("s") is not None:
                            seq = pkt["s"]
                        op = pkt.get("op")
                        t = pkt.get("t")
                        d = pkt.get("d") or {}

                        if op == 1:
                            await ws.send(json.dumps({"op": 1, "d": seq}))
                        elif op in (7, 9):
                            break
                        elif op == 0:
                            if t == "READY":
                                user_obj = d.get("user") or {}
                                bot_user_id = user_obj.get("id")
                                bot_name = user_obj.get("username", "CommandBot")
                                app_id = (d.get("application") or {}).get("id") or bot_user_id
                                slash_ok = False
                                if app_id and allowed_guild_id:
                                    slash_ok = await _register_guild_slash_commands(token, str(app_id), str(allowed_guild_id))
                                discord_bot_state.update({
                                    "connected": True,
                                    "bot_username": bot_name,
                                    "slash_commands_ready": slash_ok,
                                    "last_error": None,
                                })
                                # Ensure cheat sheet is posted/synced (and auto-pinned if permitted)
                                asyncio.create_task(sync_discord_cheatsheet(force_new=False))
                                print(f"[DISCORD BOT] Connected as {bot_name} (slash_commands={slash_ok}, msg_intent={use_message_content_intent})")
                            elif t == "INTERACTION_CREATE" and d.get("type") == 2:
                                asyncio.create_task(_handle_slash_interaction(cv_engine, token, d, allowed_guild_id))
                            elif t == "MESSAGE_CREATE":
                                asyncio.create_task(_handle_text_message(cv_engine, token, d, bot_user_id, allowed_guild_id))
                finally:
                    stop_hb.set()
                    hb_task.cancel()
        except Exception as e:
            err_str = str(e)
            close_code = getattr(e, "code", None) or getattr(getattr(e, "rcvd", None), "code", None)
            discord_bot_state["connected"] = False
            if close_code == 4014 or "4014" in err_str:
                use_message_content_intent = False
                discord_bot_state["last_error"] = "Connected in Slash-Only mode (enable Message Content Intent for !commands)"
                print("[DISCORD BOT] 4014 Disallowed Intent — auto-switching to Slash Commands (/snap, /away, /status) mode")
                await asyncio.sleep(1.5)
                continue
            if close_code == 4004 or "4004" in err_str:
                discord_bot_state["last_error"] = "Invalid Discord Bot Token (4004)"
                print("[DISCORD BOT ERROR] Invalid Bot Token (4004). Waiting for updated token...")
                try:
                    await asyncio.wait_for(_RELOAD_EVENT.wait(), timeout=30.0)
                except asyncio.TimeoutError:
                    pass
                continue
            discord_bot_state["last_error"] = err_str[:120]
            print(f"[DISCORD BOT RECONNECTING] {err_str}")
            await asyncio.sleep(5.0)


async def await_first(t1: asyncio.Task, t2: asyncio.Task):
    return await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_COMPLETED)
