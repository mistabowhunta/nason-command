import asyncio
import json
import re
from datetime import datetime, timezone
import cv2
import httpx
import numpy as np
from sqlalchemy import select
from app.db.session import AsyncSessionLocal
from app.db.models import KnownFace
from app.services.face_engine import face_engine

FBI_API_URL = "https://api.fbi.gov/wanted/v1/list"
OREGON_DOC_URL = "https://www.oregon.gov/doc/pages/oregon-doc-most-wanted.aspx"


async def sync_public_mugshot_feeds() -> dict:
    """
    Scrapes Portland/West-Coast FBI Wanted API + Oregon DOC Most Wanted mugshots,
    extracts 128-D SFace vectors in RAM (0 KB disk usage), and stores them as 'watchlist_auto'.
    """
    if not face_engine.enabled:
        return {"added": 0, "total_auto": 0, "error": "Face engine disabled"}

    async with AsyncSessionLocal() as session:
        res = await session.execute(select(KnownFace.sample_crop).where(KnownFace.role == "watchlist_auto"))
        existing_keys = {r[0] for r in res.all() if r[0]}

    candidates: list[tuple[str, str, str]] = []  # (unique_key, full_name, img_url)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36",
        "Accept": "application/json, text/html",
    }

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=headers) as client:
        # 1. FBI Wanted API: Portland Field Office + Top Fugitives (Pages 1-2)
        for params in [
            {"field_offices": "portland", "pageSize": 50},
            {"pageSize": 50, "page": 1},
        ]:
            try:
                r = await client.get(FBI_API_URL, params=params)
                if r.status_code == 200:
                    items = r.json().get("items", [])
                    for item in items:
                        uid = item.get("uid")
                        title = (item.get("title") or "UNKNOWN_FUGITIVE").strip().upper()
                        # Skip non-person posters (seeking info / unknown bank robbers without names)
                        if any(skip in title for skip in ("UNKNOWN", "UNIDENTIFIED", "INFORMATION", "BANK ROBBER")):
                            continue
                        clean_name = re.sub(r"[^A-Z0-9 _-]", "", title)[:48].strip()
                        images = item.get("images") or []
                        if uid and images:
                            img_url = images[0].get("original") or images[0].get("large") or images[0].get("thumb")
                            key = f"fbi:{uid}"
                            if img_url and key not in existing_keys:
                                candidates.append((key, f"FBI: {clean_name}", img_url))
            except Exception as e:
                print(f"[WATCHLIST SCRAPER] FBI feed warning: {e}")

        # 2. Oregon DOC Most Wanted Page
        try:
            r_or = await client.get(OREGON_DOC_URL)
            if r_or.status_code == 200:
                html = r_or.text
                # Match image tags inside Oregon DOC page
                matches = re.findall(r'<img[^>]+src="([^"]+\.(?:jpg|jpeg|png))"[^>]*alt="([^"]*)"', html, re.IGNORECASE)
                for src, alt in matches:
                    if any(skip in src.lower() for skip in ("logo", "banner", "icon", "seal")):
                        continue
                    full_url = src if src.startswith("http") else f"https://www.oregon.gov{src}"
                    clean_alt = re.sub(r"[^A-Z0-9 _-]", "", alt.upper())[:42].strip() or "OR_DOC_FUGITIVE"
                    key = f"ordoc:{os.path.basename(src)}" if "os" in globals() else f"ordoc:{full_url[-32:]}"
                    if key not in existing_keys:
                        candidates.append((key, f"OR_DOC: {clean_alt}", full_url))
        except Exception as e:
            print(f"[WATCHLIST SCRAPER] Oregon DOC feed warning: {e}")

        added_count = 0
        # Process up to 35 new mugshots per sync pass so CPU stays cool
        for key, name, img_url in candidates[:35]:
            if key in existing_keys:
                continue
            try:
                img_resp = await client.get(img_url)
                if img_resp.status_code != 200:
                    continue
                np_arr = np.frombuffer(img_resp.content, np.uint8)
                bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                vec, det_score, _ = face_engine.extract_face_data(bgr)
                if vec is not None and det_score >= 0.70:
                    async with AsyncSessionLocal() as session:
                        session.add(
                            KnownFace(
                                name=name,
                                role="watchlist_auto",
                                embedding_json=json.dumps(vec.tolist()),
                                sample_crop=key,
                                sightings=0,
                                last_seen=datetime.now(timezone.utc),
                            )
                        )
                        await session.commit()
                    existing_keys.add(key)
                    added_count += 1
                await asyncio.sleep(0.08)
            except Exception:
                continue

    await face_engine.sync_gallery_from_db_and_disk()
    auto_total = sum(1 for f in face_engine.gallery if f["role"] == "watchlist_auto")
    print(f"[WATCHLIST SCRAPER] Sync complete: +{added_count} new vectors ({auto_total} total auto-watchlist)")
    return {"added": added_count, "total_auto": auto_total}


async def periodic_watchlist_worker():
    # Initial sync 15 seconds after startup, then every 12 hours
    await asyncio.sleep(15)
    while True:
        try:
            await sync_public_mugshot_feeds()
        except Exception as e:
            print(f"[WATCHLIST WORKER ERROR] {e}")
        await asyncio.sleep(43200)
