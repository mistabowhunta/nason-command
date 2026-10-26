import asyncio
import json
import os
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


def _extract_in_thread(img_bytes: bytes):
    np_arr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return face_engine.extract_face_data(bgr)


async def sync_public_mugshot_feeds(max_new_per_pass: int = 65) -> dict:
    """
    Paginates across West Coast FBI Field Offices (Portland, Seattle, SF, Sacramento, LA)
    + National FBI Wanted Pages 1..8 + Oregon DOC Most Wanted.
    Extracts 128-D SFace vectors in RAM (0 KB disk used) with sightings=0.
    """
    if not face_engine.enabled:
        return {"added": 0, "total_auto": 0, "error": "Face engine disabled"}

    async with AsyncSessionLocal() as session:
        res = await session.execute(select(KnownFace.sample_crop).where(KnownFace.role == "watchlist_auto"))
        existing_keys = {r[0] for r in res.all() if r[0]}

    candidates: list[tuple[str, str, str]] = []
    seen_uids: set[str] = set()

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36",
        "Accept": "application/json, text/html",
    }

    query_plan = [
        {"field_offices": "portland", "pageSize": 50, "page": 1},
        {"field_offices": "seattle", "pageSize": 50, "page": 1},
        {"field_offices": "sanfrancisco", "pageSize": 50, "page": 1},
        {"field_offices": "sacramento", "pageSize": 50, "page": 1},
        {"field_offices": "losangeles", "pageSize": 50, "page": 1},
    ] + [{"pageSize": 50, "page": p} for p in range(1, 9)]

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=headers) as client:
        for params in query_plan:
            try:
                r = await client.get(FBI_API_URL, params=params)
                if r.status_code == 200:
                    items = r.json().get("items", [])
                    for item in items:
                        uid = item.get("uid")
                        if not uid or uid in seen_uids:
                            continue
                        seen_uids.add(uid)
                        title = (item.get("title") or "UNKNOWN").strip().upper()
                        if any(skip in title for skip in ("UNKNOWN", "UNIDENTIFIED", "INFORMATION", "BANK ROBBER", "JOHN DOE", "JANE DOE")):
                            continue
                        clean_name = re.sub(r"[^A-Z0-9 _-]", "", title)[:46].strip()
                        images = item.get("images") or []
                        if images:
                            img_url = images[0].get("original") or images[0].get("large") or images[0].get("thumb")
                            key = f"fbi:{uid}"
                            if img_url and key not in existing_keys:
                                candidates.append((key, f"FBI: {clean_name}", img_url))
                await asyncio.sleep(0.25)
            except Exception as e:
                print(f"[WATCHLIST SCRAPER] FBI page warning: {e}")

        try:
            r_or = await client.get(OREGON_DOC_URL)
            if r_or.status_code == 200:
                html = r_or.text
                matches = re.findall(r'<img[^>]+src="([^"]+\.(?:jpg|jpeg|png))"[^>]*alt="([^"]*)"', html, re.IGNORECASE)
                for src, alt in matches:
                    if any(skip in src.lower() for skip in ("logo", "banner", "icon", "seal")):
                        continue
                    full_url = src if src.startswith("http") else f"https://www.oregon.gov{src}"
                    clean_alt = re.sub(r"[^A-Z0-9 _-]", "", alt.upper())[:42].strip() or "OR_DOC_FUGITIVE"
                    key = f"ordoc:{os.path.basename(src)}"
                    if key not in existing_keys:
                        candidates.append((key, f"OR_DOC: {clean_alt}", full_url))
        except Exception as e:
            print(f"[WATCHLIST SCRAPER] Oregon DOC warning: {e}")

        added_count = 0
        for key, name, img_url in candidates[:max_new_per_pass]:
            if key in existing_keys:
                continue
            try:
                img_resp = await client.get(img_url)
                if img_resp.status_code != 200:
                    continue
                vec, det_score, _ = await asyncio.to_thread(_extract_in_thread, img_resp.content)
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
                await asyncio.sleep(0.05)
            except Exception:
                continue

    await face_engine.sync_gallery_from_db_and_disk()
    auto_total = sum(1 for f in face_engine.gallery if f["role"] == "watchlist_auto")
    print(f"[WATCHLIST SCRAPER] Sync complete: +{added_count} new vectors ({auto_total} total auto-watchlist)")
    return {"added": added_count, "total_auto": auto_total}


async def periodic_watchlist_worker():
    await asyncio.sleep(10)
    while True:
        try:
            await sync_public_mugshot_feeds(max_new_per_pass=65)
        except Exception as e:
            print(f"[WATCHLIST WORKER ERROR] {e}")
        await asyncio.sleep(21600)  # Syncs every 6 hours until all pages are indexed
