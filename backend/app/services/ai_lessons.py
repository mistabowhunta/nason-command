import json
import os
import threading
from datetime import datetime, timezone

LESSONS_FILE = "/tmp/command_faces/ai_lessons.json"
_lock = threading.Lock()

DEFAULT_LESSONS = [
    {
        "id": 1,
        "camera_id": "annke_1",
        "wrong_sub_type": "cat",
        "correct_category": "animal",
        "correct_sub_type": "dog",
        "note": "Dog being walked by owner with leash on the sidewalk (only classify as dog when person/leash is present).",
        "created_at": datetime.now(timezone.utc).isoformat(),
    },
    {
        "id": 2,
        "camera_id": "annke_1",
        "wrong_sub_type": "bird",
        "correct_category": "animal",
        "correct_sub_type": "cat",
        "note": "Solo four-legged animal walking on driveway/pavement without a person or leash is a neighborhood cat (never bird or dog).",
        "created_at": datetime.now(timezone.utc).isoformat(),
    },
]

_MEM_LESSONS = [dict(x) for x in DEFAULT_LESSONS]


def _sanitize_lessons(raw_list: list[dict]) -> list[dict]:
    cleaned = []
    for item in raw_list:
        c_cat = str(item.get("correct_category") or "").lower()
        c_sub = str(item.get("correct_sub_type") or "").lower()
        if c_cat == "person" and any(n in c_sub for n in ("isaac", "jennifer", "beau", "lyla")):
            continue
        cleaned.append(item)
    if not any(x.get("wrong_sub_type") == "bird" and x.get("correct_sub_type") == "cat" for x in cleaned):
        cleaned.append(dict(DEFAULT_LESSONS[1], id=max([int(x.get("id", 0)) for x in cleaned] + [1]) + 1))
    return cleaned


def reset_ai_lessons() -> list[dict]:
    global _MEM_LESSONS
    with _lock:
        _MEM_LESSONS = [dict(x) for x in DEFAULT_LESSONS]
        try:
            os.makedirs(os.path.dirname(LESSONS_FILE), exist_ok=True)
            with open(LESSONS_FILE, "w") as f:
                json.dump(_MEM_LESSONS, f, indent=2)
        except Exception:
            pass
        return list(_MEM_LESSONS)


def get_ai_lessons(camera_id: str | None = None) -> list[dict]:
    global _MEM_LESSONS
    with _lock:
        try:
            if os.path.exists(LESSONS_FILE):
                with open(LESSONS_FILE, "r") as f:
                    loaded = json.load(f)
                sanitized = _sanitize_lessons(loaded)
                if len(sanitized) != len(loaded):
                    with open(LESSONS_FILE, "w") as fw:
                        json.dump(sanitized, fw, indent=2)
                _MEM_LESSONS = sanitized
            else:
                os.makedirs(os.path.dirname(LESSONS_FILE), exist_ok=True)
                with open(LESSONS_FILE, "w") as f:
                    json.dump(_MEM_LESSONS, f, indent=2)
        except Exception:
            pass
        if camera_id:
            return [x for x in _MEM_LESSONS if x.get("camera_id") in (camera_id, "ALL", None)]
        return list(_MEM_LESSONS)


def add_ai_lesson(camera_id: str, wrong_sub_type: str, correct_category: str, correct_sub_type: str, note: str) -> list[dict]:
    global _MEM_LESSONS
    lessons = get_ai_lessons()
    next_id = max([int(x.get("id", 0)) for x in lessons] + [0]) + 1
    entry = {
        "id": next_id,
        "camera_id": camera_id or "ALL",
        "wrong_sub_type": (wrong_sub_type or "").strip().lower(),
        "correct_category": (correct_category or "animal").strip().lower(),
        "correct_sub_type": (correct_sub_type or "dog").strip().lower(),
        "note": (note or "").strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    lessons.append(entry)
    with _lock:
        _MEM_LESSONS = lessons
        try:
            os.makedirs(os.path.dirname(LESSONS_FILE), exist_ok=True)
            with open(LESSONS_FILE, "w") as f:
                json.dump(_MEM_LESSONS, f, indent=2)
        except Exception:
            pass
    return list(_MEM_LESSONS)
