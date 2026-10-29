import json
import os
import threading
from datetime import datetime, timezone

LESSONS_FILE = "/app/models/ai_lessons.json"
_lock = threading.Lock()

DEFAULT_LESSONS = [
    {
        "id": 1,
        "camera_id": "annke_1",
        "wrong_sub_type": "cat",
        "correct_category": "animal",
        "correct_sub_type": "dog",
        "note": "Black and white dog being walked by owner with leash on the sidewalk (never classify sidewalk dogs with owners/leashes as cat).",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
]

_MEM_LESSONS = [dict(x) for x in DEFAULT_LESSONS]


def get_ai_lessons(camera_id: str | None = None) -> list[dict]:
    global _MEM_LESSONS
    with _lock:
        try:
            if os.path.exists(LESSONS_FILE):
                with open(LESSONS_FILE, "r") as f:
                    _MEM_LESSONS = json.load(f)
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
