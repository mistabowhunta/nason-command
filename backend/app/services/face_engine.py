import glob
import json
import os
import threading
import time
from datetime import datetime, timezone
import cv2
import numpy as np
from sqlalchemy import select, update
from app.db.session import AsyncSessionLocal
from app.db.models import KnownFace

YUNET_PATH = "/app/app/models_zoo/face_detection_yunet_2023mar.onnx"
SFACE_PATH = "/app/app/models_zoo/face_recognition_sface_2021dec.onnx"
KNOWN_FACES_DIR = "/app/app/known_faces"
FACE_CROPS_DIR = "/tmp/command_faces"
os.makedirs(FACE_CROPS_DIR, exist_ok=True)

WHITELIST_THRESHOLD = 0.360
VISITOR_THRESHOLD = 0.385
WATCHLIST_THRESHOLD = 0.520


class BiometricFaceEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.enabled = False
        self.gallery: list[dict] = []
        self.last_sighting_bump: dict[str, float] = {}

        if os.path.exists(YUNET_PATH) and os.path.exists(SFACE_PATH):
            try:
                self.detector = cv2.FaceDetectorYN.create(
                    model=YUNET_PATH,
                    config="",
                    input_size=(320, 320),
                    score_threshold=0.60,
                    nms_threshold=0.30,
                    top_k=50,
                )
                self.recognizer = cv2.FaceRecognizerSF.create(model=SFACE_PATH, config="")
                self.enabled = True
            except Exception as e:
                print(f"[FACE ENGINE ERROR] {e}")

    def extract_face_data(self, bgr_img: np.ndarray) -> tuple[np.ndarray | None, float, np.ndarray | None]:
        if not self.enabled or bgr_img is None or bgr_img.size == 0:
            return None, 0.0, None

        with self.lock:
            ih, iw = bgr_img.shape[:2]
            if iw < 24 or ih < 24:
                return None, 0.0, None

            work_img = bgr_img
            if max(iw, ih) < 260:
                scale = 260.0 / max(iw, ih)
                work_img = cv2.resize(bgr_img, (int(iw * scale), int(ih * scale)), interpolation=cv2.INTER_CUBIC)
            elif max(iw, ih) > 960:
                scale = 960.0 / max(iw, ih)
                work_img = cv2.resize(bgr_img, (int(iw * scale), int(ih * scale)))

            wh, ww = work_img.shape[:2]
            self.detector.setInputSize((ww, wh))
            _, faces = self.detector.detect(work_img)
            if faces is None or len(faces) == 0:
                return None, 0.0, None

            best_face = max(faces, key=lambda f: (f[2] * f[3]) * f[14])
            det_score = float(best_face[14])

            aligned = self.recognizer.alignCrop(work_img, best_face)
            feat = self.recognizer.feature(aligned)
            vec = np.asarray(feat, dtype=np.float32).flatten()
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            fx, fy, fw_box, fh_box = int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3])
            pad_w, pad_h = int(fw_box * 0.45), int(fh_box * 0.45)
            cx1, cy1 = max(0, fx - pad_w), max(0, fy - pad_h)
            cx2, cy2 = min(ww, fx + fw_box + pad_w), min(wh, fy + fh_box + pad_h)
            face_crop = work_img[cy1:cy2, cx1:cx2].copy() if cy2 > cy1 and cx2 > cx1 else aligned.copy()
            return vec, det_score, face_crop

    def extract_largest_face_embedding(self, bgr_img: np.ndarray) -> np.ndarray | None:
        vec, _, _ = self.extract_face_data(bgr_img)
        return vec

    def match_embedding(self, query_vec: np.ndarray, det_score: float = 0.80) -> tuple[str | None, str | None, float, int]:
        if query_vec is None or not self.gallery:
            return None, None, 0.0, 0

        best_wl, best_wl_score = None, 0.0
        for item in self.gallery:
            if item["role"] != "whitelist":
                continue
            sc = float(np.dot(query_vec, item["vec"]))
            if sc > best_wl_score:
                best_wl_score, best_wl = sc, item

        if best_wl and best_wl_score >= WHITELIST_THRESHOLD:
            return best_wl["name"].upper(), "whitelist", round(best_wl_score, 3), int(best_wl.get("sightings") or 1)

        best_vis, best_vis_score = None, 0.0
        for item in self.gallery:
            if item["role"] != "visitor":
                continue
            sc = float(np.dot(query_vec, item["vec"]))
            if sc > best_vis_score:
                best_vis_score, best_vis = sc, item

        if best_vis and best_vis_score >= VISITOR_THRESHOLD:
            return best_vis["name"].upper(), "visitor", round(best_vis_score, 3), int(best_vis.get("sightings") or 1)

        if det_score >= 0.78:
            best_watch, best_watch_score = None, 0.0
            for item in self.gallery:
                if item["role"] not in ("watchlist", "watchlist_auto"):
                    continue
                sc = float(np.dot(query_vec, item["vec"]))
                if sc > best_watch_score:
                    best_watch_score, best_watch = sc, item

            if best_watch and best_watch_score >= WATCHLIST_THRESHOLD:
                return (
                    best_watch["name"].upper(),
                    best_watch["role"],
                    round(best_watch_score, 3),
                    int(best_watch.get("sightings") or 0),
                )

        return None, None, round(max(best_wl_score, best_vis_score), 3), 0

    def identify_person_in_frame(self, frame: np.ndarray, det: dict) -> tuple[str | None, str | None, float, int]:
        if not self.enabled or not self.gallery:
            return None, None, 0.0, 0

        fh, fw = frame.shape[:2]
        pad_x, pad_y = det["w"] * 0.15, det["h"] * 0.12
        x1 = int(max(0.0, det["x"] - pad_x) * fw)
        y1 = int(max(0.0, det["y"] - pad_y) * fh)
        x2 = int(min(1.0, det["x"] + det["w"] + pad_x) * fw)
        y2 = int(min(1.0, det["y"] + det["h"] * 0.72 + pad_y) * fh)

        vec, det_score, _ = self.extract_face_data(frame[y1:y2, x1:x2])
        if vec is None:
            y2_full = int(min(1.0, det["y"] + det["h"] + pad_y) * fh)
            vec, det_score, _ = self.extract_face_data(frame[y1:y2_full, x1:x2])

        if vec is None:
            return None, None, 0.0, 0
        return self.match_embedding(vec, det_score)

    async def process_snapshot_biometrics(self, crop_path: str) -> dict | None:
        if not self.enabled or not os.path.exists(crop_path):
            return None

        img = cv2.imread(crop_path)
        if img is None or img.size == 0:
            return None

        vec, det_score, face_crop = self.extract_face_data(img)
        if vec is None:
            return None

        fname, frole, fscore, sightings = self.match_embedding(vec, det_score)
        now_ts = time.time()

        if fname:
            if now_ts - self.last_sighting_bump.get(fname, 0.0) > 45.0:
                self.last_sighting_bump[fname] = now_ts
                sightings += 1
                async with AsyncSessionLocal() as session:
                    await session.execute(
                        update(KnownFace)
                        .where(KnownFace.name == fname)
                        .values(sightings=sightings, last_seen=datetime.now(timezone.utc))
                    )
                    await session.commit()
                for item in self.gallery:
                    if item["name"] == fname:
                        item["sightings"] = sightings
                        item["last_seen"] = datetime.now(timezone.utc).isoformat()

            return {"name": fname, "role": frole, "score": fscore, "sightings": sightings, "auto_enrolled": False}

        if det_score < 0.78 or face_crop is None:
            return None

        existing_nums = [
            int(item["name"].replace("VISITOR_", "").strip())
            for item in self.gallery
            if item["name"].startswith("VISITOR_") and item["name"].replace("VISITOR_", "").strip().isdigit()
        ]
        next_num = (max(existing_nums) + 1) if existing_nums else 1
        visitor_name = f"VISITOR_{next_num:02d}"

        crop_filename = f"face_{visitor_name}_{int(now_ts * 1000)}.jpg"
        cv2.imwrite(f"{FACE_CROPS_DIR}/{crop_filename}", face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        async with AsyncSessionLocal() as session:
            session.add(
                KnownFace(
                    name=visitor_name,
                    role="visitor",
                    embedding_json=json.dumps(vec.tolist()),
                    sample_crop=f"/face_crops/{crop_filename}",
                    sightings=1,
                    last_seen=datetime.now(timezone.utc),
                )
            )
            await session.commit()

        await self.sync_gallery_from_db_and_disk()
        self.last_sighting_bump[visitor_name] = now_ts
        return {"name": visitor_name, "role": "visitor", "score": 1.0, "sightings": 1, "auto_enrolled": True}

    async def sync_gallery_from_db_and_disk(self):
        async with AsyncSessionLocal() as session:
            res = await session.execute(select(KnownFace).order_by(KnownFace.last_seen.desc()))
            rows = res.scalars().all()
            loaded = []
            for r in rows:
                try:
                    arr = np.asarray(json.loads(r.embedding_json), dtype=np.float32).flatten()
                    norm = np.linalg.norm(arr)
                    if norm > 0:
                        arr = arr / norm
                    loaded.append({
                        "id": r.id,
                        "name": r.name.upper(),
                        "role": r.role or "whitelist",
                        "vec": arr,
                        "sample_crop": r.sample_crop,
                        "sightings": r.sightings if r.sightings is not None else 0,
                        "last_seen": r.last_seen.isoformat() if r.last_seen else None,
                        "created_at": r.created_at.isoformat() if r.created_at else None,
                    })
                except Exception:
                    pass
            self.gallery = loaded
            return loaded


face_engine = BiometricFaceEngine()
