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

COSINE_THRESHOLD = 0.363


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
                    score_threshold=0.58,
                    nms_threshold=0.30,
                    top_k=50,
                )
                self.recognizer = cv2.FaceRecognizerSF.create(
                    model=SFACE_PATH,
                    config="",
                )
                self.enabled = True
                print("[FACE ENGINE] YuNet + SFace Biometric Engine ready.")
            except Exception as e:
                print(f"[FACE ENGINE ERROR] {e}")

    def extract_face_data(self, bgr_img: np.ndarray) -> tuple[np.ndarray | None, float, np.ndarray | None]:
        """Returns (128D_norm_vector, detection_confidence, padded_face_crop_bgr)."""
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

            # Create a clean padded portrait crop for the UI / OSINT export
            fx, fy, fw_box, fh_box = int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3])
            pad_w, pad_h = int(fw_box * 0.45), int(fh_box * 0.45)
            cx1 = max(0, fx - pad_w)
            cy1 = max(0, fy - pad_h)
            cx2 = min(ww, fx + fw_box + pad_w)
            cy2 = min(wh, fy + fh_box + pad_h)
            face_crop = work_img[cy1:cy2, cx1:cx2].copy() if cy2 > cy1 and cx2 > cx1 else aligned.copy()

            return vec, det_score, face_crop

    def extract_largest_face_embedding(self, bgr_img: np.ndarray) -> np.ndarray | None:
        vec, _, _ = self.extract_face_data(bgr_img)
        return vec

    def match_embedding(self, query_vec: np.ndarray) -> tuple[str | None, str | None, float, int]:
        """Returns (NAME, role, cosine_score, sightings) or (None, None, 0.0, 0)."""
        if query_vec is None or not self.gallery:
            return None, None, 0.0, 0

        best_item = None
        best_score = 0.0

        for item in self.gallery:
            score = float(np.dot(query_vec, item["vec"]))
            if score > best_score:
                best_score = score
                best_item = item

        if best_score >= COSINE_THRESHOLD and best_item:
            return (
                best_item["name"].upper(),
                best_item["role"],
                round(best_score, 3),
                int(best_item.get("sightings", 1)),
            )
        return None, None, round(best_score, 3), 0

    def identify_person_in_frame(self, frame: np.ndarray, det: dict) -> tuple[str | None, str | None, float, int]:
        if not self.enabled or not self.gallery:
            return None, None, 0.0, 0

        fh, fw = frame.shape[:2]
        pad_x = det["w"] * 0.15
        pad_y = det["h"] * 0.12
        x1 = int(max(0.0, det["x"] - pad_x) * fw)
        y1 = int(max(0.0, det["y"] - pad_y) * fh)
        x2 = int(min(1.0, det["x"] + det["w"] + pad_x) * fw)
        y2 = int(min(1.0, det["y"] + det["h"] * 0.72 + pad_y) * fh)

        vec, _, _ = self.extract_face_data(frame[y1:y2, x1:x2])
        if vec is None:
            y2_full = int(min(1.0, det["y"] + det["h"] + pad_y) * fh)
            vec, _, _ = self.extract_face_data(frame[y1:y2_full, x1:x2])

        if vec is None:
            return None, None, 0.0, 0
        return self.match_embedding(vec)

    async def process_snapshot_biometrics(self, crop_path: str) -> dict | None:
        """
        Runs on newly saved person snapshots:
        1. If face matches existing Whitelist / Watchlist / Visitor -> bumps sightings count & last_seen.
        2. If face is unknown AND clear (det_score >= 0.75) -> auto-enrolls as VISITOR_XX!
        """
        if not self.enabled or not os.path.exists(crop_path):
            return None

        img = cv2.imread(crop_path)
        if img is None or img.size == 0:
            return None

        vec, det_score, face_crop = self.extract_face_data(img)
        if vec is None:
            return None

        fname, frole, fscore, sightings = self.match_embedding(vec)
        now_ts = time.time()

        if fname:
            # Known person or repeat visitor: bump sightings (throttled once per 45s per name)
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

            return {
                "name": fname,
                "role": frole,
                "score": fscore,
                "sightings": sightings,
                "auto_enrolled": False,
            }

        # Unknown face: only auto-cluster as VISITOR_XX if face is high confidence
        if det_score < 0.75 or face_crop is None:
            return None

        existing_visitor_nums = []
        for item in self.gallery:
            nm = item["name"]
            if nm.startswith("VISITOR_"):
                tail = nm.replace("VISITOR_", "").strip()
                if tail.isdigit():
                    existing_visitor_nums.append(int(tail))
        next_num = (max(existing_visitor_nums) + 1) if existing_visitor_nums else 1
        visitor_name = f"VISITOR_{next_num:02d}"

        crop_filename = f"face_{visitor_name}_{int(now_ts * 1000)}.jpg"
        saved_crop_path = f"{FACE_CROPS_DIR}/{crop_filename}"
        cv2.imwrite(saved_crop_path, face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        async with AsyncSessionLocal() as session:
            row = KnownFace(
                name=visitor_name,
                role="visitor",
                embedding_json=json.dumps(vec.tolist()),
                sample_crop=f"/face_crops/{crop_filename}",
                sightings=1,
                last_seen=datetime.now(timezone.utc),
            )
            session.add(row)
            await session.commit()

        await self.sync_gallery_from_db_and_disk()
        self.last_sighting_bump[visitor_name] = now_ts

        return {
            "name": visitor_name,
            "role": "visitor",
            "score": 1.0,
            "sightings": 1,
            "auto_enrolled": True,
        }

    async def sync_gallery_from_db_and_disk(self):
        async with AsyncSessionLocal() as session:
            if os.path.exists(KNOWN_FACES_DIR):
                res = await session.execute(select(KnownFace))
                existing_rows = res.scalars().all()
                existing_samples = {r.sample_crop for r in existing_rows if r.sample_crop}

                for person_dir in sorted(os.listdir(KNOWN_FACES_DIR)):
                    full_dir = os.path.join(KNOWN_FACES_DIR, person_dir)
                    if not os.path.isdir(full_dir) or person_dir.startswith("."):
                        continue
                    person_name = person_dir.strip().upper()
                    role = "watchlist" if person_name.startswith("WATCHLIST") else "whitelist"
                    for img_path in glob.glob(f"{full_dir}/*.*"):
                        bname = os.path.basename(img_path)
                        crop_url = f"/face_crops/disk_{person_name}_{bname}"
                        if crop_url in existing_samples:
                            continue
                        img = cv2.imread(img_path)
                        vec, _, face_crop = self.extract_face_data(img)
                        if vec is not None:
                            if face_crop is not None:
                                cv2.imwrite(f"{FACE_CROPS_DIR}/disk_{person_name}_{bname}", face_crop)
                            session.add(
                                KnownFace(
                                    name=person_name,
                                    role=role,
                                    embedding_json=json.dumps(vec.tolist()),
                                    sample_crop=crop_url,
                                    sightings=1,
                                    last_seen=datetime.now(timezone.utc),
                                )
                            )
                await session.commit()

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
                        "sightings": r.sightings or 1,
                        "last_seen": r.last_seen.isoformat() if r.last_seen else None,
                        "created_at": r.created_at.isoformat() if r.created_at else None,
                    })
                except Exception:
                    pass

            self.gallery = loaded
            return loaded


face_engine = BiometricFaceEngine()
