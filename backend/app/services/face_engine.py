import json
import os
import threading
import time
from datetime import datetime, timezone
import cv2
import numpy as np
from sqlalchemy import delete, select, update
from app.db.session import AsyncSessionLocal
from app.db.models import KnownFace

YUNET_PATH = "/app/app/models_zoo/face_detection_yunet_2023mar.onnx"
SFACE_PATH = "/app/app/models_zoo/face_recognition_sface_2021dec.onnx"
KNOWN_FACES_DIR = "/app/app/known_faces"
FACE_CROPS_DIR = "/tmp/command_faces"
os.makedirs(FACE_CROPS_DIR, exist_ok=True)

# Calibrated SFace cosine similarity thresholds (requires verified frontal/3-quarter anatomical face)
WHITELIST_THRESHOLD = 0.480
VISITOR_THRESHOLD = 0.460
WATCHLIST_THRESHOLD = 0.520

MIN_MATCH_DET_SCORE = 0.84
MIN_ENROLL_DET_SCORE = 0.87
MIN_AUTO_VISITOR_SCORE = 0.88

# Legacy dynamic crops verified to contain occluded eyes, ears, or motion blur
KNOWN_POISONED_CROPS = {
    "enrolled_ISAAC_1794107099115.jpg",
    "enrolled_ISAAC_1794107108506.jpg",
    "enrolled_ISAAC_1794199252312.jpg",
    "enrolled_ISAAC_1794199265479.jpg",
    "enrolled_JENNIFER_1794025379798.jpg",
    "enrolled_JENNIFER_1794281987530.jpg",
    "enrolled_JENNIFER_1794285637809.jpg",
}

VERIFIED_CLEAN_CROPS = {
    "upload_ISAAC_1793989003311.jpg",
    "upload_BEAU_1794180460162.jpg",
    "upload_LYLA_1794180572181.jpg",
    "enrolled_JENNIFER_1794178054955.jpg",
}


class BiometricFaceEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.enabled = False
        self.gallery: list[dict] = []
        self.last_sighting_bump: dict[str, float] = {}
        self._sanitized_once = False

        if os.path.exists(YUNET_PATH) and os.path.exists(SFACE_PATH):
            try:
                self.detector = cv2.FaceDetectorYN.create(
                    model=YUNET_PATH,
                    config="",
                    input_size=(320, 320),
                    score_threshold=0.76,
                    nms_threshold=0.30,
                    top_k=25,
                )
                self.recognizer = cv2.FaceRecognizerSF.create(model=SFACE_PATH, config="")
                self.enabled = True
            except Exception as e:
                print(f"[FACE ENGINE ERROR] {e}")

    @staticmethod
    def _validate_face_candidate(
        work_img: np.ndarray,
        face: np.ndarray,
        scale: float,
        min_score: float,
        min_raw_px: int,
        min_sharpness: float,
        upper_fraction: float | None,
    ) -> bool:
        det_score = float(face[14])
        if det_score < min_score:
            return False

        wh, ww = work_img.shape[:2]
        fx, fy, fw_box, fh_box = float(face[0]), float(face[1]), float(face[2]), float(face[3])
        if fw_box <= 1.0 or fh_box <= 1.0:
            return False

        # Reject tiny distant pixels upscaled into pareidolia hallucinations
        raw_w = fw_box / max(scale, 1e-4)
        raw_h = fh_box / max(scale, 1e-4)
        if raw_w < min_raw_px or raw_h < min_raw_px:
            return False

        aspect = fw_box / fh_box
        if aspect < 0.52 or aspect > 1.40:
            return False

        # On person crops, a real head must sit in the upper region (never jeans, hips, or backpacks)
        if upper_fraction is not None:
            cy_norm = (fy + fh_box * 0.5) / max(float(wh), 1.0)
            if cy_norm > upper_fraction:
                return False

        # Validate 5 YuNet anatomical landmarks: right_eye, left_eye, nose, right_mouth, left_mouth
        rex, rey = float(face[4]), float(face[5])
        lex, ley = float(face[6]), float(face[7])
        nx, ny = float(face[8]), float(face[9])
        rmx, rmy = float(face[10]), float(face[11])
        lmx, lmy = float(face[12]), float(face[13])

        mx_pad, my_pad = 0.08 * fw_box, 0.08 * fh_box
        for px, py in ((rex, rey), (lex, ley), (nx, ny), (rmx, rmy), (lmx, lmy)):
            if not (fx - mx_pad <= px <= fx + fw_box + mx_pad and fy - my_pad <= py <= fy + fh_box + my_pad):
                return False

        # Eyes must be horizontally separated (rejects ears, side-caps, hoodie seams, denim folds)
        eye_dx = abs(lex - rex)
        eye_dy = abs(ley - rey)
        if eye_dx < 0.21 * fw_box or eye_dx > 0.72 * fw_box:
            return False
        if eye_dy > 0.45 * eye_dx:
            return False

        # Strict vertical anatomy: eyes above nose, nose above mouth
        eye_max_y = max(rey, ley)
        mouth_mid_y = 0.5 * (rmy + lmy)
        if ny <= eye_max_y + 0.05 * fh_box:
            return False
        if mouth_mid_y <= ny + 0.05 * fh_box:
            return False
        if abs(lmx - rmx) < 0.11 * fw_box:
            return False

        # Luma, contrast, and focus sanity (rejects solid dark hair backs like VISITOR_21 and motion smudges)
        ix1, iy1 = max(0, int(fx)), max(0, int(fy))
        ix2, iy2 = min(ww, int(fx + fw_box)), min(wh, int(fy + fh_box))
        if (ix2 - ix1) < 20 or (iy2 - iy1) < 20:
            return False

        gray = cv2.cvtColor(work_img[iy1:iy2, ix1:ix2], cv2.COLOR_BGR2GRAY)
        mean_luma = float(np.mean(gray))
        std_luma = float(np.std(gray))
        if mean_luma < 20.0 or mean_luma > 242.0 or std_luma < 15.0:
            return False

        norm_gray = cv2.resize(gray, (96, 96))
        lap_var = float(cv2.Laplacian(norm_gray, cv2.CV_64F).var())
        if lap_var < min_sharpness:
            return False

        return True

    def extract_face_data(
        self,
        bgr_img: np.ndarray,
        min_score: float = MIN_MATCH_DET_SCORE,
        min_raw_px: int = 26,
        min_sharpness: float = 20.0,
        upper_fraction: float | None = None,
    ) -> tuple[np.ndarray | None, float, np.ndarray | None]:
        if not self.enabled or bgr_img is None or bgr_img.size == 0:
            return None, 0.0, None

        with self.lock:
            ih, iw = bgr_img.shape[:2]
            if iw < 32 or ih < 32:
                return None, 0.0, None

            scale = 1.0
            work_img = bgr_img
            max_dim = max(iw, ih)
            if max_dim < 240:
                scale = min(1.75, 240.0 / max_dim)
                work_img = cv2.resize(
                    bgr_img,
                    (max(32, int(iw * scale)), max(32, int(ih * scale))),
                    interpolation=cv2.INTER_LINEAR,
                )
            elif max_dim > 960:
                scale = 960.0 / max_dim
                work_img = cv2.resize(bgr_img, (int(iw * scale), int(ih * scale)))

            wh, ww = work_img.shape[:2]
            self.detector.setInputSize((ww, wh))
            _, faces = self.detector.detect(work_img)
            if faces is None or len(faces) == 0:
                return None, 0.0, None

            valid_faces = [
                f
                for f in faces
                if self._validate_face_candidate(
                    work_img=work_img,
                    face=f,
                    scale=scale,
                    min_score=min_score,
                    min_raw_px=min_raw_px,
                    min_sharpness=min_sharpness,
                    upper_fraction=upper_fraction,
                )
            ]
            if not valid_faces:
                return None, 0.0, None

            best_face = max(valid_faces, key=lambda f: (f[2] * f[3]) * f[14])
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

    def extract_person_crop_face(
        self,
        person_crop_bgr: np.ndarray,
        for_enrollment: bool = False,
    ) -> tuple[np.ndarray | None, float, np.ndarray | None]:
        """
        Extracts a verified face strictly from the upper head/shoulders portion of a person crop,
        preventing backpacks, hoodies, jeans, or ground textures from ever matching or enrolling.
        """
        if person_crop_bgr is None or person_crop_bgr.size == 0:
            return None, 0.0, None
        ih, iw = person_crop_bgr.shape[:2]
        if ih < 48 or iw < 32:
            return None, 0.0, None

        head_region_h = max(36, int(ih * 0.52))
        upper_crop = person_crop_bgr[:head_region_h, :]
        return self.extract_face_data(
            upper_crop,
            min_score=MIN_ENROLL_DET_SCORE if for_enrollment else MIN_MATCH_DET_SCORE,
            min_raw_px=30 if for_enrollment else 26,
            min_sharpness=32.0 if for_enrollment else 20.0,
            upper_fraction=0.82,
        )

    def extract_largest_face_embedding(self, bgr_img: np.ndarray) -> np.ndarray | None:
        vec, _, _ = self.extract_face_data(bgr_img, min_score=MIN_MATCH_DET_SCORE)
        return vec

    def match_embedding(self, query_vec: np.ndarray, det_score: float = 0.85) -> tuple[str | None, str | None, float, int]:
        if query_vec is None or not self.gallery or det_score < MIN_MATCH_DET_SCORE:
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

        # Skip headless waist-down detections clipped against the very top edge of the camera frame
        if det["y"] <= 0.008 and det["h"] < 0.36:
            return None, None, 0.0, 0

        fh, fw = frame.shape[:2]
        pad_x = det["w"] * 0.14
        pad_top = det["h"] * 0.08
        x1 = int(max(0.0, det["x"] - pad_x) * fw)
        y1 = int(max(0.0, det["y"] - pad_top) * fh)
        x2 = int(min(1.0, det["x"] + det["w"] + pad_x) * fw)
        # Strictly inspect ONLY head & upper chest (top 48% of person bbox) — never legs/jeans/backpacks
        y2 = int(min(1.0, det["y"] + det["h"] * 0.48) * fh)
        if (x2 - x1) < 32 or (y2 - y1) < 32:
            return None, None, 0.0, 0

        vec, det_score, _ = self.extract_face_data(
            frame[y1:y2, x1:x2],
            min_score=MIN_MATCH_DET_SCORE,
            min_raw_px=26,
            min_sharpness=20.0,
            upper_fraction=0.82,
        )
        if vec is None:
            return None, None, 0.0, 0
        return self.match_embedding(vec, det_score)

    async def process_snapshot_biometrics(self, crop_path: str) -> dict | None:
        if not self.enabled or not os.path.exists(crop_path):
            return None

        img = cv2.imread(crop_path)
        if img is None or img.size == 0:
            return None

        vec, det_score, face_crop = self.extract_person_crop_face(img, for_enrollment=False)
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

        # Require a genuinely sharp, high-confidence frontal face before auto-creating a VISITOR_XX profile
        if det_score < MIN_AUTO_VISITOR_SCORE or face_crop is None:
            return None
        fh_c, fw_c = face_crop.shape[:2]
        if fw_c < 72 or fh_c < 72:
            return None
        gray_vis = cv2.cvtColor(cv2.resize(face_crop, (96, 96)), cv2.COLOR_BGR2GRAY)
        if float(cv2.Laplacian(gray_vis, cv2.CV_64F).var()) < 34.0:
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

    async def purge_matching_enrolled_vectors(self, crop_path: str | None, wrong_name: str | None) -> int:
        """
        When an operator teaches that a detection labeled `ID: <NAME>` is actually a STRANGER,
        removes any dynamically captured `enrolled_<NAME>_*` or `face_VISITOR_*` vectors that
        contributed to the false match (while never touching uploaded clean portraits `upload_*`).
        """
        if not wrong_name:
            return 0
        target_upper = wrong_name.replace("ID:", "").strip().upper().split()[0]
        if not target_upper:
            return 0

        query_vec = None
        if crop_path and os.path.exists(crop_path):
            img = cv2.imread(crop_path)
            if img is not None and img.size > 0:
                query_vec, _, _ = self.extract_person_crop_face(img, for_enrollment=False)

        removed = 0
        async with AsyncSessionLocal() as session:
            res = await session.execute(select(KnownFace).where(KnownFace.name == target_upper))
            rows = res.scalars().all()
            for r in rows:
                bname = os.path.basename(r.sample_crop or "")
                if bname.startswith("upload_") or bname in VERIFIED_CLEAN_CROPS:
                    continue
                should_drop = bname in KNOWN_POISONED_CROPS or target_upper.startswith("VISITOR_")
                if not should_drop and query_vec is not None:
                    try:
                        arr = np.asarray(json.loads(r.embedding_json), dtype=np.float32).flatten()
                        norm = np.linalg.norm(arr)
                        if norm > 0:
                            arr = arr / norm
                        if float(np.dot(query_vec, arr)) >= 0.40:
                            should_drop = True
                    except Exception:
                        should_drop = True
                if not should_drop and bname.startswith("enrolled_"):
                    disk_p = os.path.join(FACE_CROPS_DIR, bname)
                    if not os.path.exists(disk_p):
                        should_drop = True
                    else:
                        c_img = cv2.imread(disk_p)
                        if c_img is None:
                            should_drop = True
                        else:
                            ph, pw = c_img.shape[:2]
                            pad = cv2.copyMakeBorder(
                                c_img, int(ph * 0.25), int(ph * 0.25), int(pw * 0.25), int(pw * 0.25), cv2.BORDER_REPLICATE
                            )
                            v_chk, _, _ = self.extract_face_data(pad, min_score=0.84, min_raw_px=28, min_sharpness=28.0)
                            if v_chk is None:
                                should_drop = True
                if should_drop:
                    if bname:
                        disk_p = os.path.join(FACE_CROPS_DIR, bname)
                        if os.path.exists(disk_p):
                            try:
                                os.remove(disk_p)
                            except OSError:
                                pass
                    await session.delete(r)
                    removed += 1
            if removed:
                await session.commit()
        if removed:
            await self.sync_gallery_from_db_and_disk()
        return removed

    async def sanitize_gallery_on_startup(self) -> dict:
        """
        Audits KnownFace rows on startup to purge corrupted/blurry/back-of-head dynamic enrollments
        (`enrolled_*` and `face_VISITOR_*`) while preserving verified family reference portraits.
        """
        if not self.enabled:
            return {"purged_db": 0, "purged_files": 0}

        purged_db = 0
        purged_names: list[str] = []
        surviving_files: set[str] = set()

        async with AsyncSessionLocal() as session:
            res = await session.execute(select(KnownFace))
            rows = res.scalars().all()
            for r in rows:
                role = (r.role or "whitelist").lower()
                if role == "watchlist_auto":
                    continue

                bname = os.path.basename(r.sample_crop or "")
                if bname in VERIFIED_CLEAN_CROPS or bname.startswith("upload_"):
                    surviving_files.add(bname)
                    continue

                drop_reason = None
                if bname in KNOWN_POISONED_CROPS:
                    drop_reason = "known_occluded_or_blurred"
                elif bname.startswith(("enrolled_", "face_VISITOR_")):
                    disk_p = os.path.join(FACE_CROPS_DIR, bname)
                    if not os.path.exists(disk_p):
                        drop_reason = "missing_crop_file"
                    else:
                        img = cv2.imread(disk_p)
                        if img is None or img.size == 0:
                            drop_reason = "unreadable_crop"
                        else:
                            ih, iw = img.shape[:2]
                            if iw < 70 or ih < 70:
                                drop_reason = f"too_small_{iw}x{ih}"
                            else:
                                padded = cv2.copyMakeBorder(
                                    img,
                                    int(ih * 0.25),
                                    int(ih * 0.25),
                                    int(iw * 0.25),
                                    int(iw * 0.25),
                                    cv2.BORDER_REPLICATE,
                                )
                                vec_chk, _, _ = self.extract_face_data(
                                    padded,
                                    min_score=0.84,
                                    min_raw_px=28,
                                    min_sharpness=28.0,
                                )
                                if vec_chk is None:
                                    drop_reason = "failed_anatomical_or_sharpness_gate"

                if drop_reason:
                    purged_names.append(f"{r.name}({bname}:{drop_reason})")
                    await session.delete(r)
                    purged_db += 1
                elif bname:
                    surviving_files.add(bname)

            if purged_db:
                await session.commit()

        purged_files = 0
        for fname in os.listdir(FACE_CROPS_DIR):
            if not fname.endswith(".jpg"):
                continue
            if fname in surviving_files or fname in VERIFIED_CLEAN_CROPS or fname.startswith("upload_"):
                continue
            if fname in KNOWN_POISONED_CROPS or fname.startswith(("enrolled_", "face_VISITOR_")):
                try:
                    os.remove(os.path.join(FACE_CROPS_DIR, fname))
                    purged_files += 1
                except OSError:
                    pass

        self._sanitized_once = True
        await self.sync_gallery_from_db_and_disk()
        retained_wl = sorted({g["name"] for g in self.gallery if g["role"] == "whitelist"})
        print(
            f"[FACE SANITIZER] Purged {purged_db} poisoned DB vectors & {purged_files} orphan crops. "
            f"Retained whitelist: {retained_wl}. Details: {purged_names[:12]}"
        )
        return {"purged_db": purged_db, "purged_files": purged_files, "retained_whitelist": retained_wl}

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
