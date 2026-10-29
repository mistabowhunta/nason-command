import asyncio
from collections import deque
import glob
import math
import os
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import cv2
import numpy as np
import onnxruntime as ort
from app.db.session import AsyncSessionLocal
from app.db.models import DetectionEvent
from app.agent.orchestrator import deep_inspect_detection
from app.services.face_engine import face_engine

cv2.setNumThreads(1)

LOCAL_TZ = ZoneInfo("America/Los_Angeles")
RTSP_BASE = os.getenv("MEDIAMTX_RTSP_BASE", "rtsp://mediamtx:8554")
MODEL_PATH = "/app/models/yolov8s.onnx" if os.path.exists("/app/models/yolov8s.onnx") else "/app/models/yolov8n.onnx"
SNAPSHOT_DIR = "/tmp/command_snapshots"
MAX_STORAGE_BYTES = 2048 * 1024 * 1024
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

YOLO_CLASS_MAP: dict[int, tuple[str, str]] = {
    0: ("person", "person"),
    1: ("vehicle", "bicycle"),
    2: ("vehicle", "car"),
    3: ("vehicle", "motorcycle"),
    5: ("vehicle", "suv/van"),
    7: ("vehicle", "truck"),
    14: ("animal", "bird"),
    15: ("animal", "cat"),
    16: ("animal", "dog"),
}

MIN_CONF_BY_CATEGORY = {"person": 0.45, "vehicle": 0.45, "animal": 0.56}


def get_snapshots_storage_mb() -> float:
    total = sum(os.path.getsize(f) for f in glob.glob(f"{SNAPSHOT_DIR}/*") if os.path.exists(f))
    return round(total / (1024.0 * 1024.0), 1)


def enforce_storage_limits():
    files, now, total_bytes = [], time.time(), 0
    for path in glob.glob(f"{SNAPSHOT_DIR}/*"):
        try:
            st = os.stat(path)
            if now - st.st_mtime > 604800:
                os.remove(path)
                continue
            files.append((st.st_mtime, st.st_size, path))
            total_bytes += st.st_size
        except OSError:
            pass
    if total_bytes > MAX_STORAGE_BYTES:
        files.sort(key=lambda x: x[0])
        for _, sz, path in files:
            try:
                os.remove(path)
                total_bytes -= sz
                if total_bytes <= MAX_STORAGE_BYTES * 0.85:
                    break
            except OSError:
                pass


def compute_iou(a: dict, b: dict) -> float:
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / max(0.0001, (a["w"] * a["h"]) + (b["w"] * b["h"]) - inter)


def point_in_polygon(x: float, y: float, poly: list[dict]) -> bool:
    n = len(poly)
    if n < 3:
        return False
    inside = False
    p1x, p1y = float(poly[0]["x"]), float(poly[0]["y"])
    for i in range(1, n + 1):
        p2x, p2y = float(poly[i % n]["x"]), float(poly[i % n]["y"])
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x if p1y != p2y else p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def segments_intersect(ax: float, ay: float, bx: float, by: float, cx: float, cy: float, dx: float, dy: float) -> bool:
    def ccw(p1x, p1y, p2x, p2y, p3x, p3y):
        return (p3y - p1y) * (p2x - p1x) > (p2y - p1y) * (p3x - p1x)
    return (ccw(ax, ay, cx, cy, dx, dy) != ccw(bx, by, cx, cy, dx, dy)) and (
        ccw(ax, ay, bx, by, cx, cy) != ccw(ax, ay, bx, by, dx, dy)
    )


def bbox_or_step_touches_polygon(det: dict, poly: list[dict], prev_foot: tuple[float, float] | None = None) -> bool:
    n = len(poly)
    if n < 3:
        return False
    x, y, w, h = det["x"], det["y"], det["w"], det["h"]
    curr_fx, curr_fy = x + w * 0.50, y + h * 0.95
    for px, py in [
        (curr_fx, curr_fy),
        (x + w * 0.20, y + h * 0.93),
        (x + w * 0.80, y + h * 0.93),
        (x + w * 0.50, y + h * 0.80),
        (x + w * 0.50, y + h * 0.65),
        (x + w * 0.50, y + h * 0.50),
    ]:
        if point_in_polygon(px, py, poly):
            return True
    for pt in poly:
        if x <= float(pt["x"]) <= (x + w) and y <= float(pt["y"]) <= (y + h):
            return True
    if prev_foot is not None:
        pfx, pfy = prev_foot
        for i in range(n):
            p1x, p1y = float(poly[i]["x"]), float(poly[i]["y"])
            p2x, p2y = float(poly[(i + 1) % n]["x"]), float(poly[(i + 1) % n]["y"])
            if segments_intersect(pfx, pfy, curr_fx, curr_fy, p1x, p1y, p2x, p2y):
                return True
    return False


def is_within_time_window(now_hm: str, start_hm: str, end_hm: str) -> bool:
    return (start_hm <= now_hm <= end_hm) if start_hm <= end_hm else (now_hm >= start_hm or now_hm <= end_hm)


def bbox_intersection_ratio(det: dict, motion_boxes: list[tuple[float, float, float, float]]) -> float:
    dx1, dy1, dx2, dy2 = det["x"], det["y"], det["x"] + det["w"], det["y"] + det["h"]
    det_area = max(0.0001, det["w"] * det["h"])
    overlap_sum = 0.0
    for mx, my, mw, mh in motion_boxes:
        ix1, iy1 = max(dx1, mx), max(dy1, my)
        ix2, iy2 = min(dx2, mx + mw), min(dy2, my + mh)
        if ix2 > ix1 and iy2 > iy1:
            overlap_sum += (ix2 - ix1) * (iy2 - iy1)
    return min(1.0, overlap_sum / det_area)


class ParkedVehicleRegistry:
    def __init__(self):
        self.parked_footprints: list[dict] = []

    def filter_and_lock_vehicles(self, detections: list[dict]) -> list[dict]:
        now = time.time()
        self.parked_footprints = [p for p in self.parked_footprints if now - p["last_seen"] < 21600.0]
        for det in detections:
            if det["category"] != "vehicle":
                det["is_parked_vehicle"] = False
                continue
            matched_fp = next((fp for fp in self.parked_footprints if compute_iou(det, fp) >= 0.68), None)
            if matched_fp is not None:
                for k in ("x", "y", "w", "h"):
                    matched_fp[k] = matched_fp[k] * 0.85 + det[k] * 0.15
                matched_fp["last_seen"] = now
                matched_fp["sightings"] += 1
                det["is_parked_vehicle"] = matched_fp["sightings"] >= 6
            else:
                self.parked_footprints.append({
                    "x": det["x"], "y": det["y"], "w": det["w"], "h": det["h"], "last_seen": now, "sightings": 1
                })
                det["is_parked_vehicle"] = False
        return detections


class LatestFrameGrabber:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.frame: np.ndarray | None = None
        self.frame_time: float = 0.0
        self.clip_buffer: deque[tuple[float, np.ndarray]] = deque(maxlen=48)
        self.last_buf_time: float = 0.0
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        while self.running:
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                time.sleep(2.0)
                continue
            while self.running and cap.isOpened():
                ret, frm = cap.read()
                if not ret or frm is None or frm.size == 0:
                    break
                now = time.time()
                with self.lock:
                    self.frame = frm
                    self.frame_time = now
                    if now - self.last_buf_time >= 0.125:
                        self.last_buf_time = now
                        self.clip_buffer.append((now, cv2.resize(frm, (960, 540))))
            cap.release()
            time.sleep(1.0)

    def get_latest(self) -> tuple[np.ndarray | None, float]:
        with self.lock:
            return (self.frame.copy(), self.frame_time) if self.frame is not None else (None, 0.0)

    def get_recent_clip_frames(self, since_ts: float) -> list[tuple[float, np.ndarray]]:
        with self.lock:
            return [(t, f.copy()) for t, f in self.clip_buffer if t >= since_ts]

    def stop(self):
        self.running = False


def encode_mp4_replay_async(grabber: LatestFrameGrabber, snap_id: str, trigger_ts: float):
    def _worker():
        time.sleep(2.4)
        frames_with_ts = grabber.get_recent_clip_frames(trigger_ts - 2.6)
        if len(frames_with_ts) < 6:
            return
        out_mp4 = f"{SNAPSHOT_DIR}/clip_{snap_id}.mp4"
        h, w = frames_with_ts[0][1].shape[:2]
        fps = max(6.0, min(12.0, len(frames_with_ts) / max(1.0, frames_with_ts[-1][0] - frames_with_ts[0][0])))
        try:
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", f"{fps:.2f}", "-i", "-",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", out_mp4,
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            for _, frm in frames_with_ts:
                if proc.stdin:
                    proc.stdin.write(frm.tobytes())
            if proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=15)
        except Exception as e:
            print(f"[MP4 REPLAY ERROR] {e}")

    threading.Thread(target=_worker, daemon=True).start()


class TrueDisplacementTracker:
    def __init__(self):
        self.tracks: dict[int, dict] = {}
        self.spatial_cooldowns: list[dict] = []
        self.last_valid_boxes: list[dict] = []
        self.last_valid_time: float = 0.0

    def update(self, frame: np.ndarray, raw_detections: list[dict], motion_boxes: list, cam_zones: list) -> tuple[list[dict], list[dict]]:
        now = time.time()
        self.tracks = {s: t for s, t in self.tracks.items() if now - t["last_seen"] < 25.0}
        self.spatial_cooldowns = [c for c in self.spatial_cooldowns if now - c["time"] < 60.0]

        if not raw_detections:
            if now - self.last_valid_time < 0.45:
                return self.last_valid_boxes, []
            self.last_valid_boxes = []
            return [], []

        unmatched_dets = [d for d in raw_detections[:10] if not d.get("is_parked_vehicle", False)]
        assigned: list[tuple[int, dict]] = []
        used_slots: set[int] = set()

        for max_dist in (0.055, 0.32):
            for det in list(unmatched_dets):
                cx, cy = det["x"] + det["w"] / 2.0, det["y"] + det["h"] / 2.0
                best_slot, best_d = None, max_dist
                for slot, track in self.tracks.items():
                    if slot in used_slots or track["category"] != det["category"] or (now - track["last_seen"] > 1.8):
                        continue
                    dist = math.hypot(cx - track["cx"], cy - track["cy"])
                    if dist < best_d:
                        best_d, best_slot = dist, slot
                if best_slot is not None:
                    used_slots.add(best_slot)
                    tr = self.tracks[best_slot]
                    step_disp = math.hypot(cx - tr["cx"], cy - tr["cy"])
                    det["prev_foot"] = (tr["cx"], tr["foot_y"])
                    tr.update({"cx": cx, "cy": cy, "foot_y": det["y"] + det["h"] * 0.95, "area": det["area"], "last_seen": now})
                    tr["history"].append((cx, cy, step_disp, now))
                    det["step_disp"] = step_disp
                    assigned.append((best_slot, det))
                    unmatched_dets.remove(det)

        for det in unmatched_dets:
            cx, cy = det["x"] + det["w"] / 2.0, det["y"] + det["h"] / 2.0
            slot = 1
            while slot in used_slots or slot in self.tracks:
                slot += 1
            used_slots.add(slot)
            self.tracks[slot] = {
                "cx": cx, "cy": cy, "foot_y": det["y"] + det["h"] * 0.95, "area": det["area"],
                "category": det["category"], "face_name": None, "face_role": None, "face_sightings": 1,
                "breach_latched": False, "breach_logged": False, "last_seen": now,
                "history": deque([(cx, cy, 0.0, now)], maxlen=8),
            }
            det["step_disp"] = 0.0
            det["prev_foot"] = None
            assigned.append((slot, det))

        assigned.sort(key=lambda item: item[0])
        active_output, new_db_events = [], []

        for slot, det in assigned:
            tr = self.tracks[slot]
            hist = tr["history"]
            if len(hist) < 2:
                continue

            if bbox_intersection_ratio(det, motion_boxes) < (0.25 if det["category"] == "vehicle" else 0.025):
                continue

            oldest_cx, oldest_cy, _, _ = hist[0]
            net_disp = math.hypot(tr["cx"] - oldest_cx, tr["cy"] - oldest_cy)
            if net_disp < (0.038 if det["category"] == "vehicle" else 0.022) or det.get("step_disp", 0.0) < (0.008 if det["category"] == "vehicle" else 0.004):
                continue

            if det["category"] == "person":
                fname, frole, _, fsightings = face_engine.identify_person_in_frame(frame, det)
                if fname:
                    tr.update({"face_name": fname, "face_role": frole, "face_sightings": fsightings})

            recognized_name = tr.get("face_name")
            recognized_role = tr.get("face_role")
            recognized_sightings = tr.get("face_sightings", 1)

            crossed_tripwire_now = any(
                z["zone_type"] == "tripwire" and bbox_or_step_touches_polygon(det, z["points"], det.get("prev_foot"))
                for z in cam_zones
            ) and (recognized_role != "whitelist")

            if crossed_tripwire_now:
                tr["breach_latched"] = True
            elif recognized_role == "whitelist":
                tr["breach_latched"] = False

            is_breach = bool(tr.get("breach_latched", False))
            is_watchlist = recognized_role in ("watchlist", "watchlist_auto")

            slot_name = f"Motion_{slot}"
            suffix = "_BREACH" if is_breach else ""
            if recognized_name:
                full_label = f"{slot_name}_{recognized_name}{suffix}"
                sub_type_label = (
                    f"ID: {recognized_name} ({recognized_sightings}x)" if recognized_role == "visitor"
                    else (f"⚠️ WATCHLIST: {recognized_name}" if is_watchlist else f"ID: {recognized_name}")
                )
            else:
                full_label = f"{slot_name}_{det['category']}{suffix}"
                sub_type_label = det["sub_type"]

            box_item = {
                "slot": slot_name, "category": det["category"], "sub_type": sub_type_label, "attributes": "",
                "full_label": full_label, "confidence": round(det["confidence"], 2),
                "x": round(det["x"], 4), "y": round(det["y"], 4), "w": round(det["w"], 4), "h": round(det["h"], 4),
                "breach": is_breach,
            }
            active_output.append(box_item)

            just_breached = is_breach and not tr.get("breach_logged", False)
            already_logged = any(
                c["category"] == det["category"] and math.hypot(tr["cx"] - c["cx"], tr["cy"] - c["cy"]) < 0.14 and (now - c["time"]) < 45.0
                for c in self.spatial_cooldowns
            )
            if just_breached or not already_logged:
                if is_breach:
                    tr["breach_logged"] = True
                self.spatial_cooldowns.append({"category": det["category"], "cx": tr["cx"], "cy": tr["cy"], "time": now})
                new_db_events.append(dict(box_item))

        if active_output:
            self.last_valid_boxes, self.last_valid_time = active_output, now
            return active_output, new_db_events

        if now - self.last_valid_time < 0.45:
            return self.last_valid_boxes, []
        self.last_valid_boxes = []
        return [], []


class EdgeCVEngine:
    def __init__(self, streams: list[str], event_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        self.streams = streams
        self.event_queue = event_queue
        self.loop = loop
        self.running = True
        self.latest_detections: dict[str, list[dict]] = {s: [] for s in streams}
        self.heatmap_traces: dict[str, deque] = {s: deque(maxlen=1500) for s in streams}
        self.heatmap_cleared_at: dict[str, datetime] = {s: datetime.now(timezone.utc) - timedelta(hours=24) for s in streams}
        self.last_heat_sample: dict[str, float] = {s: 0.0 for s in streams}
        self.zones: dict[str, list[dict]] = {s: [] for s in streams}
        self.schedules: dict[str, dict] = {
            s: {"camera_id": s, "mode": "always_record", "start_time": "21:00", "end_time": "06:00"} for s in streams
        }
        self.zones_lock = threading.Lock()
        self.grabbers: list[LatestFrameGrabber] = []
        self.infer_lock = threading.Lock()
        self.last_storage_check = 0.0

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        sess_opts.inter_op_num_threads = 1
        self.ort_session = ort.InferenceSession(MODEL_PATH, sess_options=sess_opts, providers=["CPUExecutionProvider"])
        self.input_name = self.ort_session.get_inputs()[0].name

    def clear_camera_heatmap(self, camera_id: str):
        if camera_id in self.heatmap_traces:
            self.heatmap_traces[camera_id].clear()
        self.heatmap_cleared_at[camera_id] = datetime.now(timezone.utc)

    def set_schedules(self, sched_map: dict[str, dict]):
        with self.zones_lock:
            for s in self.streams:
                if s in sched_map:
                    self.schedules[s] = dict(sched_map[s])

    def get_schedules_status(self) -> dict[str, dict]:
        now_hm = datetime.now(LOCAL_TZ).strftime("%H:%M")
        with self.zones_lock:
            return {s: {**cfg, "currently_recording": self._is_recording_unlocked(cfg, now_hm), "server_time": now_hm} for s, cfg in self.schedules.items()}

    def _is_recording_unlocked(self, cfg: dict, now_hm: str) -> bool:
        mode = cfg.get("mode", "always_record")
        if mode == "paused":
            return False
        if mode == "schedule":
            return is_within_time_window(now_hm, cfg.get("start_time", "21:00"), cfg.get("end_time", "06:00"))
        return True

    def is_camera_recording(self, camera_id: str) -> bool:
        now_hm = datetime.now(LOCAL_TZ).strftime("%H:%M")
        with self.zones_lock:
            return self._is_recording_unlocked(self.schedules.get(camera_id, {"mode": "always_record"}), now_hm)

    def set_zones(self, zones_by_cam: dict[str, list[dict]]):
        with self.zones_lock:
            for s in self.streams:
                self.zones[s] = list(zones_by_cam.get(s, []))

    def get_zones(self, stream_name: str) -> list[dict]:
        with self.zones_lock:
            return list(self.zones.get(stream_name, []))

    def start(self):
        for stream in self.streams:
            threading.Thread(target=self._worker, args=(stream,), daemon=True).start()

    def stop(self):
        self.running = False
        for g in self.grabbers:
            g.stop()

    def _run_yolo(self, frame: np.ndarray) -> list[dict]:
        blob = cv2.resize(frame, (640, 640))
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
        with self.infer_lock:
            preds = self.ort_session.run(None, {self.input_name: blob})[0][0].T

        target_ids = list(YOLO_CLASS_MAP.keys())
        class_scores = preds[:, 4:]
        best_class_idxs = np.argmax(class_scores, axis=1)
        best_scores = class_scores[np.arange(len(preds)), best_class_idxs]

        mask = (best_scores > 0.42) & np.isin(best_class_idxs, target_ids)
        filtered_preds, filtered_scores, filtered_classes = preds[mask], best_scores[mask], best_class_idxs[mask]
        if len(filtered_preds) == 0:
            return []

        boxes_xywh = [
            [float((p[0] - p[2] / 2.0) / 640.0), float((p[1] - p[3] / 2.0) / 640.0), float(p[2] / 640.0), float(p[3] / 640.0)]
            for p in filtered_preds
        ]
        indices = cv2.dnn.NMSBoxes(boxes_xywh, filtered_scores.tolist(), 0.42, 0.45)
        results = []
        if len(indices) > 0:
            for i in np.array(indices).flatten():
                x, y, w, h = boxes_xywh[int(i)]
                x, y = max(0.0, x), max(0.0, y)
                w, h = min(1.0 - x, w), min(1.0 - y, h)
                category, sub_type = YOLO_CLASS_MAP[int(filtered_classes[int(i)])]
                conf = float(filtered_scores[int(i)])
                if conf < MIN_CONF_BY_CATEGORY.get(category, 0.45):
                    continue
                area = w * h
                if (category == "person" and (h < 0.055 or area < 0.0020)) or (category == "vehicle" and (w < 0.045 or area < 0.0025)) or (category == "animal" and area < 0.0025):
                    continue
                results.append({"category": category, "sub_type": sub_type, "confidence": conf, "x": x, "y": y, "w": w, "h": h, "area": area})
        return results

    def _save_snapshots_and_clips(self, stream_name: str, grabber: LatestFrameGrabber, frame: np.ndarray, db_events: list[dict]):
        fh, fw, _ = frame.shape
        now_ts = time.time()
        ts_ms = int(now_ts * 1000)
        if now_ts - self.last_storage_check > 600.0:
            self.last_storage_check = now_ts
            threading.Thread(target=enforce_storage_limits, daemon=True).start()

        for ev in db_events:
            snap_id = f"{stream_name}_{ts_ms}_{ev['slot']}.jpg"
            pad_x, pad_y = ev["w"] * 0.18, ev["h"] * 0.18
            x1, y1 = int(max(0.0, ev["x"] - pad_x) * fw), int(max(0.0, ev["y"] - pad_y) * fh)
            x2, y2 = int(min(1.0, ev["x"] + ev["w"] + pad_x) * fw), int(min(1.0, ev["y"] + ev["h"] + pad_y) * fh)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                qual = 96 if ev["category"] == "vehicle" else 90
                cv2.imwrite(f"{SNAPSHOT_DIR}/crop_{snap_id}", crop, [int(cv2.IMWRITE_JPEG_QUALITY), qual])
                if ev["category"] == "vehicle":
                    py1 = int(max(0.0, ev["y"] + ev["h"] * 0.30) * fh)
                    py2 = int(min(1.0, ev["y"] + ev["h"] + pad_y) * fh)
                    plate_crop = frame[py1:py2, x1:x2]
                    if plate_crop.size > 0:
                        cv2.imwrite(f"{SNAPSHOT_DIR}/plate_{snap_id}", plate_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
            cv2.imwrite(f"{SNAPSHOT_DIR}/full_{snap_id}", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
            ev["attributes"] = snap_id
            if ev["category"] in ("person", "vehicle", "delivery", "animal"):
                encode_mp4_replay_async(grabber, snap_id, now_ts)

    def _worker(self, stream_name: str):
        grabber = LatestFrameGrabber(f"{RTSP_BASE}/{stream_name}")
        self.grabbers.append(grabber)
        bg_sub = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=20, detectShadows=False)
        tracker = TrueDisplacementTracker()
        parked_registry = ParkedVehicleRegistry()
        last_processed_time = 0.0

        while self.running:
            time.sleep(0.14)
            frame, f_time = grabber.get_latest()
            if frame is None or f_time <= last_processed_time:
                continue
            last_processed_time = f_time

            small_blur = cv2.GaussianBlur(cv2.resize(frame, (640, 360)), (5, 5), 0)
            fg_mask = cv2.dilate(cv2.morphologyEx(bg_sub.apply(small_blur, learningRate=0.004), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), None, iterations=2)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_boxes = [
                (mx / 640.0, my / 360.0, mw / 640.0, mh / 360.0)
                for c in contours if cv2.contourArea(c) >= 85
                for mx, my, mw, mh in [cv2.boundingRect(c)]
            ]
            cam_zones = self.get_zones(stream_name)
            if not motion_boxes:
                active_out, _ = tracker.update(frame, [], [], cam_zones)
                self.latest_detections[stream_name] = active_out
                continue

            candidate_detections = [
                det for det in self._run_yolo(frame)
                if not any(z["zone_type"] == "exclusion" and point_in_polygon(det["x"] + det["w"] / 2.0, det["y"] + det["h"] / 2.0, z["points"]) for z in cam_zones)
            ]
            candidate_detections = parked_registry.filter_and_lock_vehicles(candidate_detections)
            # If YOLO sees a 'cat' when a person is in the frame (or on DRIVEWAY where dogs are walked on leashes), upgrade 'cat' -> 'dog'
            has_person_in_frame = any(d["category"] == "person" for d in candidate_detections)
            for d in candidate_detections:
                if d["category"] == "animal" and d["sub_type"] == "cat" and (has_person_in_frame or stream_name == "annke_1"):
                    d["sub_type"] = "dog"
            candidate_detections.sort(key=lambda d: d["area"], reverse=True)

            active_out, db_events = tracker.update(frame, candidate_detections, motion_boxes, cam_zones)
            now_heat = time.time()
            if now_heat - self.last_heat_sample.get(stream_name, 0.0) >= 0.55:
                sampled_any = False
                for det in candidate_detections:
                    if not det.get("is_parked_vehicle") and (det["category"] in ("person", "animal") or any(a["slot"] for a in active_out if a["category"] == "vehicle")):
                        self.heatmap_traces[stream_name].append({
                            "x": round(det["x"] + det["w"] * 0.5, 4),
                            "y": round(det["y"] + (det["h"] * 0.88 if det["category"] in ("person", "animal") else det["h"] * 0.5), 4),
                            "category": det["category"], "breach": False, "ts": now_heat,
                        })
                        sampled_any = True
                if sampled_any:
                    self.last_heat_sample[stream_name] = now_heat

            self.latest_detections[stream_name] = active_out
            if db_events and self.is_camera_recording(stream_name):
                self._save_snapshots_and_clips(stream_name, grabber, frame, db_events)
                asyncio.run_coroutine_threadsafe(self.event_queue.put((stream_name, db_events)), self.loop)


async def db_persistence_worker(event_queue: asyncio.Queue):
    while True:
        stream_name, events = await event_queue.get()
        try:
            inserted_ids = []
            async with AsyncSessionLocal() as session:
                for ev in events:
                    row = DetectionEvent(
                        timestamp=datetime.now(timezone.utc), camera_id=stream_name, motion_slot=ev["slot"],
                        category=ev["category"], sub_type=ev["sub_type"], attributes=ev.get("attributes", ""),
                        full_label=ev["full_label"], confidence=ev["confidence"],
                        bbox_x=ev["x"], bbox_y=ev["y"], bbox_w=ev["w"], bbox_h=ev["h"],
                    )
                    session.add(row)
                    await session.flush()
                    inserted_ids.append(row.id)
                await session.commit()
            for ev_id in inserted_ids:
                asyncio.create_task(deep_inspect_detection(ev_id))
        except Exception as e:
            print(f"[DB ERROR] {e}")
