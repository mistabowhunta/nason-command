import asyncio
import math
import os
import threading
import time
from datetime import datetime, timezone
import cv2
import numpy as np
import onnxruntime as ort
from app.db.session import AsyncSessionLocal
from app.db.models import DetectionEvent
from app.agent.orchestrator import deep_inspect_detection

cv2.setNumThreads(1)

RTSP_BASE = os.getenv("MEDIAMTX_RTSP_BASE", "rtsp://mediamtx:8554")
MODEL_PATH = "/app/models/yolov8s.onnx" if os.path.exists("/app/models/yolov8s.onnx") else "/app/models/yolov8n.onnx"
SNAPSHOT_DIR = "/tmp/command_snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

YOLO_CLASS_MAP: dict[int, tuple[str, str]] = {
    0: ("person", "person"),
    1: ("vehicle", "bicycle"),
    2: ("vehicle", "car"),
    3: ("vehicle", "motorcycle"),
    5: ("vehicle", "bus"),
    7: ("vehicle", "truck"),
    14: ("animal", "bird"),
    15: ("animal", "cat"),
    16: ("animal", "dog"),
}

MIN_CONF_BY_CATEGORY = {
    "person": 0.56,
    "vehicle": 0.58,
    "animal": 0.65,
}


def point_in_polygon(x: float, y: float, poly: list[dict]) -> bool:
    n = len(poly)
    if n < 3:
        return False
    inside = False
    p1x, p1y = float(poly[0]["x"]), float(poly[0]["y"])
    for i in range(1, n + 1):
        p2x, p2y = float(poly[i % n]["x"]), float(poly[i % n]["y"])
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    else:
                        xinters = p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def bbox_intersection_ratio(det: dict, motion_boxes: list[tuple[float, float, float, float]]) -> float:
    dx1, dy1, dx2, dy2 = det["x"], det["y"], det["x"] + det["w"], det["y"] + det["h"]
    det_area = max(0.0001, det["w"] * det["h"])
    overlap_sum = 0.0
    for mx, my, mw, mh in motion_boxes:
        mx2, my2 = mx + mw, my + mh
        ix1, iy1 = max(dx1, mx), max(dy1, my)
        ix2, iy2 = min(dx2, mx2), min(dy2, my2)
        if ix2 > ix1 and iy2 > iy1:
            overlap_sum += (ix2 - ix1) * (iy2 - iy1)
    return min(1.0, overlap_sum / det_area)


class ParkedObjectFilter:
    def __init__(self):
        self.anchors: list[dict] = []

    def is_truly_moving(self, det: dict, motion_ratio: float) -> bool:
        now = time.time()
        cx = det["x"] + det["w"] / 2.0
        cy = det["y"] + det["h"] / 2.0

        self.anchors = [a for a in self.anchors if now - a["last_seen"] < 600.0]

        for a in self.anchors:
            if a["category"] != det["category"]:
                continue
            dist = math.hypot(cx - a["origin_cx"], cy - a["origin_cy"])
            if dist < 0.035:
                a["last_seen"] = now
                a["frames_stationary"] += 1
                if det["category"] == "vehicle":
                    return False
                if a["frames_stationary"] > 10 and motion_ratio < 0.25:
                    return False
                return True

        if det["category"] == "vehicle" and motion_ratio < 0.24:
            self.anchors.append({
                "category": det["category"],
                "origin_cx": cx,
                "origin_cy": cy,
                "last_seen": now,
                "frames_stationary": 5,
            })
            return False

        self.anchors.append({
            "category": det["category"],
            "origin_cx": cx,
            "origin_cy": cy,
            "last_seen": now,
            "frames_stationary": 1,
        })
        return True


class LatestFrameGrabber:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.frame: np.ndarray | None = None
        self.frame_time: float = 0.0
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

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
                with self.lock:
                    self.frame = frm
                    self.frame_time = time.time()

            cap.release()
            time.sleep(1.0)

    def get_latest(self) -> tuple[np.ndarray | None, float]:
        with self.lock:
            if self.frame is None:
                return None, 0.0
            return self.frame.copy(), self.frame_time

    def stop(self):
        self.running = False


class CameraMotionTracker:
    def __init__(self):
        self.active_tracks: dict[int, dict] = {}
        self.logged_track_keys: set[str] = set()
        self.last_valid_boxes: list[dict] = []
        self.last_valid_time: float = 0.0

    def update(self, raw_detections: list[dict]) -> tuple[list[dict], list[dict]]:
        now = time.time()
        self.active_tracks = {
            slot: t for slot, t in self.active_tracks.items() if now - t["last_seen"] < 1.2
        }

        if not raw_detections:
            if now - self.last_valid_time < 0.35:
                return self.last_valid_boxes, []
            self.last_valid_boxes = []
            return [], []

        unmatched_dets = list(raw_detections[:8])
        assigned: list[tuple[int, dict]] = []
        used_slots: set[int] = set()

        for det in list(unmatched_dets):
            cx = det["x"] + det["w"] / 2.0
            cy = det["y"] + det["h"] / 2.0
            best_slot = None
            best_dist = 0.22

            for slot, track in self.active_tracks.items():
                if slot in used_slots or track["category"] != det["category"]:
                    continue
                dist = math.hypot(cx - track["cx"], cy - track["cy"])
                if dist < best_dist:
                    best_dist = dist
                    best_slot = slot

            if best_slot is not None:
                used_slots.add(best_slot)
                prev_hits = self.active_tracks[best_slot].get("hits", 1) + 1
                self.active_tracks[best_slot] = {
                    "cx": cx,
                    "cy": cy,
                    "category": det["category"],
                    "last_seen": now,
                    "hits": prev_hits,
                }
                det["hits"] = prev_hits
                assigned.append((best_slot, det))
                unmatched_dets.remove(det)

        for det in unmatched_dets:
            cx = det["x"] + det["w"] / 2.0
            cy = det["y"] + det["h"] / 2.0
            slot = 1
            while slot in used_slots or slot in self.active_tracks:
                slot += 1
            used_slots.add(slot)
            self.active_tracks[slot] = {
                "cx": cx,
                "cy": cy,
                "category": det["category"],
                "last_seen": now,
                "hits": 1,
            }
            det["hits"] = 1
            assigned.append((slot, det))

        assigned.sort(key=lambda item: item[0])
        active_output = []
        new_db_events = []

        for slot, det in assigned:
            if det.get("hits", 1) < 2:
                continue

            slot_name = f"Motion_{slot}"
            is_breach = bool(det.get("breach", False))
            suffix = "_BREACH" if is_breach else ""
            full_label = f"{slot_name}_{det['category']}{suffix}"

            box_item = {
                "slot": slot_name,
                "category": det["category"],
                "sub_type": det["sub_type"],
                "attributes": "",
                "full_label": full_label,
                "confidence": round(det["confidence"], 2),
                "x": round(det["x"], 4),
                "y": round(det["y"], 4),
                "w": round(det["w"], 4),
                "h": round(det["h"], 4),
                "breach": is_breach,
            }
            active_output.append(box_item)

            dedupe_key = f"{full_label}_{int(now // 15)}"
            if dedupe_key not in self.logged_track_keys:
                self.logged_track_keys.add(dedupe_key)
                if len(self.logged_track_keys) > 200:
                    self.logged_track_keys.clear()
                new_db_events.append(dict(box_item))

        self.last_valid_boxes = active_output
        self.last_valid_time = now
        return active_output, new_db_events


class EdgeCVEngine:
    def __init__(self, streams: list[str], event_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        self.streams = streams
        self.event_queue = event_queue
        self.loop = loop
        self.running = True
        self.latest_detections: dict[str, list[dict]] = {s: [] for s in streams}
        self.zones: dict[str, list[dict]] = {s: [] for s in streams}
        self.paused_persistence: set[str] = set()
        self.zones_lock = threading.Lock()
        self.grabbers: list[LatestFrameGrabber] = []
        self.infer_lock = threading.Lock()

        print(f"[CV ENGINE] Loading ONNX model: {MODEL_PATH}")
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        sess_opts.inter_op_num_threads = 1
        self.ort_session = ort.InferenceSession(
            MODEL_PATH, sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.ort_session.get_inputs()[0].name

    def toggle_pause_camera(self, camera_id: str) -> list[str]:
        with self.zones_lock:
            if camera_id in self.paused_persistence:
                self.paused_persistence.remove(camera_id)
            else:
                self.paused_persistence.add(camera_id)
            return list(self.paused_persistence)

    def get_paused_cameras(self) -> list[str]:
        with self.zones_lock:
            return list(self.paused_persistence)

    def is_camera_paused(self, camera_id: str) -> bool:
        with self.zones_lock:
            return camera_id in self.paused_persistence

    def set_zones(self, zones_by_cam: dict[str, list[dict]]):
        with self.zones_lock:
            for s in self.streams:
                self.zones[s] = list(zones_by_cam.get(s, []))

    def get_zones(self, stream_name: str) -> list[dict]:
        with self.zones_lock:
            return list(self.zones.get(stream_name, []))

    def start(self):
        for stream in self.streams:
            t = threading.Thread(target=self._worker, args=(stream,), daemon=True)
            t.start()

    def stop(self):
        self.running = False
        for g in self.grabbers:
            g.stop()

    def _run_yolo(self, frame: np.ndarray) -> list[dict]:
        blob = cv2.resize(frame, (640, 640))
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        with self.infer_lock:
            outputs = self.ort_session.run(None, {self.input_name: blob})[0][0]
        preds = outputs.T

        target_ids = list(YOLO_CLASS_MAP.keys())
        class_scores = preds[:, 4:]
        best_class_idxs = np.argmax(class_scores, axis=1)
        best_scores = class_scores[np.arange(len(preds)), best_class_idxs]

        mask = (best_scores > 0.54) & np.isin(best_class_idxs, target_ids)
        filtered_preds = preds[mask]
        filtered_scores = best_scores[mask]
        filtered_classes = best_class_idxs[mask]

        if len(filtered_preds) == 0:
            return []

        boxes_xywh = []
        for p in filtered_preds:
            cx, cy, w, h = p[0], p[1], p[2], p[3]
            x1 = float((cx - w / 2.0) / 640.0)
            y1 = float((cy - h / 2.0) / 640.0)
            boxes_xywh.append([x1, y1, float(w / 640.0), float(h / 640.0)])

        indices = cv2.dnn.NMSBoxes(boxes_xywh, filtered_scores.tolist(), 0.54, 0.45)
        results = []
        if len(indices) > 0:
            for idx in np.array(indices).flatten():
                i = int(idx)
                x, y, w, h = boxes_xywh[i]
                x, y = max(0.0, x), max(0.0, y)
                w, h = min(1.0 - x, w), min(1.0 - y, h)
                cls_id = int(filtered_classes[i])
                category, sub_type = YOLO_CLASS_MAP[cls_id]
                conf = float(filtered_scores[i])

                if conf < MIN_CONF_BY_CATEGORY.get(category, 0.56):
                    continue

                area = w * h
                if category == "person" and (h < 0.08 or area < 0.0038):
                    continue
                if category == "vehicle" and (w < 0.07 or area < 0.0065):
                    continue
                if category == "animal" and area < 0.003:
                    continue

                results.append({
                    "category": category,
                    "sub_type": sub_type,
                    "confidence": conf,
                    "x": x, "y": y, "w": w, "h": h,
                    "area": area,
                })
        return results

    def _save_snapshots(self, stream_name: str, frame: np.ndarray, db_events: list[dict]):
        fh, fw, _ = frame.shape
        ts_ms = int(time.time() * 1000)
        for ev in db_events:
            snap_id = f"{stream_name}_{ts_ms}_{ev['slot']}.jpg"
            pad_x = ev["w"] * 0.18
            pad_y = ev["h"] * 0.18
            x1 = int(max(0.0, ev["x"] - pad_x) * fw)
            y1 = int(max(0.0, ev["y"] - pad_y) * fh)
            x2 = int(min(1.0, ev["x"] + ev["w"] + pad_x) * fw)
            y2 = int(min(1.0, ev["y"] + ev["h"] + pad_y) * fh)

            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                cv2.imwrite(f"{SNAPSHOT_DIR}/crop_{snap_id}", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            cv2.imwrite(f"{SNAPSHOT_DIR}/full_{snap_id}", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
            ev["attributes"] = snap_id

    def _worker(self, stream_name: str):
        rtsp_url = f"{RTSP_BASE}/{stream_name}"
        grabber = LatestFrameGrabber(rtsp_url)
        self.grabbers.append(grabber)

        bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=32, detectShadows=False)
        tracker = CameraMotionTracker()
        parked_filter = ParkedObjectFilter()
        last_processed_time = 0.0

        while self.running:
            time.sleep(0.20)
            frame, f_time = grabber.get_latest()
            if frame is None or f_time <= last_processed_time:
                continue
            last_processed_time = f_time

            small = cv2.resize(frame, (640, 360))
            small_blur = cv2.GaussianBlur(small, (5, 5), 0)
            fg_mask = bg_sub.apply(small_blur, learningRate=0.003)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            fg_mask = cv2.dilate(fg_mask, None, iterations=2)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_boxes = []
            for c in contours:
                if cv2.contourArea(c) < 280:
                    continue
                mx, my, mw, mh = cv2.boundingRect(c)
                motion_boxes.append((mx / 640.0, my / 360.0, mw / 640.0, mh / 360.0))

            if not motion_boxes:
                active_out, _ = tracker.update([])
                self.latest_detections[stream_name] = active_out
                continue

            yolo_dets = self._run_yolo(frame)
            cam_zones = self.get_zones(stream_name)
            moving_detections = []

            for det in yolo_dets:
                motion_ratio = bbox_intersection_ratio(det, motion_boxes)
                if motion_ratio < 0.14:
                    continue

                if not parked_filter.is_truly_moving(det, motion_ratio):
                    continue

                cx = det["x"] + det["w"] / 2.0
                cy = det["y"] + det["h"] / 2.0

                is_excluded = False
                is_breach = False
                for z in cam_zones:
                    if point_in_polygon(cx, cy, z["points"]):
                        if z["zone_type"] == "exclusion":
                            is_excluded = True
                            break
                        elif z["zone_type"] == "tripwire":
                            is_breach = True

                if is_excluded:
                    continue

                det["breach"] = is_breach
                moving_detections.append(det)

            moving_detections.sort(key=lambda d: d["area"], reverse=True)
            active_out, db_events = tracker.update(moving_detections)
            self.latest_detections[stream_name] = active_out

            if db_events and not self.is_camera_paused(stream_name):
                self._save_snapshots(stream_name, frame, db_events)
                asyncio.run_coroutine_threadsafe(
                    self.event_queue.put((stream_name, db_events)), self.loop
                )


async def db_persistence_worker(event_queue: asyncio.Queue):
    while True:
        stream_name, events = await event_queue.get()
        try:
            inserted_ids = []
            async with AsyncSessionLocal() as session:
                for ev in events:
                    row = DetectionEvent(
                        timestamp=datetime.now(timezone.utc),
                        camera_id=stream_name,
                        motion_slot=ev["slot"],
                        category=ev["category"],
                        sub_type=ev["sub_type"],
                        attributes=ev.get("attributes", ""),
                        full_label=ev["full_label"],
                        confidence=ev["confidence"],
                        bbox_x=ev["x"],
                        bbox_y=ev["y"],
                        bbox_w=ev["w"],
                        bbox_h=ev["h"],
                    )
                    session.add(row)
                    await session.flush()
                    inserted_ids.append(row.id)
                await session.commit()

            for ev_id in inserted_ids:
                asyncio.create_task(deep_inspect_detection(ev_id))
        except Exception as e:
            print(f"[DB ERROR] Failed to persist detection: {e}")
