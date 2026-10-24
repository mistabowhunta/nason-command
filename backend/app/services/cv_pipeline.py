import asyncio
from collections import deque
import math
import os
import threading
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import cv2
import numpy as np
import onnxruntime as ort
from app.db.session import AsyncSessionLocal
from app.db.models import DetectionEvent
from app.agent.orchestrator import deep_inspect_detection

cv2.setNumThreads(1)

LOCAL_TZ = ZoneInfo("America/Los_Angeles")
RTSP_BASE = os.getenv("MEDIAMTX_RTSP_BASE", "rtsp://mediamtx:8554")
MODEL_PATH = "/app/models/yolov8s.onnx" if os.path.exists("/app/models/yolov8s.onnx") else "/app/models/yolov8n.onnx"
SNAPSHOT_DIR = "/tmp/command_snapshots"
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

MIN_CONF_BY_CATEGORY = {
    "person": 0.48,
    "vehicle": 0.45,
    "animal": 0.58,
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


def bbox_touches_polygon(det: dict, poly: list[dict]) -> bool:
    x, y, w, h = det["x"], det["y"], det["w"], det["h"]
    test_points = [
        (x + w * 0.50, y + h * 0.50),
        (x + w * 0.50, y + h * 0.95),
        (x + w * 0.20, y + h * 0.92),
        (x + w * 0.80, y + h * 0.92),
        (x + w * 0.50, y + h * 0.75),
    ]
    for px, py in test_points:
        if point_in_polygon(px, py, poly):
            return True
    for pt in poly:
        vx, vy = float(pt["x"]), float(pt["y"])
        if x <= vx <= (x + w) and y <= vy <= (y + h):
            return True
    return False


def is_within_time_window(now_hm: str, start_hm: str, end_hm: str) -> bool:
    if start_hm <= end_hm:
        return start_hm <= now_hm <= end_hm
    return now_hm >= start_hm or now_hm <= end_hm


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


class TrueDisplacementTracker:
    """
    Requires BOTH:
    1. Active pixel motion inside the bounding box on the current frame (motion_ratio >= 0.25 for vehicles).
    2. True multi-frame spatial travel (>= 7.5% of screen width for vehicles, >= 4.5% for people)
       plus active frame-to-frame step velocity so parked cars can never latch on.
    """
    def __init__(self):
        self.tracks: dict[int, dict] = {}
        self.spatial_cooldowns: list[dict] = []
        self.last_valid_boxes: list[dict] = []
        self.last_valid_time: float = 0.0

    def update(
        self,
        raw_detections: list[dict],
        motion_boxes: list[tuple[float, float, float, float]],
        cam_zones: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        now = time.time()

        # Expire old tracks & spatial cooldowns
        self.tracks = {
            slot: t for slot, t in self.tracks.items() if now - t["last_seen"] < 30.0
        }
        self.spatial_cooldowns = [
            c for c in self.spatial_cooldowns if now - c["time"] < 120.0
        ]

        if not raw_detections:
            if now - self.last_valid_time < 0.40:
                return self.last_valid_boxes, []
            self.last_valid_boxes = []
            return [], []

        unmatched_dets = list(raw_detections[:10])
        assigned: list[tuple[int, dict]] = []
        used_slots: set[int] = set()

        # PASS 1: Tight Stationary Match (< 0.055) so parked cars stay pinned to their resting slot
        for det in list(unmatched_dets):
            cx = det["x"] + det["w"] / 2.0
            cy = det["y"] + det["h"] / 2.0
            best_slot = None
            best_dist = 0.055

            for slot, track in self.tracks.items():
                if slot in used_slots or track["category"] != det["category"]:
                    continue
                dist = math.hypot(cx - track["cx"], cy - track["cy"])
                if dist < best_dist:
                    best_dist = dist
                    best_slot = slot

            if best_slot is not None:
                used_slots.add(best_slot)
                tr = self.tracks[best_slot]
                step_disp = math.hypot(cx - tr["cx"], cy - tr["cy"])
                tr["cx"] = cx
                tr["cy"] = cy
                tr["last_seen"] = now
                tr["history"].append((cx, cy, now))
                det["step_disp"] = step_disp
                assigned.append((best_slot, det))
                unmatched_dets.remove(det)

        # PASS 2: Fast Moving Target Match (up to 0.50 for street vehicles, 0.30 for people)
        for det in list(unmatched_dets):
            cx = det["x"] + det["w"] / 2.0
            cy = det["y"] + det["h"] / 2.0
            best_slot = None
            max_jump = 0.50 if det["category"] == "vehicle" else 0.30
            best_dist = max_jump

            for slot, track in self.tracks.items():
                if slot in used_slots or track["category"] != det["category"]:
                    continue
                if now - track["last_seen"] > 2.0:
                    continue
                dist = math.hypot(cx - track["cx"], cy - track["cy"])
                if dist < best_dist:
                    best_dist = dist
                    best_slot = slot

            if best_slot is not None:
                used_slots.add(best_slot)
                tr = self.tracks[best_slot]
                step_disp = math.hypot(cx - tr["cx"], cy - tr["cy"])
                tr["cx"] = cx
                tr["cy"] = cy
                tr["last_seen"] = now
                tr["history"].append((cx, cy, now))
                det["step_disp"] = step_disp
                assigned.append((best_slot, det))
                unmatched_dets.remove(det)

        # PASS 3: New object registration (starts with 0 displacement so it cannot fire on frame 1)
        for det in unmatched_dets:
            cx = det["x"] + det["w"] / 2.0
            cy = det["y"] + det["h"] / 2.0
            slot = 1
            while slot in used_slots or slot in self.tracks:
                slot += 1
            used_slots.add(slot)
            self.tracks[slot] = {
                "cx": cx,
                "cy": cy,
                "category": det["category"],
                "last_seen": now,
                "history": deque([(cx, cy, now)], maxlen=8),
            }
            det["step_disp"] = 0.0
            assigned.append((slot, det))

        assigned.sort(key=lambda item: item[0])
        active_output = []
        new_db_events = []

        for slot, det in assigned:
            tr = self.tracks[slot]
            hist = tr["history"]
            if len(hist) < 2:
                continue

            # 1. Check live pixel motion overlap inside this bounding box RIGHT NOW
            motion_ratio = bbox_intersection_ratio(det, motion_boxes)
            req_motion = 0.25 if det["category"] == "vehicle" else 0.06
            if motion_ratio < req_motion:
                continue

            # 2. Check net displacement across recent track history AND current frame step velocity
            oldest_cx, oldest_cy, _ = hist[0]
            net_disp = math.hypot(tr["cx"] - oldest_cx, tr["cy"] - oldest_cy)
            step_disp = det.get("step_disp", 0.0)

            req_net_disp = 0.075 if det["category"] == "vehicle" else 0.042
            req_step_disp = 0.016 if det["category"] == "vehicle" else 0.010

            # If it hasn't traveled across the frame AND moved on this frame, it is stationary!
            if net_disp < req_net_disp or step_disp < req_step_disp:
                continue

            # 3. Check Tripwire Breach only for verified moving targets
            is_breach = False
            for z in cam_zones:
                if z["zone_type"] == "tripwire" and bbox_touches_polygon(det, z["points"]):
                    is_breach = True
                    break

            slot_name = f"Motion_{slot}"
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

            # 4. Spatial Cooldown: prevent logging multiple events at the same coordinates
            already_logged_here = any(
                c["category"] == det["category"]
                and math.hypot(tr["cx"] - c["cx"], tr["cy"] - c["cy"]) < 0.14
                and (now - c["time"]) < 60.0
                for c in self.spatial_cooldowns
            )
            if not already_logged_here:
                self.spatial_cooldowns.append({
                    "category": det["category"],
                    "cx": tr["cx"],
                    "cy": tr["cy"],
                    "time": now,
                })
                new_db_events.append(dict(box_item))

        if active_output:
            self.last_valid_boxes = active_output
            self.last_valid_time = now
            return active_output, new_db_events

        if now - self.last_valid_time < 0.40:
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
        self.zones: dict[str, list[dict]] = {s: [] for s in streams}
        self.schedules: dict[str, dict] = {
            s: {"camera_id": s, "mode": "always_record", "start_time": "21:00", "end_time": "06:00"}
            for s in streams
        }
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

    def set_schedules(self, sched_map: dict[str, dict]):
        with self.zones_lock:
            for s in self.streams:
                if s in sched_map:
                    self.schedules[s] = dict(sched_map[s])

    def get_schedules_status(self) -> dict[str, dict]:
        now_hm = datetime.now(LOCAL_TZ).strftime("%H:%M")
        with self.zones_lock:
            out = {}
            for s, cfg in self.schedules.items():
                rec = self._is_recording_unlocked(cfg, now_hm)
                out[s] = {**cfg, "currently_recording": rec, "server_time": now_hm}
            return out

    def _is_recording_unlocked(self, cfg: dict, now_hm: str) -> bool:
        mode = cfg.get("mode", "always_record")
        if mode == "always_record":
            return True
        if mode == "paused":
            return False
        if mode == "schedule":
            return is_within_time_window(now_hm, cfg.get("start_time", "21:00"), cfg.get("end_time", "06:00"))
        return True

    def is_camera_recording(self, camera_id: str) -> bool:
        now_hm = datetime.now(LOCAL_TZ).strftime("%H:%M")
        with self.zones_lock:
            cfg = self.schedules.get(camera_id, {"mode": "always_record"})
            return self._is_recording_unlocked(cfg, now_hm)

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

        mask = (best_scores > 0.42) & np.isin(best_class_idxs, target_ids)
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

        indices = cv2.dnn.NMSBoxes(boxes_xywh, filtered_scores.tolist(), 0.42, 0.45)
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

                if conf < MIN_CONF_BY_CATEGORY.get(category, 0.45):
                    continue

                area = w * h
                if category == "person" and (h < 0.065 or area < 0.0025):
                    continue
                if category == "vehicle" and (w < 0.045 or area < 0.0025):
                    continue
                if category == "animal" and area < 0.0025:
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

        bg_sub = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=22, detectShadows=False)
        tracker = TrueDisplacementTracker()
        last_processed_time = 0.0

        while self.running:
            time.sleep(0.14)
            frame, f_time = grabber.get_latest()
            if frame is None or f_time <= last_processed_time:
                continue
            last_processed_time = f_time

            small = cv2.resize(frame, (640, 360))
            small_blur = cv2.GaussianBlur(small, (5, 5), 0)
            fg_mask = bg_sub.apply(small_blur, learningRate=0.004)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            fg_mask = cv2.dilate(fg_mask, None, iterations=2)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_boxes = []
            for c in contours:
                if cv2.contourArea(c) < 110:
                    continue
                mx, my, mw, mh = cv2.boundingRect(c)
                motion_boxes.append((mx / 640.0, my / 360.0, mw / 640.0, mh / 360.0))

            cam_zones = self.get_zones(stream_name)

            if not motion_boxes:
                active_out, _ = tracker.update([], [], cam_zones)
                self.latest_detections[stream_name] = active_out
                continue

            yolo_dets = self._run_yolo(frame)
            candidate_detections = []

            for det in yolo_dets:
                cx = det["x"] + det["w"] / 2.0
                cy = det["y"] + det["h"] / 2.0

                if any(z["zone_type"] == "exclusion" and point_in_polygon(cx, cy, z["points"]) for z in cam_zones):
                    continue

                candidate_detections.append(det)

            candidate_detections.sort(key=lambda d: d["area"], reverse=True)
            active_out, db_events = tracker.update(candidate_detections, motion_boxes, cam_zones)
            self.latest_detections[stream_name] = active_out

            if db_events and self.is_camera_recording(stream_name):
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
