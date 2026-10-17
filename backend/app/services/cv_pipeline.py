import asyncio
import os
import threading
import time
from datetime import datetime, timezone
import cv2
import numpy as np
import onnxruntime as ort
from app.db.session import AsyncSessionLocal
from app.db.models import DetectionEvent

# Prevent OpenCV from spawning excessive threads that starve MediaMTX
cv2.setNumThreads(1)

RTSP_BASE = os.getenv("MEDIAMTX_RTSP_BASE", "rtsp://mediamtx:8554")
MODEL_PATH = "/app/models/yolov8n.onnx"

# Strictly person, vehicle, and animal
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
    17: ("animal", "horse"),
    21: ("animal", "bear"),
}


class LatestFrameGrabber:
    """Continuously drains RTSP frames so CV inference always gets the 0ms real-time frame."""
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
        self.logged_track_keys: set[str] = set()
        self.last_valid_boxes: list[dict] = []
        self.last_valid_time: float = 0.0

    def update(self, raw_detections: list[dict]) -> tuple[list[dict], list[dict]]:
        now = time.time()
        active_output = []
        new_db_events = []

        if not raw_detections:
            # Hold boxes for 400ms so they don't flicker between frames
            if now - self.last_valid_time < 0.4:
                return self.last_valid_boxes, []
            self.last_valid_boxes = []
            return [], []

        for idx, det in enumerate(raw_detections[:8], start=1):
            slot_name = f"Motion_{idx}"
            full_label = f"{slot_name}_{det['category']}"

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
            }
            active_output.append(box_item)

            dedupe_key = f"{full_label}_{int(now // 10)}"
            if dedupe_key not in self.logged_track_keys:
                self.logged_track_keys.add(dedupe_key)
                if len(self.logged_track_keys) > 200:
                    self.logged_track_keys.clear()
                new_db_events.append(box_item)

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
        self.grabbers: list[LatestFrameGrabber] = []
        self.infer_lock = threading.Lock()

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        sess_opts.inter_op_num_threads = 1
        self.ort_session = ort.InferenceSession(
            MODEL_PATH, sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.ort_session.get_inputs()[0].name

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

        mask = (best_scores > 0.40) & np.isin(best_class_idxs, target_ids)
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

        indices = cv2.dnn.NMSBoxes(boxes_xywh, filtered_scores.tolist(), 0.40, 0.45)
        results = []
        if len(indices) > 0:
            for idx in np.array(indices).flatten():
                i = int(idx)
                x, y, w, h = boxes_xywh[i]
                x, y = max(0.0, x), max(0.0, y)
                w, h = min(1.0 - x, w), min(1.0 - y, h)
                cls_id = int(filtered_classes[i])
                category, sub_type = YOLO_CLASS_MAP[cls_id]

                results.append({
                    "category": category,
                    "sub_type": sub_type,
                    "confidence": float(filtered_scores[i]),
                    "x": x, "y": y, "w": w, "h": h,
                    "area": w * h,
                })
        return results

    def _worker(self, stream_name: str):
        rtsp_url = f"{RTSP_BASE}/{stream_name}"
        grabber = LatestFrameGrabber(rtsp_url)
        self.grabbers.append(grabber)

        bg_sub = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=20, detectShadows=False)
        tracker = CameraMotionTracker()
        last_processed_time = 0.0

        while self.running:
            time.sleep(0.20)  # ~5 FPS inspection rate per camera
            frame, f_time = grabber.get_latest()
            if frame is None or f_time <= last_processed_time:
                continue
            last_processed_time = f_time

            small = cv2.resize(frame, (640, 360))
            fg_mask = bg_sub.apply(small, learningRate=0.002)
            fg_mask = cv2.dilate(fg_mask, None, iterations=2)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_boxes = []
            for c in contours:
                if cv2.contourArea(c) < 160:
                    continue
                mx, my, mw, mh = cv2.boundingRect(c)
                motion_boxes.append((mx / 640.0, my / 360.0, mw / 640.0, mh / 360.0))

            if not motion_boxes:
                active_out, _ = tracker.update([])
                self.latest_detections[stream_name] = active_out
                continue

            yolo_dets = self._run_yolo(frame)
            moving_detections = []

            for det in yolo_dets:
                # Ignore the Christmas tree skirt fur pile in the garage on annke_1
                if stream_name == "annke_1" and det["category"] == "animal" and (0.48 < det["x"] < 0.82 and 0.38 < det["y"] < 0.85):
                    continue

                xmin, ymin, xmax, ymax = det["x"], det["y"], det["x"] + det["w"], det["y"] + det["h"]
                if any(
                    not (xmax < mx or xmin > mx + mw or ymax < my or ymin > my + mh)
                    for (mx, my, mw, mh) in motion_boxes
                ):
                    moving_detections.append(det)

            moving_detections.sort(key=lambda d: d["area"], reverse=True)
            active_out, db_events = tracker.update(moving_detections)
            self.latest_detections[stream_name] = active_out

            if db_events:
                asyncio.run_coroutine_threadsafe(
                    self.event_queue.put((stream_name, db_events)), self.loop
                )


async def db_persistence_worker(event_queue: asyncio.Queue):
    while True:
        stream_name, events = await event_queue.get()
        try:
            async with AsyncSessionLocal() as session:
                for ev in events:
                    row = DetectionEvent(
                        timestamp=datetime.now(timezone.utc),
                        camera_id=stream_name,
                        motion_slot=ev["slot"],
                        category=ev["category"],
                        sub_type=ev["sub_type"],
                        attributes="",
                        full_label=ev["full_label"],
                        confidence=ev["confidence"],
                        bbox_x=ev["x"],
                        bbox_y=ev["y"],
                        bbox_w=ev["w"],
                        bbox_h=ev["h"],
                    )
                    session.add(row)
                await session.commit()
        except Exception as e:
            print(f"[DB ERROR] Failed to persist detection: {e}")
