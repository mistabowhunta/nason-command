
<div align="center">

# 🛡️ NASONNATION // COMMAND
**Low-Latency WebRTC Video Management System (VMS), Edge AI Vision Pipeline & Hybrid Autonomy Command Platform**

[![React 18](https://img.shields.io/badge/React_18-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://react.dev/)
[![TypeScript 5](https://img.shields.io/badge/TypeScript_5-007ACC?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![WebRTC WHEP](https://img.shields.io/badge/WebRTC-WHEP_Sub--250ms-333333?style=for-the-badge&logo=webrtc&logoColor=white)](https://www.ietf.org/archive/id/draft-murillo-whep-03.html)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![YOLOv8s ONNX](https://img.shields.io/badge/YOLOv8s-ONNX_Runtime-FF6F00?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![PostgreSQL 16](https://img.shields.io/badge/PostgreSQL_16-316192?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Docker Compose](https://img.shields.io/badge/Docker_Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docs.docker.com/compose/)

</div>

---

## ⚡ Overview

**NasonNation | Command** is a self-hosted, containerized full-stack Video Management System (VMS) and edge-autonomy telemetry platform engineered for real-time situational awareness. 

Designed around the constraints of edge hardware, **Command** decouples ultra-low-latency video transport (**WebRTC WHEP over single-port UDP**) from asynchronous computer vision and AI orchestration (**FastAPI + ONNX Runtime + Hybrid LAN GPU/Cloud LLM routing**). Video streams render at native 30 FPS with sub-250ms glass-to-glass latency while the backend performs motion-gated object detection, persistent centroid tracking, automated evidence capture, and natural-language tactical tool calling.

---

## 📸 Tactical Interface Showcase

> *Screenshot below reflects the live multi-camera deployment running on bare-metal Debian Linux.*

| **3-Camera Tactical Matrix & Kill-Feed Rail** |
| :---: | :---: |
| <img width="1904" height="860" alt="driveway_zoom" src="https://github.com/user-attachments/assets/186b549f-20ee-4a31-85c9-83a3685bc5c8" /><br>*Live WebRTC grid with L-bracket reticles, 2.5s SVG trajectory vectors, and real-time Kill-Feed crop rail.* |

---

## 🏗️ System Architecture

> 📄 **[Download Full 3-Page System Architecture Specification (PDF)](docs/NasonNation_Command_Architecture_Portfolio.pdf)** — *Includes distributed topology, 6-stage AI/biometric pipeline, 2-way outbound Discord Gateway C2, role-based alert matrix, and dual-disk NAND wear-isolation design.*

```mermaid
flowchart LR
    subgraph Cameras["Edge Camera Array (ONVIF / RTSP H.264)"]
        C1["PERIMETER (cam_ext_01) - 24/7 Active + Road Mask + Bumper LPR"]
        C2["INTERIOR_A (cam_int_01) - Configurable Schedule / Away 24/7"]
        C3["INTERIOR_B (cam_int_02) - Configurable Schedule / Away 24/7"]
    end

    subgraph Host["Edge Docker Host (192.168.1.x - Debian Linux / Core i5-6400 / 16GB RAM)"]
        MTX["command-mediamtx (:8554 RTSP | :8889 WHEP | :8189 ICE UDP)"]

        subgraph Backend["command-backend (FastAPI :8080 REST + WebSocket)"]
            FG["LatestFrameGrabber (48-Frame 5s Clip Ring Buffer + 90s Watchdog)"]
            MOG["OpenCV MOG2 Motion Gate + Polygon Exclusion/Tripwire"]
            YOLO["YOLOv8s ONNX + TrueDisplacement & ParkedVehicleRegistry"]
            FACE["FaceEngine (YuNet + SFace 128-D Cosine Biometrics & RBAC)"]
            ORCH["AI Orchestrator (VLM Deep Inspect, False-Alarm Purge, LPR & HITL Teach)"]
            CONSOL["Best-Confidence Consolidator (4.5s Burst Window / 90s Cooldown)"]
            WATCH["Posture & Health Worker (1-Click Away Mode, Dropout Alerts & Silent Sitrep)"]
            MAINT["6h Self-Healing Worker (30GB Quota, 14d Prune, VACUUM & DR Backups)"]
        end

        PG[("command-postgres (:5432 PostgreSQL 16 - Tier 1 SATA SSD)")]
        HDD[("Tier 2 1TB 7200RPM HDD (30GB Media Quota, 5s MP4s, Crops & 7d Backups)")]
        UI["command-frontend (:3000 React + TypeScript + Nginx)"]
    end

    subgraph RemoteAI["Dedicated LAN GPU Node (192.168.1.y:11434)"]
        VLM["NVIDIA RTX 3060 Ti - Ollama (qwen2.5vl:7b VLM + Plate OCR & qwen2.5:3b)"]
    end

    subgraph External["Zero-Inbound-Port Mobile C2 & Alerting"]
        PUSH["Multipart HTTPS Webhook (Full Scene + Target/Plate Crops + 5s MP4)"]
        BOT["2-Way Discord Gateway v10 Bot (/snap, /away, /mute, /status & Auto-Pinned Cheat Sheet)"]
    end

    subgraph Client["Operator Web Client"]
        VIEW["React Tactical Matrix (Sub-250ms WebRTC, 1-Click Away Mode, HITL Teach & 24h Heatmap)"]
    end

    C1 & C2 & C3 -->|"RTSP TCP"| MTX
    MTX ==>|"DTLS-SRTP UDP :8189"| VIEW
    MTX -->|"Internal RTSP"| FG
    FG --> MOG --> YOLO --> FACE --> ORCH
    FG -.->|"Stream Health"| WATCH
    YOLO & ORCH <-->|"Async SQL"| PG
    FG & FACE & MAINT -->|"Sequential I/O"| HDD
    MAINT -->|"VACUUM & Prune"| PG
    ORCH <-->|"LAN HTTP (30s Failover Cache)"| VLM
    ORCH --> CONSOL -->|"Outbound TLS"| PUSH
    WATCH -->|"Silent Sitrep & Watchdog"| PUSH
    WATCH <-->|"Outbound WSS v10"| BOT
    FG -.->|"On-Demand /snap"| BOT
    UI <-->|"HTTP :3000"| VIEW
    ORCH <-->|"6.6Hz WS Telemetry & REST :8080"| VIEW
```

---

## 🎯 Core Capabilities (The 6 Tactical Pillars)

### 1. Tactical Corner-Bracket Reticles & Centroid Vector Trails
* **Euclidean Centroid Slot Locking:** `CameraMotionTracker` matches moving objects across frames by normalized Euclidean distance so `Motion_1` stays locked onto the same physical entity without ID swapping.
* **L-Bracket HUD & Breadcrumb Vectors:** Renders segmented corner brackets, a center target pip, confidence telemetry (`M1 // PERSON · 92%`), and a 2.5-second fading SVG trajectory polyline showing target direction and velocity at zero server CPU cost.

### 2. Click-to-Focus "Theater Mode" & Auto-Tracking Digital PTZ
* **Dynamic Viewport Promotion:** Double-clicking any camera feed promotes it to a 75%-width primary Theater Viewport while stacking secondary cameras vertically on the right rail.
* **Hardware-Accelerated Digital PTZ & `AUTO-LOCK`:** Supports mouse-wheel digital zoom (`1.0x` to `5.0x`) and click-and-drag panning. Engaging **`LOCK`** automatically zooms to `2.1x` and continuously pans the viewport to keep the primary target centered in the crosshairs, easing back to `1.0x` when the scene clears.

### 3. Live "Kill-Feed" Evidence Rail & High-Res Lightbox
* **Automated Crop & Frame Extraction:** When a new target track is committed to PostgreSQL, OpenCV extracts both a padded high-res target crop (`crop_*.jpg`) and the full 1080p frame (`full_*.jpg`).
* **Real-Time Evidence Drawer:** A collapsible right-hand rail streams live target mugshots/vehicle crops with timestamps, camera IDs, and confidence scores. Clicking any card opens the full-frame capture in a modal lightbox.

### 4. Interactive Polygon Exclusion & Perimeter Tripwire Zones
* **In-Browser Zone Editor:** Operators can toggle **Zone Draw Mode** directly on any camera tile to click-and-place normalized polygon vertices (`[(x1,y1), (x2,y2), ...]`) persisted to PostgreSQL.
* **Dual Zone Logic:**
  * **Exclusion Masks (Red):** Suppresses false-positive motion/detections inside designated static regions before inference.
  * **Tripwire Perimeters (Amber):** Triggers high-priority visual/audible alerts when a `person` or `vehicle` centroid breaches a protected polygon.

### 5. Hybrid RTX 3060 Ti VLM "Deep Inspect" & Active Human-in-the-Loop Learning
* **Parasitic LAN GPU Offload with Edge Failover:** The Debian server stays lightweight by offloading multimodal Vision-Language Model (VLM) inference and license plate OCR over gigabit LAN to an **NVIDIA RTX 3060 Ti** running **Ollama** (`qwen2.5vl:7b` / `qwen2.5:3b`), caching probe state and gracefully degrading to local `YOLOv8s + YuNet/SFace` edge captions if the GPU workstation is asleep.
* **VLM False-Alarm Purging & 1-Click Active `TEACH`:** Automatically purges inanimate false triggers (`valid_target == false`) before mobile alerts fire, while 1-click corrective UI presets enroll 128-D facial embeddings and inject camera-specific visual rules into subsequent VLM prompts.

### 6. Zero-Port-Forwarding 2-Way Mobile C2, 1-Click `AWAY MODE` & Stream Watchdog
* **Outbound Discord Gateway v10 Bot (`0` Open Router Ports):** Connects outbound over encrypted WebSocket (`wss://gateway.discord.gg`) scoped strictly to a private guild ID—registering mobile Slash Commands (`/snap`, `/away`, `/mute`, `/unmute`, `/status`, `/sitrep`, `/help`) alongside a SHA-256 hashed, self-updating pinned Command Cheat Sheet.
* **Non-Destructive `AWAY MODE` & Encounter Consolidation:** Arms 24/7 multi-camera recording and full-property alerting with a configurable exit grace timer while preserving saved Home schedules, consolidating multi-frame encounters into a single highest-confidence Discord push containing wide-scene JPEG, close-up crop, plate crop, and 5-second H.264 MP4 replay.
* **Self-Healing Watchdog & Silent Daily Sitrep:** Monitors RTSP frame starvation (>90s dropout/recovery alerts), dispatches a scheduled silent multi-camera Sitrep (`SUPPRESS_NOTIFICATIONS`), and runs a 6-hour maintenance loop enforcing a 30 GB Tier-2 HDD media quota, 14-day pruning, `VACUUM ANALYZE`, and rolling `.json.gz` disaster-recovery backups.

---

## ⚙️ Key Engineering Challenges Solved

1. **WebRTC NAT Traversal in Docker Bridge Networks:**
   Standard Docker bridge networking causes MediaMTX to advertise internal `172.18.x.x` container IPs in SDP answers, breaking LAN browser ICE negotiation. Solved without sacrificing container isolation by pinning single-port UDP muxing (`MTX_WEBRTCLOCALUDPADDRESS=:8189`) and injecting the host LAN IP via `MTX_WEBRTCADDITIONALHOSTS`.
2. **Eliminating RTSP Buffer Lag During CPU Inference:**
   Standard `cv2.VideoCapture.read()` loops block while ONNX inference executes, causing TCP RTSP buffers to accumulate multi-second delays. Solved by decoupling stream ingestion into a dedicated `LatestFrameGrabber` daemon thread per camera that continuously drains the socket, guaranteeing the CV worker always evaluates the `0ms` live frame.
3. **Preventing Event-Loop & Media Starvation on 4-Core Hardware:**
   Capped OpenCV internal threads (`cv2.setNumThreads(1)`), gated `YOLOv8s` execution behind an ultra-fast `0.5ms` `MOG2` pixel-motion check, serialized ONNX Runtime calls with a thread lock (`intra_op_num_threads=2`), isolated sequential MP4/snapshot writes onto a dedicated 1 TB HDD (`/dev/sdb1`) to eliminate boot-SSD wear, and offloaded PostgreSQL `asyncpg` writes to a non-blocking `asyncio.Queue`.

---

## 🚀 Quickstart & Deployment

### 1. Clone & Configure Environment
```bash
git clone git@github.com:mistabowhunta/nason-command.git
cd nason-command
cp .env.example .env
cp mediamtx/mediamtx.example.yml mediamtx/mediamtx.yml
```

### 2. Build & Launch Stack
```bash
sudo docker compose up -d --build
```

### 3. Access Endpoints
* **Command Tactical UI:** `http://<HOST_LAN_IP>:3000`
* **FastAPI OpenAPI / Swagger Docs:** `http://<HOST_LAN_IP>:8080/docs`
* **Recent Detections REST Feed:** `http://<HOST_LAN_IP>:8080/api/detections`
* **MediaMTX WHEP Signaling Port:** `http://<HOST_LAN_IP>:8889/<stream_name>/whep`
