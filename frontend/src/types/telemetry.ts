export interface BoundingBoxDetection {
  slot: string;
  category: 'person' | 'vehicle' | 'animal';
  sub_type: string;
  attributes: string;
  full_label: string;
  confidence: number;
  x: number;
  y: number;
  w: number;
  h: number;
  breach?: boolean;
}

export interface ZonePoint {
  x: number;
  y: number;
}

export interface CameraZone {
  id: number;
  camera_id: string;
  name: string;
  zone_type: 'exclusion' | 'tripwire';
  points: ZonePoint[];
}

export interface CameraScheduleConfig {
  camera_id: string;
  mode: 'always_record' | 'paused' | 'schedule';
  start_time: string;
  end_time: string;
  currently_recording: boolean;
  server_time?: string;
}

export interface PersistedDetectionEvent {
  id: number;
  timestamp: string;
  camera_id: string;
  category: string;
  sub_type: string;
  full_label: string;
  confidence: number;
  bbox_x: number;
  bbox_y: number;
  bbox_w: number;
  bbox_h: number;
  crop_url: string | null;
  full_url: string | null;
  ai_notes?: string | null;
}

export interface TelemetryHeartbeat {
  type: 'HEARTBEAT' | 'PONG';
  timestamp: string;
  uptime_seconds: number;
  active_streams: string[];
  agent_mode: string;
  detections?: Record<string, BoundingBoxDetection[]>;
}

export type ConnectionState = 'CONNECTING' | 'ONLINE' | 'OFFLINE';
