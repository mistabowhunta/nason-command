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

export interface PersistedDetectionEvent {
  id: number;
  timestamp: string;
  camera_id: string;
  category: string;
  sub_type: string;
  full_label: string;
  confidence: number;
  crop_url: string | null;
  full_url: string | null;
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
