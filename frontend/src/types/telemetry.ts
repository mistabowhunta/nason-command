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
