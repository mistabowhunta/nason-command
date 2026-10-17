import { useEffect, useState, useRef } from 'react';
import type { TelemetryHeartbeat, ConnectionState } from '../types/telemetry';

export function useTelemetryWs(wsUrl: string) {
  const [telemetry, setTelemetry] = useState<TelemetryHeartbeat | null>(null);
  const [status, setStatus] = useState<ConnectionState>('CONNECTING');
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let reconnectTimer: number;

    const connect = () => {
      setStatus('CONNECTING');
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => setStatus('ONLINE');

      ws.onmessage = (event) => {
        try {
          const data: TelemetryHeartbeat = JSON.parse(event.data);
          if (data.type === 'HEARTBEAT') {
            setTelemetry(data);
          }
        } catch (err) {
          console.error('Failed to parse telemetry DTO:', err);
        }
      };

      ws.onclose = () => {
        setStatus('OFFLINE');
        reconnectTimer = window.setTimeout(connect, 3000);
      };

      ws.onerror = () => ws.close();
    };

    connect();

    return () => {
      clearTimeout(reconnectTimer);
      wsRef.current?.close();
    };
  }, [wsUrl]);

  return { telemetry, status };
}
