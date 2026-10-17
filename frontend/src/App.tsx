import { useTelemetryWs } from './hooks/useTelemetryWs';
import { StatusHeader } from './components/StatusHeader';
import { VideoTile } from './components/VideoTile';

const HOST_IP = window.location.hostname || '192.168.1.27';

export default function App() {
  const { telemetry, status } = useTelemetryWs(`ws://${HOST_IP}:8080/ws/telemetry`);
  const detections = telemetry?.detections || {};

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col">
      <StatusHeader telemetry={telemetry} status={status} />

      <main className="flex-1 p-6 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 content-start">
        <VideoTile
          title="GARAGE"
          streamName="annke_1"
          hostIp={HOST_IP}
          detections={detections['annke_1'] || []}
        />
        <VideoTile
          title="EXTERIOR 1"
          streamName="tapo_1"
          hostIp={HOST_IP}
          detections={detections['tapo_1'] || []}
        />
        <VideoTile
          title="EXTERIOR 2"
          streamName="tapo_2"
          hostIp={HOST_IP}
          detections={detections['tapo_2'] || []}
        />
      </main>
    </div>
  );
}
