import { useState } from 'react';
import { useTelemetryWs } from './hooks/useTelemetryWs';
import { StatusHeader } from './components/StatusHeader';
import { VideoTile } from './components/VideoTile';
import { LayoutGrid } from 'lucide-react';

const HOST_IP = window.location.hostname || '192.168.1.27';

const CAMERAS = [
  { title: 'GARAGE', streamName: 'annke_1' },
  { title: 'EXTERIOR 1', streamName: 'tapo_1' },
  { title: 'EXTERIOR 2', streamName: 'tapo_2' },
];

export default function App() {
  const { telemetry, status } = useTelemetryWs(`ws://${HOST_IP}:8080/ws/telemetry`);
  const [focusedStream, setFocusedStream] = useState<string | null>(null);
  const detections = telemetry?.detections || {};

  const primaryCam = CAMERAS.find((c) => c.streamName === focusedStream);
  const secondaryCams = CAMERAS.filter((c) => c.streamName !== focusedStream);

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col">
      <StatusHeader telemetry={telemetry} status={status} />

      {focusedStream && primaryCam ? (
        /* Theater Mode: 75% Primary Feed + 25% Stacked Secondary Rail */
        <main className="flex-1 p-6 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono uppercase tracking-wider text-emerald-400">
              // THEATER FOCUS: {primaryCam.title} ({primaryCam.streamName})
            </span>
            <button
              onClick={() => setFocusedStream(null)}
              className="bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 px-3 py-1 rounded text-xs font-mono text-zinc-200 flex items-center gap-1.5 cursor-pointer"
            >
              <LayoutGrid className="w-3.5 h-3.5 text-emerald-400" />
              RETURN TO 3-CAM GRID
            </button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 items-start">
            <div className="lg:col-span-3">
              <VideoTile
                title={primaryCam.title}
                streamName={primaryCam.streamName}
                hostIp={HOST_IP}
                detections={detections[primaryCam.streamName] || []}
                isFocused={true}
                onToggleFocus={() => setFocusedStream(null)}
              />
            </div>

            <div className="lg:col-span-1 flex flex-col gap-6">
              {secondaryCams.map((cam) => (
                <VideoTile
                  key={cam.streamName}
                  title={cam.title}
                  streamName={cam.streamName}
                  hostIp={HOST_IP}
                  detections={detections[cam.streamName] || []}
                  isFocused={false}
                  onToggleFocus={() => setFocusedStream(cam.streamName)}
                />
              ))}
            </div>
          </div>
        </main>
      ) : (
        /* Standard 3-Column Tactical Grid */
        <main className="flex-1 p-6 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 content-start">
          {CAMERAS.map((cam) => (
            <VideoTile
              key={cam.streamName}
              title={cam.title}
              streamName={cam.streamName}
              hostIp={HOST_IP}
              detections={detections[cam.streamName] || []}
              isFocused={false}
              onToggleFocus={() => setFocusedStream(cam.streamName)}
            />
          ))}
        </main>
      )}
    </div>
  );
}
