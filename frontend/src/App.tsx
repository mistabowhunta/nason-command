import { useEffect, useState } from 'react';
import { useTelemetryWs } from './hooks/useTelemetryWs';
import { StatusHeader } from './components/StatusHeader';
import { VideoTile } from './components/VideoTile';
import { EvidenceRail } from './components/EvidenceRail';
import { CommandPrompt } from './components/CommandPrompt';
import type { CameraZone } from './types/telemetry';
import { LayoutGrid, PanelRightClose, PanelRightOpen } from 'lucide-react';

const HOST_IP = window.location.hostname || '192.168.1.27';

const CAMERAS = [
  { title: 'GARAGE / DRIVEWAY', streamName: 'annke_1' },
  { title: 'EXTERIOR 1', streamName: 'tapo_1' },
  { title: 'EXTERIOR 2', streamName: 'tapo_2' },
];

export default function App() {
  const { telemetry, status } = useTelemetryWs(`ws://${HOST_IP}:8080/ws/telemetry`);
  const [focusedStream, setFocusedStream] = useState<string | null>(null);
  const [showRail, setShowRail] = useState<boolean>(true);
  const [railRefresh, setRailRefresh] = useState<number>(0);
  const [zonesByCam, setZonesByCam] = useState<Record<string, CameraZone[]>>({});
  const [pausedCams, setPausedCams] = useState<string[]>([]);
  const detections = telemetry?.detections || {};

  useEffect(() => {
    fetch(`http://${HOST_IP}:8080/api/zones`)
      .then((res) => (res.ok ? res.json() : {}))
      .then((data) => setZonesByCam(data))
      .catch((err) => console.error('Failed to load camera zones:', err));

    fetch(`http://${HOST_IP}:8080/api/cameras/paused`)
      .then((res) => (res.ok ? res.json() : { paused: [] }))
      .then((data) => setPausedCams(data.paused || []))
      .catch((err) => console.error('Failed to load paused cameras:', err));
  }, []);

  const togglePauseCam = async (streamName: string) => {
    try {
      const res = await fetch(`http://${HOST_IP}:8080/api/cameras/${streamName}/pause`, {
        method: 'POST',
      });
      if (res.ok) {
        const data = await res.json();
        setPausedCams(data.paused || []);
      }
    } catch (err) {
      console.error('Failed to toggle camera pause:', err);
    }
  };

  const primaryCam = CAMERAS.find((c) => c.streamName === focusedStream);
  const secondaryCams = CAMERAS.filter((c) => c.streamName !== focusedStream);

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col">
      <StatusHeader telemetry={telemetry} status={status} />

      <div className="flex-1 flex overflow-hidden">
        <div className="flex-1 flex flex-col p-6 gap-4 overflow-y-auto">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono uppercase tracking-wider text-zinc-400">
              {focusedStream && primaryCam
                ? `// THEATER FOCUS: ${primaryCam.title} (${primaryCam.streamName})`
                : '// TACTICAL MATRIX (YOLOv8s + RTX 3060 Ti VLM)'}
            </span>

            <div className="flex items-center gap-2">
              {focusedStream && (
                <button
                  onClick={() => setFocusedStream(null)}
                  className="bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 px-3 py-1 rounded text-xs font-mono text-zinc-200 flex items-center gap-1.5 cursor-pointer"
                >
                  <LayoutGrid className="w-3.5 h-3.5 text-emerald-400" />
                  3-CAM GRID
                </button>
              )}
              <button
                onClick={() => setShowRail((prev) => !prev)}
                className="bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 px-2.5 py-1 rounded text-xs font-mono text-zinc-300 flex items-center gap-1.5 cursor-pointer"
              >
                {showRail ? <PanelRightClose className="w-3.5 h-3.5 text-emerald-400" /> : <PanelRightOpen className="w-3.5 h-3.5 text-emerald-400" />}
                {showRail ? 'HIDE KILL-FEED' : 'SHOW KILL-FEED'}
              </button>
            </div>
          </div>

          {focusedStream && primaryCam ? (
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 items-start">
              <div className="lg:col-span-3">
                <VideoTile
                  title={primaryCam.title}
                  streamName={primaryCam.streamName}
                  hostIp={HOST_IP}
                  detections={detections[primaryCam.streamName] || []}
                  zones={zonesByCam[primaryCam.streamName] || []}
                  onZonesUpdated={setZonesByCam}
                  isPaused={pausedCams.includes(primaryCam.streamName)}
                  onTogglePause={() => togglePauseCam(primaryCam.streamName)}
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
                    zones={zonesByCam[cam.streamName] || []}
                    onZonesUpdated={setZonesByCam}
                    isPaused={pausedCams.includes(cam.streamName)}
                    onTogglePause={() => togglePauseCam(cam.streamName)}
                    isFocused={false}
                    onToggleFocus={() => setFocusedStream(cam.streamName)}
                  />
                ))}
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 content-start">
              {CAMERAS.map((cam) => (
                <VideoTile
                  key={cam.streamName}
                  title={cam.title}
                  streamName={cam.streamName}
                  hostIp={HOST_IP}
                  detections={detections[cam.streamName] || []}
                  zones={zonesByCam[cam.streamName] || []}
                  onZonesUpdated={setZonesByCam}
                  isPaused={pausedCams.includes(cam.streamName)}
                  onTogglePause={() => togglePauseCam(cam.streamName)}
                  isFocused={false}
                  onToggleFocus={() => setFocusedStream(cam.streamName)}
                />
              ))}
            </div>
          )}
        </div>

        {showRail && <EvidenceRail hostIp={HOST_IP} refreshTrigger={railRefresh} />}
      </div>

      <CommandPrompt
        hostIp={HOST_IP}
        onFocusStream={setFocusedStream}
        onRefreshRail={() => setRailRefresh((prev) => prev + 1)}
      />
    </div>
  );
}
