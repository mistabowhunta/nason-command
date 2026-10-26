import { useEffect, useState } from 'react';
import { useTelemetryWs } from './hooks/useTelemetryWs';
import { StatusHeader } from './components/StatusHeader';
import { VideoTile } from './components/VideoTile';
import { EvidenceRail } from './components/EvidenceRail';
import { FaceGallery } from './components/FaceGallery';
import { CommandPrompt } from './components/CommandPrompt';
import type { CameraScheduleConfig, CameraZone } from './types/telemetry';
import { Camera, Eye, LayoutGrid, PanelRightClose, PanelRightOpen, Users } from 'lucide-react';

const HOST_IP = window.location.hostname || '192.168.1.27';

const CAMERAS = [
  { title: 'DRIVEWAY', streamName: 'annke_1' },
  { title: 'LOFT', streamName: 'tapo_1' },
  { title: 'LIVING_ROOM', streamName: 'tapo_2' },
];

export default function App() {
  const { telemetry, status } = useTelemetryWs(`ws://${HOST_IP}:8080/ws/telemetry`);
  const [focusedStream, setFocusedStream] = useState<string | null>(null);
  const [desktopView, setDesktopView] = useState<'matrix' | 'faces'>('matrix');
  const [showRail, setShowRail] = useState<boolean>(true);
  const [mobileTab, setMobileTab] = useState<'feeds' | 'killfeed' | 'faces'>('feeds');
  const [railRefresh, setRailRefresh] = useState<number>(0);
  const [zonesByCam, setZonesByCam] = useState<Record<string, CameraZone[]>>({});
  const [schedules, setSchedules] = useState<Record<string, CameraScheduleConfig>>({});
  const detections = telemetry?.detections || {};

  useEffect(() => {
    fetch(`http://${HOST_IP}:8080/api/zones`)
      .then((res) => (res.ok ? res.json() : {}))
      .then((data) => setZonesByCam(data))
      .catch((err) => console.error('Failed to load camera zones:', err));

    const fetchSchedules = () => {
      fetch(`http://${HOST_IP}:8080/api/cameras/schedules`)
        .then((res) => (res.ok ? res.json() : {}))
        .then((data) => setSchedules(data))
        .catch((err) => console.error('Failed to load schedules:', err));
    };
    fetchSchedules();
    const timer = window.setInterval(fetchSchedules, 15000);
    return () => window.clearInterval(timer);
  }, []);

  const primaryCam = CAMERAS.find((c) => c.streamName === focusedStream);
  const secondaryCams = CAMERAS.filter((c) => c.streamName !== focusedStream);

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col pb-12 lg:pb-0">
      <StatusHeader telemetry={telemetry} status={status} />

      <div className="flex-1 flex overflow-hidden">
        {/* Main Viewport */}
        <div
          className={`flex-1 flex flex-col p-3 sm:p-6 gap-4 overflow-y-auto ${
            mobileTab === 'killfeed' ? 'hidden lg:flex' : 'flex'
          }`}
        >
          {/* Top Navigation / Mode Bar */}
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <button
                onClick={() => {
                  setDesktopView('matrix');
                  setMobileTab('feeds');
                }}
                className={`px-3 py-1 rounded text-xs font-mono font-bold border flex items-center gap-1.5 cursor-pointer transition-colors ${
                  desktopView === 'matrix' && mobileTab !== 'faces'
                    ? 'bg-emerald-600 text-white border-emerald-400'
                    : 'bg-zinc-900 hover:bg-zinc-800 text-zinc-300 border-zinc-800'
                }`}
              >
                <Camera className="w-3.5 h-3.5" />
                TACTICAL MATRIX
              </button>
              <button
                onClick={() => {
                  setDesktopView('faces');
                  setMobileTab('faces');
                }}
                className={`px-3 py-1 rounded text-xs font-mono font-bold border flex items-center gap-1.5 cursor-pointer transition-colors ${
                  desktopView === 'faces' || mobileTab === 'faces'
                    ? 'bg-emerald-600 text-white border-emerald-400'
                    : 'bg-zinc-900 hover:bg-zinc-800 text-zinc-300 border-zinc-800'
                }`}
              >
                <Users className="w-3.5 h-3.5" />
                FACE GALLERY & WATCHLIST
              </button>
            </div>

            <div className="flex items-center gap-2">
              {focusedStream && desktopView === 'matrix' && (
                <button
                  onClick={() => setFocusedStream(null)}
                  className="bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 px-2.5 py-1 rounded text-xs font-mono text-zinc-200 flex items-center gap-1.5 cursor-pointer"
                >
                  <LayoutGrid className="w-3.5 h-3.5 text-emerald-400" />
                  3-CAM GRID
                </button>
              )}
              <button
                onClick={() => setShowRail((prev) => !prev)}
                className="hidden lg:flex bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 px-2.5 py-1 rounded text-xs font-mono text-zinc-300 items-center gap-1.5 cursor-pointer"
              >
                {showRail ? <PanelRightClose className="w-3.5 h-3.5 text-emerald-400" /> : <PanelRightOpen className="w-3.5 h-3.5 text-emerald-400" />}
                {showRail ? 'HIDE KILL-FEED' : 'SHOW KILL-FEED'}
              </button>
            </div>
          </div>

          {desktopView === 'faces' || mobileTab === 'faces' ? (
            <FaceGallery hostIp={HOST_IP} />
          ) : focusedStream && primaryCam ? (
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 sm:gap-6 items-start">
              <div className="lg:col-span-3">
                <VideoTile
                  title={primaryCam.title}
                  streamName={primaryCam.streamName}
                  hostIp={HOST_IP}
                  detections={detections[primaryCam.streamName] || []}
                  zones={zonesByCam[primaryCam.streamName] || []}
                  onZonesUpdated={setZonesByCam}
                  schedule={schedules[primaryCam.streamName]}
                  onScheduleUpdated={setSchedules}
                  isFocused={true}
                  onToggleFocus={() => setFocusedStream(null)}
                />
              </div>
              <div className="lg:col-span-1 flex flex-col gap-4 sm:gap-6">
                {secondaryCams.map((cam) => (
                  <VideoTile
                    key={cam.streamName}
                    title={cam.title}
                    streamName={cam.streamName}
                    hostIp={HOST_IP}
                    detections={detections[cam.streamName] || []}
                    zones={zonesByCam[cam.streamName] || []}
                    onZonesUpdated={setZonesByCam}
                    schedule={schedules[cam.streamName]}
                    onScheduleUpdated={setSchedules}
                    isFocused={false}
                    onToggleFocus={() => setFocusedStream(cam.streamName)}
                  />
                ))}
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4 sm:gap-6 content-start">
              {CAMERAS.map((cam) => (
                <VideoTile
                  key={cam.streamName}
                  title={cam.title}
                  streamName={cam.streamName}
                  hostIp={HOST_IP}
                  detections={detections[cam.streamName] || []}
                  zones={zonesByCam[cam.streamName] || []}
                  onZonesUpdated={setZonesByCam}
                  schedule={schedules[cam.streamName]}
                  onScheduleUpdated={setSchedules}
                  isFocused={false}
                  onToggleFocus={() => setFocusedStream(cam.streamName)}
                />
              ))}
            </div>
          )}
        </div>

        {/* Desktop Side Rail */}
        {showRail && (
          <div className="hidden lg:block">
            <EvidenceRail hostIp={HOST_IP} refreshTrigger={railRefresh} />
          </div>
        )}

        {/* Mobile Full-Screen Kill-Feed Tab */}
        {mobileTab === 'killfeed' && (
          <div className="flex-1 lg:hidden overflow-y-auto">
            <EvidenceRail hostIp={HOST_IP} refreshTrigger={railRefresh} fullWidthMobile={true} />
          </div>
        )}
      </div>

      <CommandPrompt
        hostIp={HOST_IP}
        onFocusStream={(s) => {
          setFocusedStream(s);
          setDesktopView('matrix');
          setMobileTab('feeds');
        }}
        onRefreshRail={() => setRailRefresh((prev) => prev + 1)}
      />

      {/* Mobile 3-Tab Bottom Navigation Bar */}
      <nav className="lg:hidden fixed bottom-0 left-0 right-0 bg-zinc-950 border-t border-zinc-800 flex items-center justify-around py-2 z-40 font-mono text-[11px]">
        <button
          onClick={() => {
            setMobileTab('feeds');
            setDesktopView('matrix');
          }}
          className={`flex items-center gap-1 px-3 py-1 rounded font-bold ${
            mobileTab === 'feeds' ? 'bg-emerald-600 text-white' : 'text-zinc-400'
          }`}
        >
          <Camera className="w-3.5 h-3.5" />
          CAMERAS
        </button>
        <button
          onClick={() => setMobileTab('killfeed')}
          className={`flex items-center gap-1 px-3 py-1 rounded font-bold ${
            mobileTab === 'killfeed' ? 'bg-emerald-600 text-white' : 'text-zinc-400'
          }`}
        >
          <Eye className="w-3.5 h-3.5" />
          KILL-FEED
        </button>
        <button
          onClick={() => {
            setMobileTab('faces');
            setDesktopView('faces');
          }}
          className={`flex items-center gap-1 px-3 py-1 rounded font-bold ${
            mobileTab === 'faces' ? 'bg-emerald-600 text-white' : 'text-zinc-400'
          }`}
        >
          <Users className="w-3.5 h-3.5" />
          FACES
        </button>
      </nav>
    </div>
  );
}
