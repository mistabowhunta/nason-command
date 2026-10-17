import { useWhepClient } from '../hooks/useWhepClient';
import type { BoundingBoxDetection } from '../types/telemetry';

interface VideoTileProps {
  title: string;
  streamName: string;
  hostIp: string;
  detections?: BoundingBoxDetection[];
}

const CATEGORY_STYLES: Record<string, { border: string; badge: string }> = {
  person: {
    border: 'border-red-500 shadow-[0_0_10px_rgba(239,68,68,0.5)]',
    badge: 'bg-red-600/90 text-white border-red-400',
  },
  vehicle: {
    border: 'border-sky-400 shadow-[0_0_10px_rgba(56,189,248,0.5)]',
    badge: 'bg-sky-600/90 text-white border-sky-400',
  },
  animal: {
    border: 'border-amber-400 shadow-[0_0_10px_rgba(251,191,36,0.5)]',
    badge: 'bg-amber-600/90 text-zinc-950 border-amber-300',
  },
};

export function VideoTile({ title, streamName, hostIp, detections = [] }: VideoTileProps) {
  const whepUrl = `http://${hostIp}:8889/${streamName}/whep`;
  const { videoRef, streamState } = useWhepClient(whepUrl);

  return (
    <div className="relative rounded-lg overflow-hidden border border-zinc-800 bg-black aspect-video flex flex-col justify-between select-none">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-full h-full object-cover"
      />

      {/* Smooth Bounding Box Overlays (Zero Banner Clutter) */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        {detections.map((det) => {
          const style = CATEGORY_STYLES[det.category] || CATEGORY_STYLES.vehicle;
          return (
            <div
              key={det.slot}
              className={`absolute border-2 rounded-sm transition-all duration-150 ease-out ${style.border}`}
              style={{
                left: `${det.x * 100}%`,
                top: `${det.y * 100}%`,
                width: `${det.w * 100}%`,
                height: `${det.h * 100}%`,
              }}
            >
              <span className={`absolute -top-5 left-0 px-1.5 py-0.5 text-[10px] font-mono font-bold rounded-t whitespace-nowrap border-b ${style.badge}`}>
                {det.full_label}
              </span>
            </div>
          );
        })}
      </div>

      {/* Minimal Top Header */}
      <div className="absolute top-2 left-2 right-2 flex justify-between items-center pointer-events-none">
        <span className="bg-zinc-950/85 border border-zinc-800 px-2.5 py-1 rounded text-xs font-mono text-zinc-200">
          {title} ({streamName})
        </span>
        <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
          streamState === 'LIVE' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40' :
          streamState === 'CONNECTING' ? 'bg-amber-500/20 text-amber-300' : 'bg-red-500/20 text-red-400'
        }`}>
          {streamState}
        </span>
      </div>
    </div>
  );
}
