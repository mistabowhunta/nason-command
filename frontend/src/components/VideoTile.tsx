import React, { useEffect, useRef, useState } from 'react';
import { useWhepClient } from '../hooks/useWhepClient';
import type { BoundingBoxDetection } from '../types/telemetry';
import { Crosshair, Maximize2, Minimize2, RotateCcw } from 'lucide-react';

interface VideoTileProps {
  title: string;
  streamName: string;
  hostIp: string;
  detections?: BoundingBoxDetection[];
  isFocused?: boolean;
  onToggleFocus?: () => void;
}

interface TrailPoint {
  x: number;
  y: number;
  t: number;
}

const CATEGORY_THEME: Record<
  string,
  {
    bracket: string;
    fill: string;
    badge: string;
    stroke: string;
  }
> = {
  person: {
    bracket: 'border-red-500',
    fill: 'bg-red-500/10 border-red-500/30',
    badge: 'bg-red-950/90 text-red-300 border-red-500/60',
    stroke: '#ef4444',
  },
  vehicle: {
    bracket: 'border-sky-400',
    fill: 'bg-sky-400/10 border-sky-400/30',
    badge: 'bg-sky-950/90 text-sky-300 border-sky-400/60',
    stroke: '#38bdf8',
  },
  animal: {
    bracket: 'border-amber-400',
    fill: 'bg-amber-400/10 border-amber-400/30',
    badge: 'bg-amber-950/90 text-amber-300 border-amber-400/60',
    stroke: '#fbbf24',
  },
};

export function VideoTile({
  title,
  streamName,
  hostIp,
  detections = [],
  isFocused = false,
  onToggleFocus,
}: VideoTileProps) {
  const whepUrl = `http://${hostIp}:8889/${streamName}/whep`;
  const { videoRef, streamState } = useWhepClient(whepUrl);

  // Rolling 2.5s centroid trail history keyed by slot (Motion_1, Motion_2...)
  const [trails, setTrails] = useState<Record<string, { category: string; points: TrailPoint[] }>>({});

  // Digital PTZ & Auto-Lock state
  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [autoLock, setAutoLock] = useState<boolean>(false);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const dragStartRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  // Update breadcrumb trails and handle AUTO-LOCK target tracking
  useEffect(() => {
    const now = Date.now();

    setTrails((prev) => {
      const next: Record<string, { category: string; points: TrailPoint[] }> = {};

      // Keep recent points within the last 2,500ms
      for (const [slot, data] of Object.entries(prev)) {
        const fresh = data.points.filter((p) => now - p.t < 2500);
        if (fresh.length > 0) {
          next[slot] = { category: data.category, points: fresh };
        }
      }

      // Append latest centroids
      for (const det of detections) {
        const cx = (det.x + det.w / 2) * 1000;
        const cy = (det.y + det.h / 2) * 1000;
        const existing = next[det.slot]?.points || [];
        const lastPt = existing[existing.length - 1];

        // If target jumped > 30% of screen, reset trail so it doesn't draw a line across the screen
        const jumped = lastPt && Math.hypot(cx - lastPt.x, cy - lastPt.y) > 300;
        const basePoints = jumped ? [] : existing;

        next[det.slot] = {
          category: det.category,
          points: [...basePoints.slice(-14), { x: cx, y: cy, t: now }],
        };
      }

      return next;
    });

    // If AUTO-LOCK is active, smoothly center on primary target (Motion_1)
    if (autoLock) {
      if (detections.length > 0) {
        const primary = detections[0];
        const cxNorm = primary.x + primary.w / 2;
        const cyNorm = primary.y + primary.h / 2;
        const targetZoom = Math.max(zoom, 2.1);
        const maxShift = ((targetZoom - 1) / (2 * targetZoom)) * 100;

        const desiredX = Math.max(-maxShift, Math.min(maxShift, (0.5 - cxNorm) * 100));
        const desiredY = Math.max(-maxShift, Math.min(maxShift, (0.5 - cyNorm) * 100));

        setZoom(targetZoom);
        setPan({ x: desiredX, y: desiredY });
      } else {
        // Ease back to 1x overview when no targets are present
        setZoom(1);
        setPan({ x: 0, y: 0 });
      }
    }
  }, [detections, autoLock]);

  const handleWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (autoLock) return;
    const delta = e.deltaY < 0 ? 0.25 : -0.25;
    setZoom((prev) => {
      const next = Math.min(5, Math.max(1, Number((prev + delta).toFixed(2))));
      if (next === 1) setPan({ x: 0, y: 0 });
      return next;
    });
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (zoom <= 1 || autoLock) return;
    setIsDragging(true);
    dragStartRef.current = { x: e.clientX - pan.x * 4, y: e.clientY - pan.y * 4 };
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isDragging || zoom <= 1 || autoLock) return;
    const maxShift = ((zoom - 1) / (2 * zoom)) * 100;
    const rawX = (e.clientX - dragStartRef.current.x) / 4;
    const rawY = (e.clientY - dragStartRef.current.y) / 4;
    setPan({
      x: Math.max(-maxShift, Math.min(maxShift, rawX)),
      y: Math.max(-maxShift, Math.min(maxShift, rawY)),
    });
  };

  const handleMouseUp = () => setIsDragging(false);

  const resetPtz = () => {
    setAutoLock(false);
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  return (
    <div
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onDoubleClick={onToggleFocus}
      className={`relative rounded-lg overflow-hidden border bg-black aspect-video flex flex-col justify-between select-none transition-colors ${
        detections.length > 0 ? 'border-emerald-500/50' : 'border-zinc-800'
      } ${zoom > 1 && !autoLock ? 'cursor-grab active:cursor-grabbing' : 'cursor-pointer'}`}
    >
      {/* GPU-Accelerated PTZ Transform Canvas (Video + SVG Trails + Tactical Reticles) */}
      <div
        className={`w-full h-full relative origin-center ${
          isDragging ? '' : 'transition-transform duration-300 ease-out'
        }`}
        style={{
          transform: `scale(${zoom}) translate(${pan.x}%, ${pan.y}%)`,
        }}
      >
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover"
        />

        {/* SVG Centroid Trajectory Vector Overlay */}
        <svg
          viewBox="0 0 1000 1000"
          preserveAspectRatio="none"
          className="absolute inset-0 w-full h-full pointer-events-none"
        >
          {Object.entries(trails).map(([slot, trail]) => {
            if (trail.points.length < 2) return null;
            const theme = CATEGORY_THEME[trail.category] || CATEGORY_THEME.vehicle;
            const pointsAttr = trail.points.map((p) => `${p.x},${p.y}`).join(' ');
            return (
              <g key={slot}>
                <polyline
                  fill="none"
                  stroke={theme.stroke}
                  strokeWidth="2.5"
                  strokeDasharray="6 4"
                  strokeOpacity="0.75"
                  points={pointsAttr}
                />
                {trail.points.map((pt, idx) => (
                  <circle
                    key={idx}
                    cx={pt.x}
                    cy={pt.y}
                    r={idx === trail.points.length - 1 ? 4.5 : 2.2}
                    fill={theme.stroke}
                    fillOpacity={(idx + 1) / trail.points.length}
                  />
                ))}
              </g>
            );
          })}
        </svg>

        {/* Tactical Corner-Bracket Reticle Overlays */}
        <div className="absolute inset-0 pointer-events-none overflow-hidden">
          {detections.map((det) => {
            const theme = CATEGORY_THEME[det.category] || CATEGORY_THEME.vehicle;
            const shortSlot = det.slot.replace('Motion_', 'M');
            const pctConf = Math.round(det.confidence * 100);

            return (
              <div
                key={det.slot}
                className={`absolute border transition-all duration-150 ease-out ${theme.fill}`}
                style={{
                  left: `${det.x * 100}%`,
                  top: `${det.y * 100}%`,
                  width: `${det.w * 100}%`,
                  height: `${det.h * 100}%`,
                }}
              >
                {/* 4 Tactical L-Corner Brackets */}
                <div className={`absolute -top-0.5 -left-0.5 w-3 h-3 border-t-2 border-l-2 ${theme.bracket}`} />
                <div className={`absolute -top-0.5 -right-0.5 w-3 h-3 border-t-2 border-r-2 ${theme.bracket}`} />
                <div className={`absolute -bottom-0.5 -left-0.5 w-3 h-3 border-b-2 border-l-2 ${theme.bracket}`} />
                <div className={`absolute -bottom-0.5 -right-0.5 w-3 h-3 border-b-2 border-r-2 ${theme.bracket}`} />

                {/* Center Crosshair Target Pip */}
                <div className="absolute inset-0 flex items-center justify-center opacity-60">
                  <div className={`w-1.5 h-1.5 rounded-full ${theme.bracket} border`} />
                </div>

                {/* Compact Tactical Label Pill */}
                <span
                  className={`absolute -top-5 left-0 px-1.5 py-0.5 text-[10px] font-mono font-bold tracking-tight rounded-xs whitespace-nowrap border ${theme.badge}`}
                >
                  {shortSlot} // {det.category.toUpperCase()} · {pctConf}%
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Stationary Top HUD Control Bar */}
      <div
        className="absolute top-2 left-2 right-2 flex justify-between items-center z-10"
        onDoubleClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2 pointer-events-none">
          <span className="bg-zinc-950/85 border border-zinc-800 px-2.5 py-1 rounded text-xs font-mono text-zinc-200">
            {title} <span className="text-zinc-500">({streamName})</span>
          </span>
          {zoom > 1 && (
            <span className="bg-sky-950/90 border border-sky-500/40 px-2 py-0.5 rounded text-[10px] font-mono font-bold text-sky-300">
              PTZ {zoom.toFixed(1)}x
            </span>
          )}
        </div>

        {/* Interactive Tactical Controls */}
        <div className="flex items-center gap-1.5">
          {zoom > 1 && (
            <button
              onClick={resetPtz}
              title="Reset Digital Zoom (1.0x)"
              className="bg-zinc-950/85 hover:bg-zinc-800 border border-zinc-700 px-2 py-1 rounded text-[10px] font-mono text-zinc-300 flex items-center gap-1 cursor-pointer"
            >
              <RotateCcw className="w-3 h-3" />
              1x
            </button>
          )}

          <button
            onClick={() => setAutoLock((prev) => !prev)}
            title="Auto-Zoom & Track Primary Target"
            className={`px-2 py-1 rounded text-[10px] font-mono font-bold border flex items-center gap-1 cursor-pointer transition-colors ${
              autoLock
                ? 'bg-amber-500/25 text-amber-300 border-amber-500/60'
                : 'bg-zinc-950/85 hover:bg-zinc-800 text-zinc-400 border-zinc-800'
            }`}
          >
            <Crosshair className="w-3 h-3" />
            {autoLock ? 'LOCKED' : 'LOCK'}
          </button>

          {onToggleFocus && (
            <button
              onClick={onToggleFocus}
              title={isFocused ? 'Return to Grid View' : 'Maximize to Theater View'}
              className="bg-zinc-950/85 hover:bg-zinc-800 border border-zinc-800 p-1 rounded text-zinc-300 cursor-pointer"
            >
              {isFocused ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
            </button>
          )}

          <span
            className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold pointer-events-none ${
              streamState === 'LIVE'
                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
                : streamState === 'CONNECTING'
                ? 'bg-amber-500/20 text-amber-300'
                : 'bg-red-500/20 text-red-400'
            }`}
          >
            {streamState}
          </span>
        </div>
      </div>
    </div>
  );
}
