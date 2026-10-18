import React, { useEffect, useRef, useState } from 'react';
import { useWhepClient } from '../hooks/useWhepClient';
import type { BoundingBoxDetection, CameraZone, ZonePoint } from '../types/telemetry';
import { Check, Crosshair, Maximize2, Minimize2, RotateCcw, ShieldAlert, ShieldOff, Trash2, X } from 'lucide-react';

interface VideoTileProps {
  title: string;
  streamName: string;
  hostIp: string;
  detections?: BoundingBoxDetection[];
  zones?: CameraZone[];
  onZonesUpdated?: (zonesByCam: Record<string, CameraZone[]>) => void;
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
  zones = [],
  onZonesUpdated,
  isFocused = false,
  onToggleFocus,
}: VideoTileProps) {
  const whepUrl = `http://${hostIp}:8889/${streamName}/whep`;
  const apiBase = `http://${hostIp}:8080`;
  const { videoRef, streamState } = useWhepClient(whepUrl);

  const [trails, setTrails] = useState<Record<string, { category: string; points: TrailPoint[] }>>({});
  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [autoLock, setAutoLock] = useState<boolean>(false);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const dragStartRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  // Interactive Polygon Zone Drawing State
  const [drawMode, setDrawMode] = useState<'exclusion' | 'tripwire' | null>(null);
  const [draftPoints, setDraftPoints] = useState<ZonePoint[]>([]);
  const canvasWrapRef = useRef<HTMLDivElement | null>(null);

  const hasBreach = detections.some((d) => d.breach);

  useEffect(() => {
    const now = Date.now();

    setTrails((prev) => {
      const next: Record<string, { category: string; points: TrailPoint[] }> = {};
      for (const [slot, data] of Object.entries(prev)) {
        const fresh = data.points.filter((p) => now - p.t < 2500);
        if (fresh.length > 0) {
          next[slot] = { category: data.category, points: fresh };
        }
      }
      for (const det of detections) {
        const cx = (det.x + det.w / 2) * 1000;
        const cy = (det.y + det.h / 2) * 1000;
        const existing = next[det.slot]?.points || [];
        const lastPt = existing[existing.length - 1];
        const jumped = lastPt && Math.hypot(cx - lastPt.x, cy - lastPt.y) > 300;
        const basePoints = jumped ? [] : existing;

        next[det.slot] = {
          category: det.category,
          points: [...basePoints.slice(-14), { x: cx, y: cy, t: now }],
        };
      }
      return next;
    });

    if (autoLock && !drawMode) {
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
        setZoom(1);
        setPan({ x: 0, y: 0 });
      }
    }
  }, [detections, autoLock, drawMode]);

  const startDrawing = (type: 'exclusion' | 'tripwire') => {
    setAutoLock(false);
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setDrawMode(type);
    setDraftPoints([]);
  };

  const cancelDrawing = () => {
    setDrawMode(null);
    setDraftPoints([]);
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!drawMode || !canvasWrapRef.current) return;
    e.stopPropagation();
    const rect = canvasWrapRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    setDraftPoints((prev) => [...prev, { x: Number(x.toFixed(4)), y: Number(y.toFixed(4)) }]);
  };

  const saveZone = async () => {
    if (!drawMode || draftPoints.length < 3) return;
    try {
      const res = await fetch(`${apiBase}/api/zones`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          camera_id: streamName,
          name: `${drawMode.toUpperCase()}_${zones.length + 1}`,
          zone_type: drawMode,
          points: draftPoints,
        }),
      });
      if (res.ok && onZonesUpdated) {
        const updated = await res.json();
        onZonesUpdated(updated);
      }
    } catch (err) {
      console.error('Failed to save zone:', err);
    } finally {
      setDrawMode(null);
      setDraftPoints([]);
    }
  };

  const clearZones = async () => {
    try {
      const res = await fetch(`${apiBase}/api/zones/camera/${streamName}`, {
        method: 'DELETE',
      });
      if (res.ok && onZonesUpdated) {
        const updated = await res.json();
        onZonesUpdated(updated);
      }
    } catch (err) {
      console.error('Failed to clear zones:', err);
    }
  };

  const handleWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (autoLock || drawMode) return;
    const delta = e.deltaY < 0 ? 0.25 : -0.25;
    setZoom((prev) => {
      const next = Math.min(5, Math.max(1, Number((prev + delta).toFixed(2))));
      if (next === 1) setPan({ x: 0, y: 0 });
      return next;
    });
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (zoom <= 1 || autoLock || drawMode) return;
    setIsDragging(true);
    dragStartRef.current = { x: e.clientX - pan.x * 4, y: e.clientY - pan.y * 4 };
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isDragging || zoom <= 1 || autoLock || drawMode) return;
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
      ref={canvasWrapRef}
      onClick={handleCanvasClick}
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onDoubleClick={() => !drawMode && onToggleFocus && onToggleFocus()}
      className={`relative rounded-lg overflow-hidden border bg-black aspect-video flex flex-col justify-between select-none transition-all ${
        hasBreach
          ? 'border-red-500 shadow-[0_0_25px_rgba(239,68,68,0.65)]'
          : detections.length > 0
          ? 'border-emerald-500/50'
          : 'border-zinc-800'
      } ${
        drawMode
          ? 'cursor-crosshair ring-2 ring-amber-400/70'
          : zoom > 1 && !autoLock
          ? 'cursor-grab active:cursor-grabbing'
          : 'cursor-pointer'
      }`}
    >
      {/* GPU-Accelerated PTZ Transform Canvas */}
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

        {/* SVG Overlay: Saved Zones + Draft Polygon + Centroid Trajectory Trails */}
        <svg
          viewBox="0 0 1000 1000"
          preserveAspectRatio="none"
          className="absolute inset-0 w-full h-full pointer-events-none"
        >
          {/* Saved Exclusion & Tripwire Polygons */}
          {zones.map((z) => {
            const pts = z.points.map((p) => `${p.x * 1000},${p.y * 1000}`).join(' ');
            const isExcl = z.zone_type === 'exclusion';
            const firstPt = z.points[0];
            return (
              <g key={z.id}>
                <polygon
                  points={pts}
                  fill={isExcl ? 'rgba(239, 68, 68, 0.18)' : 'rgba(245, 158, 11, 0.16)'}
                  stroke={isExcl ? '#ef4444' : '#f59e0b'}
                  strokeWidth="2.5"
                  strokeDasharray={isExcl ? '8 4' : '12 4'}
                />
                {firstPt && (
                  <text
                    x={firstPt.x * 1000 + 8}
                    y={firstPt.y * 1000 + 18}
                    fill={isExcl ? '#fca5a5' : '#fde68a'}
                    fontSize="20"
                    fontFamily="monospace"
                    fontWeight="bold"
                  >
                    {z.name}
                  </text>
                )}
              </g>
            );
          })}

          {/* Active Draft Polygon While Drawing */}
          {drawMode && draftPoints.length > 0 && (
            <g>
              <polygon
                points={draftPoints.map((p) => `${p.x * 1000},${p.y * 1000}`).join(' ')}
                fill={drawMode === 'exclusion' ? 'rgba(239, 68, 68, 0.28)' : 'rgba(245, 158, 11, 0.28)'}
                stroke={drawMode === 'exclusion' ? '#ef4444' : '#f59e0b'}
                strokeWidth="3"
              />
              {draftPoints.map((p, idx) => (
                <circle
                  key={idx}
                  cx={p.x * 1000}
                  cy={p.y * 1000}
                  r="6"
                  fill="#ffffff"
                  stroke="#000000"
                  strokeWidth="2"
                />
              ))}
            </g>
          )}

          {/* 2.5s Target Trajectory Trails */}
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
                className={`absolute border transition-all duration-150 ease-out ${
                  det.breach ? 'bg-red-600/25 border-red-500 animate-pulse' : theme.fill
                }`}
                style={{
                  left: `${det.x * 100}%`,
                  top: `${det.y * 100}%`,
                  width: `${det.w * 100}%`,
                  height: `${det.h * 100}%`,
                }}
              >
                <div className={`absolute -top-0.5 -left-0.5 w-3 h-3 border-t-2 border-l-2 ${theme.bracket}`} />
                <div className={`absolute -top-0.5 -right-0.5 w-3 h-3 border-t-2 border-r-2 ${theme.bracket}`} />
                <div className={`absolute -bottom-0.5 -left-0.5 w-3 h-3 border-b-2 border-l-2 ${theme.bracket}`} />
                <div className={`absolute -bottom-0.5 -right-0.5 w-3 h-3 border-b-2 border-r-2 ${theme.bracket}`} />

                <div className="absolute inset-0 flex items-center justify-center opacity-60">
                  <div className={`w-1.5 h-1.5 rounded-full ${theme.bracket} border`} />
                </div>

                <span
                  className={`absolute -top-5 left-0 px-1.5 py-0.5 text-[10px] font-mono font-bold tracking-tight rounded-xs whitespace-nowrap border ${
                    det.breach ? 'bg-red-600 text-white border-red-300' : theme.badge
                  }`}
                >
                  {shortSlot} // {det.category.toUpperCase()} {det.breach ? '⚠️ BREACH' : `· ${pctConf}%`}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Stationary Top HUD Control Bar */}
      <div
        className="absolute top-2 left-2 right-2 flex justify-between items-center z-10"
        onClick={(e) => e.stopPropagation()}
        onDoubleClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2 pointer-events-none">
          <span className="bg-zinc-950/85 border border-zinc-800 px-2.5 py-1 rounded text-xs font-mono text-zinc-200">
            {title} <span className="text-zinc-500">({streamName})</span>
          </span>
          {hasBreach && (
            <span className="bg-red-600 text-white px-2 py-0.5 rounded text-[10px] font-mono font-bold animate-bounce">
              ⚠️ TRIPWIRE BREACH
            </span>
          )}
          {zoom > 1 && (
            <span className="bg-sky-950/90 border border-sky-500/40 px-2 py-0.5 rounded text-[10px] font-mono font-bold text-sky-300">
              PTZ {zoom.toFixed(1)}x
            </span>
          )}
        </div>

        <div className="flex items-center gap-1">
          {drawMode ? (
            <>
              <span className="bg-amber-950/90 border border-amber-500/50 px-2 py-1 rounded text-[10px] font-mono text-amber-300">
                CLICK {Math.max(0, 3 - draftPoints.length)}+ PTS ({draftPoints.length})
              </span>
              <button
                onClick={saveZone}
                disabled={draftPoints.length < 3}
                className="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 px-2 py-1 rounded text-[10px] font-mono font-bold text-white flex items-center gap-1 cursor-pointer"
              >
                <Check className="w-3 h-3" />
                SAVE
              </button>
              <button
                onClick={cancelDrawing}
                className="bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 px-2 py-1 rounded text-[10px] font-mono text-zinc-300 flex items-center gap-1 cursor-pointer"
              >
                <X className="w-3 h-3" />
              </button>
            </>
          ) : (
            <>
              <button
                onClick={() => startDrawing('exclusion')}
                title="Draw Red Exclusion Mask (Ignore detections inside polygon)"
                className="bg-zinc-950/85 hover:bg-red-950/80 border border-zinc-800 hover:border-red-500/50 px-1.5 py-1 rounded text-[10px] font-mono text-red-300 flex items-center gap-1 cursor-pointer"
              >
                <ShieldOff className="w-3 h-3" />
                +EXCL
              </button>
              <button
                onClick={() => startDrawing('tripwire')}
                title="Draw Amber Perimeter Tripwire (Alert when target enters polygon)"
                className="bg-zinc-950/85 hover:bg-amber-950/80 border border-zinc-800 hover:border-amber-500/50 px-1.5 py-1 rounded text-[10px] font-mono text-amber-300 flex items-center gap-1 cursor-pointer"
              >
                <ShieldAlert className="w-3 h-3" />
                +TRIP
              </button>
              {zones.length > 0 && (
                <button
                  onClick={clearZones}
                  title="Clear all polygons on this camera"
                  className="bg-zinc-950/85 hover:bg-zinc-800 border border-zinc-800 p-1 rounded text-zinc-400 hover:text-red-400 cursor-pointer"
                >
                  <Trash2 className="w-3 h-3" />
                </button>
              )}
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
            </>
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
