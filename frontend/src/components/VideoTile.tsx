import React, { useEffect, useRef, useState } from 'react';
import { useWhepClient } from '../hooks/useWhepClient';
import type { BoundingBoxDetection, CameraScheduleConfig, CameraZone, HeatmapPoint, ZonePoint } from '../types/telemetry';
import { Check, Clock, Crosshair, Database, Eye, EyeOff, Flame, Maximize2, Minimize2, PauseCircle, RotateCcw, ShieldAlert, ShieldOff, Trash2, X } from 'lucide-react';

interface VideoTileProps {
  title: string;
  streamName: string;
  hostIp: string;
  detections?: BoundingBoxDetection[];
  zones?: CameraZone[];
  onZonesUpdated?: (zonesByCam: Record<string, CameraZone[]>) => void;
  schedule?: CameraScheduleConfig;
  onScheduleUpdated?: (schedules: Record<string, CameraScheduleConfig>) => void;
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
  schedule,
  onScheduleUpdated,
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
  const [showBbox, setShowBbox] = useState<boolean>(true);
  const [showHeatmap, setShowHeatmap] = useState<boolean>(false);
  const [heatmapPoints, setHeatmapPoints] = useState<HeatmapPoint[]>([]);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const dragStartRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  const [drawMode, setDrawMode] = useState<'exclusion' | 'tripwire' | null>(null);
  const [draftPoints, setDraftPoints] = useState<ZonePoint[]>([]);
  const canvasWrapRef = useRef<HTMLDivElement | null>(null);

  const [showSchedModal, setShowSchedModal] = useState<boolean>(false);
  const [schedMode, setSchedMode] = useState<'always_record' | 'paused' | 'schedule'>('always_record');
  const [schedStart, setSchedStart] = useState<string>('21:00');
  const [schedEnd, setSchedEnd] = useState<string>('06:00');

  useEffect(() => {
    if (schedule) {
      setSchedMode(schedule.mode);
      setSchedStart(schedule.start_time || '21:00');
      setSchedEnd(schedule.end_time || '06:00');
    }
  }, [schedule]);

  useEffect(() => {
    if (!showHeatmap) return;
    const fetchHeat = async () => {
      try {
        const res = await fetch(`${apiBase}/api/detections/heatmap/${streamName}`);
        if (res.ok) {
          const data = await res.json();
          setHeatmapPoints(data.points || []);
        }
      } catch (err) {
        console.error('Failed to load heatmap:', err);
      }
    };
    fetchHeat();
    const timer = window.setInterval(fetchHeat, 8000);
    return () => window.clearInterval(timer);
  }, [showHeatmap, apiBase, streamName]);

  const hasBreach = detections.some((d) => d.breach);
  const isRecordingNow = schedule ? schedule.currently_recording : true;

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

  const saveSchedule = async (modeOverride?: 'always_record' | 'paused' | 'schedule') => {
    const targetMode = modeOverride || schedMode;
    try {
      const res = await fetch(`${apiBase}/api/cameras/${streamName}/schedule`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: targetMode,
          start_time: schedStart,
          end_time: schedEnd,
        }),
      });
      if (res.ok && onScheduleUpdated) {
        const updated = await res.json();
        onScheduleUpdated(updated);
      }
      setShowSchedModal(false);
    } catch (err) {
      console.error('Failed to save camera schedule:', err);
    }
  };

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

        <svg
          viewBox="0 0 1000 1000"
          preserveAspectRatio="none"
          className="absolute inset-0 w-full h-full pointer-events-none"
        >
          <defs>
            <radialGradient id={`heatGradRed-${streamName}`} cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#ef4444" stopOpacity="0.65" />
              <stop offset="45%" stopColor="#f59e0b" stopOpacity="0.35" />
              <stop offset="100%" stopColor="#10b981" stopOpacity="0" />
            </radialGradient>
            <radialGradient id={`heatGradSky-${streamName}`} cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.55" />
              <stop offset="50%" stopColor="#10b981" stopOpacity="0.25" />
              <stop offset="100%" stopColor="#000000" stopOpacity="0" />
            </radialGradient>
          </defs>

          {/* 24-Hour Tactical Thermal Heatmap Overlay */}
          {showHeatmap &&
            heatmapPoints.map((pt, idx) => (
              <circle
                key={idx}
                cx={pt.x * 1000}
                cy={pt.y * 1000}
                r={pt.category === 'person' || pt.breach ? 68 : 52}
                fill={
                  pt.category === 'person' || pt.breach
                    ? `url(#heatGradRed-${streamName})`
                    : `url(#heatGradSky-${streamName})`
                }
              />
            ))}

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

          {showBbox &&
            Object.entries(trails).map(([slot, trail]) => {
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

        {showBbox && (
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
                    {shortSlot} // {det.sub_type.toUpperCase()} {det.breach ? '⚠️ BREACH' : `· ${pctConf}%`}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Stationary Top HUD Control Bar */}
      <div
        className="absolute top-2 left-2 right-2 flex flex-wrap justify-between items-center gap-1 z-10"
        onClick={(e) => e.stopPropagation()}
        onDoubleClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-1.5 pointer-events-none">
          <span className="bg-zinc-950/85 border border-zinc-800 px-2 py-0.5 sm:px-2.5 sm:py-1 rounded text-[11px] sm:text-xs font-mono font-bold text-zinc-100">
            {title}
          </span>
          {hasBreach && (
            <span className="bg-red-600 text-white px-2 py-0.5 rounded text-[10px] font-mono font-bold animate-bounce">
              🚨 BREACH
            </span>
          )}
          {showHeatmap && (
            <span className="bg-orange-600/90 text-white px-1.5 py-0.5 rounded text-[10px] font-mono font-bold">
              🔥 24H HEAT ({heatmapPoints.length})
            </span>
          )}
          {zoom > 1 && (
            <span className="bg-sky-950/90 border border-sky-500/40 px-1.5 py-0.5 rounded text-[10px] font-mono font-bold text-sky-300">
              {zoom.toFixed(1)}x
            </span>
          )}
        </div>

        <div className="flex items-center gap-1 flex-wrap justify-end">
          {drawMode ? (
            <>
              <span className="bg-amber-950/90 border border-amber-500/50 px-2 py-1 rounded text-[10px] font-mono text-amber-300">
                {Math.max(0, 3 - draftPoints.length)}+ PTS ({draftPoints.length})
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
                onClick={() => setShowSchedModal((prev) => !prev)}
                title="Configure Recording / Pause Schedule"
                className={`px-1.5 py-1 rounded text-[10px] font-mono font-bold border flex items-center gap-1 cursor-pointer transition-colors ${
                  !isRecordingNow
                    ? 'bg-amber-950/90 text-amber-300 border-amber-500/60'
                    : schedule?.mode === 'schedule'
                    ? 'bg-sky-950/90 text-sky-300 border-sky-500/50'
                    : 'bg-zinc-950/85 hover:bg-zinc-800 text-emerald-400 border-zinc-800'
                }`}
              >
                {schedule?.mode === 'schedule' ? (
                  <Clock className="w-3 h-3" />
                ) : isRecordingNow ? (
                  <Database className="w-3 h-3" />
                ) : (
                  <PauseCircle className="w-3 h-3" />
                )}
                <span>
                  {schedule?.mode === 'schedule'
                    ? isRecordingNow
                      ? 'SCHED·REC'
                      : 'SCHED·PAUSED'
                    : isRecordingNow
                    ? 'REC'
                    : 'PAUSED'}
                </span>
              </button>

              <button
                onClick={() => setShowHeatmap((prev) => !prev)}
                title="Toggle 24-Hour Tactical Thermal Heatmap Overlay"
                className={`px-1.5 py-1 rounded text-[10px] font-mono font-bold border flex items-center gap-1 cursor-pointer ${
                  showHeatmap
                    ? 'bg-orange-600 text-white border-orange-400'
                    : 'bg-zinc-950/85 hover:bg-zinc-800 text-orange-300 border-zinc-800'
                }`}
              >
                <Flame className="w-3 h-3" />
                <span className="hidden sm:inline">HEAT</span>
              </button>

              <button
                onClick={() => setShowBbox((prev) => !prev)}
                title="Toggle Target Bounding Boxes"
                className={`px-1.5 py-1 rounded text-[10px] font-mono border flex items-center gap-1 cursor-pointer ${
                  showBbox
                    ? 'bg-zinc-950/85 text-emerald-300 border-zinc-800'
                    : 'bg-zinc-900 text-zinc-500 border-zinc-800'
                }`}
              >
                {showBbox ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
                <span className="hidden sm:inline">BBOX</span>
              </button>

              <button
                onClick={() => startDrawing('exclusion')}
                title="Draw Red Exclusion Mask"
                className="bg-zinc-950/85 hover:bg-red-950/80 border border-zinc-800 hover:border-red-500/50 px-1.5 py-1 rounded text-[10px] font-mono text-red-300 flex items-center gap-1 cursor-pointer"
              >
                <ShieldOff className="w-3 h-3" />
                <span className="hidden sm:inline">+EXCL</span>
              </button>
              <button
                onClick={() => startDrawing('tripwire')}
                title="Draw Amber Perimeter Tripwire"
                className="bg-zinc-950/85 hover:bg-amber-950/80 border border-zinc-800 hover:border-amber-500/50 px-1.5 py-1 rounded text-[10px] font-mono text-amber-300 flex items-center gap-1 cursor-pointer"
              >
                <ShieldAlert className="w-3 h-3" />
                <span className="hidden sm:inline">+TRIP</span>
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
                  className="bg-zinc-950/85 hover:bg-zinc-800 border border-zinc-700 px-1.5 py-1 rounded text-[10px] font-mono text-zinc-300 flex items-center gap-1 cursor-pointer"
                >
                  <RotateCcw className="w-3 h-3" />
                  1x
                </button>
              )}
              <button
                onClick={() => setAutoLock((prev) => !prev)}
                title="Auto-Zoom & Track Primary Target"
                className={`px-1.5 py-1 rounded text-[10px] font-mono font-bold border flex items-center gap-1 cursor-pointer transition-colors ${
                  autoLock
                    ? 'bg-amber-500/25 text-amber-300 border-amber-500/60'
                    : 'bg-zinc-950/85 hover:bg-zinc-800 text-zinc-400 border-zinc-800'
                }`}
              >
                <Crosshair className="w-3 h-3" />
                <span className="hidden sm:inline">{autoLock ? 'LOCKED' : 'LOCK'}</span>
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
            className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold pointer-events-none ${
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

      {showSchedModal && (
        <div
          onClick={(e) => e.stopPropagation()}
          className="absolute top-11 right-2 z-30 bg-zinc-950/95 border border-zinc-700 rounded-lg p-3 w-64 shadow-2xl font-mono text-xs space-y-2.5"
        >
          <div className="flex items-center justify-between border-b border-zinc-800 pb-1.5">
            <span className="font-bold text-emerald-400">{title} // SCHEDULE</span>
            <button onClick={() => setShowSchedModal(false)} className="text-zinc-400 hover:text-white cursor-pointer">
              <X className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="grid grid-cols-3 gap-1">
            <button
              onClick={() => saveSchedule('always_record')}
              className={`py-1.5 rounded text-[10px] font-bold border cursor-pointer ${
                schedMode === 'always_record'
                  ? 'bg-emerald-600 text-white border-emerald-400'
                  : 'bg-zinc-900 text-zinc-300 border-zinc-800'
              }`}
            >
              24/7 REC
            </button>
            <button
              onClick={() => saveSchedule('paused')}
              className={`py-1.5 rounded text-[10px] font-bold border cursor-pointer ${
                schedMode === 'paused'
                  ? 'bg-amber-600 text-white border-amber-400'
                  : 'bg-zinc-900 text-zinc-300 border-zinc-800'
              }`}
            >
              PAUSE NOW
            </button>
            <button
              onClick={() => setSchedMode('schedule')}
              className={`py-1.5 rounded text-[10px] font-bold border cursor-pointer ${
                schedMode === 'schedule'
                  ? 'bg-sky-600 text-white border-sky-400'
                  : 'bg-zinc-900 text-zinc-300 border-zinc-800'
              }`}
            >
              SCHEDULE
            </button>
          </div>

          {schedMode === 'schedule' && (
            <div className="space-y-2 pt-1 border-t border-zinc-800">
              <div className="text-[10px] text-zinc-400">Active Recording Window (PT):</div>
              <div className="flex items-center justify-between gap-2">
                <label className="text-[10px] text-zinc-400">START:</label>
                <input
                  type="time"
                  value={schedStart}
                  onChange={(e) => setSchedStart(e.target.value)}
                  className="bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-100"
                />
              </div>
              <div className="flex items-center justify-between gap-2">
                <label className="text-[10px] text-zinc-400">STOP:</label>
                <input
                  type="time"
                  value={schedEnd}
                  onChange={(e) => setSchedEnd(e.target.value)}
                  className="bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-100"
                />
              </div>
              <button
                onClick={() => saveSchedule('schedule')}
                className="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-1.5 rounded text-[11px] cursor-pointer"
              >
                SAVE WINDOW
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
