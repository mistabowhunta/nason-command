import React, { useEffect, useState } from 'react';
import type { PersistedDetectionEvent } from '../types/telemetry';
import { Camera, Clock, Cpu, Eye, Sparkles, Trash2, X } from 'lucide-react';

interface EvidenceRailProps {
  hostIp: string;
  refreshTrigger?: number;
}

const BADGE_COLORS: Record<string, string> = {
  person: 'bg-red-950 text-red-300 border-red-500/50',
  vehicle: 'bg-sky-950 text-sky-300 border-sky-400/50',
  animal: 'bg-amber-950 text-amber-300 border-amber-400/50',
};

export function EvidenceRail({ hostIp, refreshTrigger = 0 }: EvidenceRailProps) {
  const [events, setEvents] = useState<PersistedDetectionEvent[]>([]);
  const [selectedEvent, setSelectedEvent] = useState<PersistedDetectionEvent | null>(null);
  const [inspectingId, setInspectingId] = useState<number | null>(null);
  const apiBase = `http://${hostIp}:8080`;

  const fetchEvents = async () => {
    try {
      const res = await fetch(`${apiBase}/api/detections?limit=25`);
      if (res.ok) {
        const data: PersistedDetectionEvent[] = await res.json();
        setEvents(data);
      }
    } catch (err) {
      console.error('Failed to fetch evidence feed:', err);
    }
  };

  useEffect(() => {
    fetchEvents();
    const timer = window.setInterval(fetchEvents, 2500);
    return () => window.clearInterval(timer);
  }, [apiBase, refreshTrigger]);

  const handlePurgeAll = async () => {
    try {
      const res = await fetch(`${apiBase}/api/detections`, { method: 'DELETE' });
      if (res.ok) {
        setEvents([]);
        setSelectedEvent(null);
      }
    } catch (err) {
      console.error('Failed to purge detections:', err);
    }
  };

  const handleDeepInspect = async (e: React.MouseEvent, evId: number) => {
    e.stopPropagation();
    setInspectingId(evId);
    try {
      const res = await fetch(`${apiBase}/api/detections/${evId}/inspect`, {
        method: 'POST',
      });
      if (res.ok) {
        const data = await res.json();
        if (data.purged) {
          setEvents((prev) => prev.filter((item) => item.id !== evId));
          setSelectedEvent(null);
        } else {
          setEvents((prev) =>
            prev.map((item) => (item.id === evId ? { ...item, ai_notes: data.ai_notes } : item))
          );
          if (selectedEvent && selectedEvent.id === evId) {
            setSelectedEvent({ ...selectedEvent, ai_notes: data.ai_notes });
          }
        }
      }
    } catch (err) {
      console.error('Deep inspect failed:', err);
    } finally {
      setInspectingId(null);
    }
  };

  return (
    <>
      <aside className="w-80 border-l border-zinc-800 bg-zinc-900/60 flex flex-col h-[calc(100vh-105px)] sticky top-[53px]">
        <div className="px-3 py-3 border-b border-zinc-800 flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <Eye className="w-4 h-4 text-emerald-400" />
            <span className="text-xs font-mono font-bold uppercase tracking-wider text-zinc-200">
              KILL-FEED ({events.length})
            </span>
          </div>
          {events.length > 0 && (
            <button
              onClick={handlePurgeAll}
              title="Wipe all saved snapshots and reset database history"
              className="text-[10px] font-mono bg-red-950/80 hover:bg-red-900 text-red-300 border border-red-500/40 px-2 py-0.5 rounded flex items-center gap-1 cursor-pointer"
            >
              <Trash2 className="w-3 h-3" />
              PURGE
            </button>
          )}
        </div>

        <div className="flex-1 overflow-y-auto p-3 space-y-2.5">
          {events.length === 0 ? (
            <div className="text-center py-12 text-xs font-mono text-zinc-500">
              AWAITING TARGET DETECTIONS...
            </div>
          ) : (
            events.map((ev) => {
              const badge = BADGE_COLORS[ev.category] || BADGE_COLORS.vehicle;
              const isBreach = ev.full_label.includes('BREACH');
              const timeStr = new Date(ev.timestamp).toLocaleTimeString([], {
                hour12: false,
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
              });

              return (
                <div
                  key={ev.id}
                  onClick={() => setSelectedEvent(ev)}
                  className={`group bg-zinc-950/90 hover:bg-zinc-900 border rounded p-2 flex flex-col gap-2 cursor-pointer transition-all ${
                    isBreach ? 'border-red-500/70' : 'border-zinc-800 hover:border-zinc-600'
                  }`}
                >
                  <div className="flex gap-3 items-center">
                    <div className="w-16 h-14 bg-zinc-900 rounded overflow-hidden shrink-0 border border-zinc-800 flex items-center justify-center">
                      {ev.crop_url ? (
                        <img
                          src={`${apiBase}${ev.crop_url}`}
                          alt={ev.full_label}
                          className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                        />
                      ) : (
                        <Camera className="w-5 h-5 text-zinc-700" />
                      )}
                    </div>

                    <div className="flex-1 min-w-0 font-mono">
                      <div className="flex items-center justify-between gap-1">
                        <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded border uppercase ${badge}`}>
                          {ev.sub_type || ev.category} · {Math.round(ev.confidence * 100)}%
                        </span>
                        <span className="text-[10px] text-zinc-400 flex items-center gap-1">
                          <Clock className="w-2.5 h-2.5" />
                          {timeStr}
                        </span>
                      </div>
                      <div className="text-xs text-zinc-200 truncate mt-1 font-semibold">
                        {ev.full_label}
                      </div>
                      <div className="flex items-center justify-between mt-1">
                        <span className="text-[10px] text-zinc-500 uppercase">
                          CAM: {ev.camera_id}
                        </span>
                        <button
                          onClick={(e) => handleDeepInspect(e, ev.id)}
                          disabled={inspectingId === ev.id}
                          className="px-1.5 py-0.5 rounded text-[10px] font-mono font-bold bg-emerald-950/90 hover:bg-emerald-900 text-emerald-300 border border-emerald-500/40 flex items-center gap-1 cursor-pointer"
                        >
                          <Sparkles className="w-2.5 h-2.5" />
                          {inspectingId === ev.id ? 'SCANNING...' : 'INSPECT'}
                        </button>
                      </div>
                    </div>
                  </div>

                  {ev.ai_notes && (
                    <div className="bg-emerald-950/30 border border-emerald-500/30 rounded px-2 py-1.5 text-[10px] font-mono text-emerald-200 leading-relaxed">
                      {ev.ai_notes}
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      </aside>

      {selectedEvent && (
        <div
          onClick={() => setSelectedEvent(null)}
          className="fixed inset-0 z-50 bg-black/85 backdrop-blur-xs flex items-center justify-center p-6"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="bg-zinc-950 border border-zinc-700 rounded-lg max-w-4xl w-full overflow-hidden shadow-2xl"
          >
            <div className="px-4 py-3 border-b border-zinc-800 flex items-center justify-between font-mono">
              <div className="flex items-center gap-3">
                <span className="text-xs font-bold text-emerald-400 uppercase">
                  EVIDENCE SNAPSHOT #{selectedEvent.id}
                </span>
                <span className="text-xs text-zinc-300">
                  {selectedEvent.full_label} ({selectedEvent.camera_id})
                </span>
                <span className="text-xs text-zinc-500">
                  {new Date(selectedEvent.timestamp).toLocaleString()}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={(e) => handleDeepInspect(e, selectedEvent.id)}
                  disabled={inspectingId === selectedEvent.id}
                  className="px-2.5 py-1 rounded text-xs font-mono font-bold bg-emerald-600 hover:bg-emerald-500 text-white flex items-center gap-1.5 cursor-pointer"
                >
                  <Cpu className="w-3.5 h-3.5" />
                  {inspectingId === selectedEvent.id ? 'RUNNING DEEP INSPECT...' : 'DEEP INSPECT TARGET'}
                </button>
                <button
                  onClick={() => setSelectedEvent(null)}
                  className="text-zinc-400 hover:text-white p-1 cursor-pointer"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            {selectedEvent.ai_notes && (
              <div className="px-4 py-2 bg-emerald-950/50 border-b border-emerald-500/30 text-xs font-mono text-emerald-200">
                {selectedEvent.ai_notes}
              </div>
            )}

            <div className="bg-black aspect-video flex items-center justify-center">
              {selectedEvent.full_url ? (
                <img
                  src={`${apiBase}${selectedEvent.full_url}`}
                  alt="Full frame capture"
                  className="w-full h-full object-contain"
                />
              ) : (
                <span className="text-xs font-mono text-zinc-500">
                  No snapshot image stored for this legacy event
                </span>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
