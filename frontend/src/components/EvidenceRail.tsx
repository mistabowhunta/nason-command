import { useEffect, useState } from 'react';
import type { PersistedDetectionEvent } from '../types/telemetry';
import { Camera, Clock, Eye, X } from 'lucide-react';

interface EvidenceRailProps {
  hostIp: string;
}

const BADGE_COLORS: Record<string, string> = {
  person: 'bg-red-950 text-red-300 border-red-500/50',
  vehicle: 'bg-sky-950 text-sky-300 border-sky-400/50',
  animal: 'bg-amber-950 text-amber-300 border-amber-400/50',
};

export function EvidenceRail({ hostIp }: EvidenceRailProps) {
  const [events, setEvents] = useState<PersistedDetectionEvent[]>([]);
  const [selectedEvent, setSelectedEvent] = useState<PersistedDetectionEvent | null>(null);
  const apiBase = `http://${hostIp}:8080`;

  useEffect(() => {
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

    fetchEvents();
    const timer = window.setInterval(fetchEvents, 2500);
    return () => window.clearInterval(timer);
  }, [apiBase]);

  return (
    <>
      <aside className="w-80 border-l border-zinc-800 bg-zinc-900/60 flex flex-col h-[calc(100vh-53px)] sticky top-[53px]">
        <div className="px-4 py-3 border-b border-zinc-800 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Eye className="w-4 h-4 text-emerald-400" />
            <span className="text-xs font-mono font-bold uppercase tracking-wider text-zinc-200">
              KILL-FEED // EVIDENCE
            </span>
          </div>
          <span className="text-[10px] font-mono bg-zinc-800 text-zinc-300 px-2 py-0.5 rounded">
            {events.length} LOGGED
          </span>
        </div>

        <div className="flex-1 overflow-y-auto p-3 space-y-2.5">
          {events.length === 0 ? (
            <div className="text-center py-12 text-xs font-mono text-zinc-500">
              AWAITING TARGET DETECTIONS...
            </div>
          ) : (
            events.map((ev) => {
              const badge = BADGE_COLORS[ev.category] || BADGE_COLORS.vehicle;
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
                  className="group bg-zinc-950/90 hover:bg-zinc-900 border border-zinc-800 hover:border-zinc-600 rounded p-2 flex gap-3 items-center cursor-pointer transition-all"
                >
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
                        {ev.category} · {Math.round(ev.confidence * 100)}%
                      </span>
                      <span className="text-[10px] text-zinc-400 flex items-center gap-1">
                        <Clock className="w-2.5 h-2.5" />
                        {timeStr}
                      </span>
                    </div>
                    <div className="text-xs text-zinc-200 truncate mt-1 font-semibold">
                      {ev.full_label}
                    </div>
                    <div className="text-[10px] text-zinc-500 uppercase">
                      CAM: {ev.camera_id}
                    </div>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </aside>

      {/* Full-Frame High-Res Lightbox Modal */}
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
              <button
                onClick={() => setSelectedEvent(null)}
                className="text-zinc-400 hover:text-white p-1 cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
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
