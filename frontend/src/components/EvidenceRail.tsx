import React, { useEffect, useState } from 'react';
import type { PersistedDetectionEvent } from '../types/telemetry';
import {
  Camera,
  Check,
  Clock,
  Cpu,
  Crosshair,
  Download,
  ExternalLink,
  Eye,
  Film,
  Image as ImageIcon,
  Lock,
  Trash2,
  UserCheck,
  X,
} from 'lucide-react';

interface EvidenceRailProps {
  hostIp: string;
  refreshTrigger?: number;
  fullWidthMobile?: boolean;
}

const CAM_NAMES: Record<string, string> = {
  annke_1: 'DRIVEWAY',
  tapo_1: 'LOFT',
  tapo_2: 'LIVING_ROOM',
};

const BADGE_COLORS: Record<string, string> = {
  person: 'bg-red-950 text-red-300 border-red-500/50',
  vehicle: 'bg-sky-950 text-sky-300 border-sky-400/50',
  delivery: 'bg-amber-950 text-amber-300 border-amber-400/60',
  animal: 'bg-amber-950 text-amber-300 border-amber-400/50',
};

export function EvidenceRail({ hostIp, refreshTrigger = 0, fullWidthMobile = false }: EvidenceRailProps) {
  const [events, setEvents] = useState<PersistedDetectionEvent[]>([]);
  const [catFilter, setCatFilter] = useState<string>('ALL');
  const [selectedEvent, setSelectedEvent] = useState<PersistedDetectionEvent | null>(null);
  const [viewMode, setViewMode] = useState<'clip' | 'still'>('clip');
  const [showSnapshotBbox, setShowSnapshotBbox] = useState<boolean>(true);
  const [inspectingId, setInspectingId] = useState<number | null>(null);
  const [enrollingId, setEnrollingId] = useState<number | null>(null);
  const [faceNameInput, setFaceNameInput] = useState<string>('ISAAC');
  const [faceRoleInput, setFaceRoleInput] = useState<'whitelist' | 'watchlist'>('whitelist');
  const [enrollStatus, setEnrollStatus] = useState<string | null>(null);
  const [teachingId, setTeachingId] = useState<number | null>(null);
  const [teachCategory, setTeachCategory] = useState<string>('animal');
  const [teachSubType, setTeachSubType] = useState<string>('dog');
  const [teachNote, setTeachNote] = useState<string>('');
  const [teachStatus, setTeachStatus] = useState<string | null>(null);
  const apiBase = `http://${hostIp}:8080`;

  const fetchEvents = async () => {
    try {
      const res = await fetch(`${apiBase}/api/detections?limit=35`);
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

  const handleTeachSubmit = async (e: React.FormEvent, evId: number) => {
    e.preventDefault();
    e.stopPropagation();
    setTeachStatus('Saving lesson to AI memory...');
    try {
      const res = await fetch(`${apiBase}/api/detections/${evId}/teach`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          category: teachCategory,
          sub_type: teachSubType.trim() || teachCategory,
          note: teachNote.trim(),
        }),
      });
      const data = await res.json();
      if (res.ok && !data.error) {
        setTeachStatus('Learned! Saved to AI memory.');
        setEvents((prev) =>
          prev.map((item) =>
            item.id === evId
              ? {
                  ...item,
                  category: data.category,
                  sub_type: data.sub_type,
                  full_label: data.full_label,
                  attributes: data.attributes,
                  ai_notes: data.ai_notes,
                }
              : item
          )
        );
        window.setTimeout(() => {
          setTeachingId(null);
          setTeachStatus(null);
        }, 900);
      } else {
        setTeachStatus(data.error || 'Failed to save lesson');
      }
    } catch {
      setTeachStatus('Network error saving lesson');
    }
  };

  const handleEnrollFace = async (e: React.FormEvent, evId: number) => {
    e.preventDefault();
    e.stopPropagation();
    if (!faceNameInput.trim()) return;
    setEnrollStatus('Extracting 128-D SFace vector...');
    try {
      const res = await fetch(`${apiBase}/api/detections/${evId}/enroll_face`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: faceNameInput.trim(), role: faceRoleInput }),
      });
      const data = await res.json();
      if (res.ok) {
        setEnrollStatus(`Enrolled ${data.name} (${faceRoleInput.toUpperCase()})!`);
        setEvents((prev) =>
          prev.map((item) =>
            item.id === evId
              ? {
                  ...item,
                  sub_type: data.sub_type,
                  full_label: data.full_label,
                  ai_notes: data.ai_notes,
                }
              : item
          )
        );
        window.setTimeout(() => {
          setEnrollingId(null);
          setEnrollStatus(null);
        }, 1400);
      } else {
        setEnrollStatus(data.detail || 'Could not isolate face in this crop.');
      }
    } catch {
      setEnrollStatus('Failed to reach Face Engine.');
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
            prev.map((item) =>
              item.id === evId
                ? {
                    ...item,
                    sub_type: data.sub_type || item.sub_type,
                    full_label: data.full_label || item.full_label,
                    ai_notes: data.ai_notes,
                  }
                : item
            )
          );
          if (selectedEvent && selectedEvent.id === evId) {
            setSelectedEvent({
              ...selectedEvent,
              sub_type: data.sub_type || selectedEvent.sub_type,
              full_label: data.full_label || selectedEvent.full_label,
              ai_notes: data.ai_notes,
            });
          }
        }
      }
    } catch (err) {
      console.error('Deep inspect failed:', err);
    } finally {
      setInspectingId(null);
    }
  };

  const launchOsintFromEvent = (cropUrl: string | null, site: 'pimeyes' | 'yandex') => {
    if (cropUrl) {
      const a = document.createElement('a');
      a.href = `${apiBase}${cropUrl}`;
      a.download = `target_crop.jpg`;
      a.target = '_blank';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
    window.open(
      site === 'pimeyes' ? 'https://pimeyes.com/en' : 'https://yandex.com/images/',
      '_blank',
      'noopener,noreferrer'
    );
  };

  const isSelectedWhitelisted =
    selectedEvent !== null &&
    (selectedEvent.sub_type || '').startsWith('ID:') &&
    !(selectedEvent.sub_type || '').includes('VISITOR_') &&
    !(selectedEvent.sub_type || '').includes('WATCHLIST');

  return (
    <>
      <aside
        className={`${
          fullWidthMobile ? 'w-full' : 'w-80 border-l border-zinc-800 h-[calc(100vh-105px)] sticky top-[53px]'
        } bg-zinc-900/60 flex flex-col`}
      >
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
              title="Wipe snapshots & clips (Face IDs are preserved)"
              className="text-[10px] font-mono bg-red-950/80 hover:bg-red-900 text-red-300 border border-red-500/40 px-2 py-0.5 rounded flex items-center gap-1 cursor-pointer"
            >
              <Trash2 className="w-3 h-3" />
              PURGE
            </button>
          )}
        </div>
      <div className="px-2.5 py-1.5 border-b border-zinc-800/80 bg-zinc-950/60 flex items-center gap-1 overflow-x-auto">
        {(['ALL', 'person', 'delivery', 'vehicle', 'animal'] as const).map((cat) => (
          <button
            key={cat}
            onClick={() => setCatFilter(cat)}
            className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase cursor-pointer border ${
              catFilter === cat
                ? 'bg-emerald-950 text-emerald-300 border-emerald-500/60'
                : 'bg-zinc-900/80 text-zinc-400 border-zinc-800 hover:text-zinc-200'
            }`}
          >
            {cat}
          </button>
        ))}
      </div>

        <div className="flex-1 overflow-y-auto p-3 space-y-2.5">
          {events.length === 0 ? (
            <div className="text-center py-12 text-xs font-mono text-zinc-500">
              AWAITING TARGET DETECTIONS...
            </div>
          ) : (
            events.filter((e) => catFilter === 'ALL' || e.category === catFilter).map((ev) => {
              const isRecognizedFace = (ev.sub_type || '').startsWith('ID:');
              const isWatchlist = (ev.sub_type || '').includes('WATCHLIST');
              const badge = isWatchlist
                ? 'bg-red-600 text-white border-red-300'
                : isRecognizedFace
                ? 'bg-emerald-950 text-emerald-300 border-emerald-400/60'
                : BADGE_COLORS[ev.category] || BADGE_COLORS.vehicle;
              const isBreach = ev.full_label.includes('BREACH') || isWatchlist;
              const friendlyCam = CAM_NAMES[ev.camera_id] || ev.camera_id.toUpperCase();
              const timeStr = new Date(ev.timestamp).toLocaleTimeString([], {
                hour12: false,
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
              });

              return (
                <div
                  key={ev.id}
                  onClick={() => {
                    setSelectedEvent(ev);
                    setViewMode(ev.clip_url ? 'clip' : 'still');
                  }}
                  className={`group rounded p-2 flex flex-col gap-2 cursor-pointer transition-all border ${
                    isBreach
                      ? 'bg-red-950/35 hover:bg-red-950/50 border-red-500 shadow-[0_0_15px_rgba(239,68,68,0.25)]'
                      : isRecognizedFace
                      ? 'bg-emerald-950/25 hover:bg-emerald-950/40 border-emerald-500/60'
                      : 'bg-zinc-950/90 hover:bg-zinc-900 border-zinc-800 hover:border-zinc-600'
                  }`}
                >
                  {isBreach && (
                    <div className="bg-red-600 text-white text-[10px] font-mono font-bold px-2 py-0.5 rounded flex items-center justify-between">
                      <span>{isWatchlist ? '🚨 WATCHLIST BIOMETRIC MATCH' : '🚨 TRIPWIRE PERIMETER BREACH'}</span>
                      <span>{friendlyCam}</span>
                    </div>
                  )}

                  <div className="flex gap-3 items-center">
                    <div className="relative w-16 h-14 bg-zinc-900 rounded overflow-hidden shrink-0 border border-zinc-800 flex items-center justify-center">
                      {ev.crop_url ? (
                        <img
                          src={`${apiBase}${ev.crop_url}`}
                          alt={ev.full_label}
                          className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                        />
                      ) : (
                        <Camera className="w-5 h-5 text-zinc-700" />
                      )}
                      {ev.clip_url && (
                        <span className="absolute bottom-0.5 right-0.5 bg-emerald-600/95 text-white text-[8px] font-mono font-bold px-1 rounded-xs flex items-center gap-0.5">
                          <Film className="w-2.5 h-2.5" />
                          5s
                        </span>
                      )}
                    </div>

                    <div className="flex-1 min-w-0 font-mono">
                      <div className="flex items-center justify-between gap-1">
                        <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded border uppercase truncate ${badge}`}>
                          {ev.sub_type || ev.category} · {Math.round(ev.confidence * 100)}%
                        </span>
                        <span className="text-[10px] text-zinc-400 flex items-center gap-1 shrink-0">
                          <Clock className="w-2.5 h-2.5" />
                          {timeStr}
                        </span>
                      </div>
                      <div className="text-xs text-zinc-200 truncate mt-1 font-semibold">
                        {ev.full_label}
                      </div>
                      <div className="flex items-center justify-between mt-1 gap-1">
                        <span className="text-[10px] text-zinc-400 uppercase font-bold">
                          {friendlyCam}
                        </span>
                        <div className="flex items-center gap-1">
                          {ev.category === 'person' && (
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                setEnrollingId(enrollingId === ev.id ? null : ev.id);
                                setEnrollStatus(null);
                              }}
                              title="Enroll this person's face into YuNet + SFace"
                              className="px-1.5 py-0.5 rounded text-[10px] font-mono font-bold bg-sky-950/90 hover:bg-sky-900 text-sky-300 border border-sky-500/40 flex items-center gap-1 cursor-pointer"
                            >
                              <UserCheck className="w-2.5 h-2.5" />
                              ID FACE
                            </button>
                          )}
                          <button
                              onClick={(e) => {
                                e.stopPropagation();
                                if (teachingId === ev.id) {
                                  setTeachingId(null);
                                } else {
                                  setTeachingId(ev.id);
                                  setTeachCategory(ev.category || 'animal');
                                  setTeachSubType(ev.sub_type === 'cat' ? 'dog' : (ev.sub_type || 'dog'));
                                  setTeachNote('');
                                  setTeachStatus(null);
                                }
                              }}
                              title="Correct a misidentification and teach the AI model"
                              className="px-1.5 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-950/90 hover:bg-amber-900 text-amber-300 border border-amber-500/40 flex items-center gap-1 cursor-pointer"
                            >
                              🎓 TEACH
                            </button>
                        </div>
                      </div>
                    </div>
                  </div>

                  {teachingId === ev.id && (
                <form
                  onSubmit={(e) => handleTeachSubmit(e, ev.id)}
                  onClick={(e) => e.stopPropagation()}
                  className="mt-2 p-2 rounded bg-zinc-950 border border-amber-500/40 space-y-1.5"
                >
                  <div className="text-[10px] font-mono font-bold text-amber-300">
                    🎓 TEACH / CORRECT AI MODEL
                  </div>
                  <div className="flex gap-1.5">
                    <select
                      value={teachCategory}
                      onChange={(e) => setTeachCategory(e.target.value)}
                      className="bg-zinc-900 border border-zinc-700 rounded px-1.5 py-1 text-[10px] font-mono text-zinc-200"
                    >
                      <option value="animal">animal</option>
                      <option value="person">person</option>
                      <option value="delivery">delivery</option>
                      <option value="vehicle">vehicle</option>
                    </select>
                    <input
                      type="text"
                      value={teachSubType}
                      onChange={(e) => setTeachSubType(e.target.value)}
                      placeholder="True object (e.g. dog)"
                      className="flex-1 bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-[10px] font-mono text-zinc-100"
                    />
                  </div>
                  <input
                    type="text"
                    value={teachNote}
                    onChange={(e) => setTeachNote(e.target.value)}
                    placeholder="Context (e.g. Black & white dog walked by owner with leash)"
                    className="w-full bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-[10px] font-mono text-zinc-100"
                  />
                  <div className="flex items-center justify-between gap-2 pt-0.5">
                    <span className="text-[10px] font-mono text-amber-300/90 truncate">
                      {teachStatus || 'Saves rule to persistent AI memory'}
                    </span>
                    <button
                      type="submit"
                      className="px-2.5 py-0.5 rounded bg-amber-500 hover:bg-amber-400 text-black font-mono font-bold text-[10px] cursor-pointer shrink-0"
                    >
                      SAVE & TEACH
                    </button>
                  </div>
                </form>
              )}
              {enrollingId === ev.id && (
                    <form
                      onClick={(e) => e.stopPropagation()}
                      onSubmit={(e) => handleEnrollFace(e, ev.id)}
                      className="bg-zinc-900 border border-sky-500/50 rounded p-2 flex flex-col gap-1.5 font-mono"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] text-sky-300 font-bold">ENROLL 128-D SFACE:</span>
                        <div className="flex gap-1">
                          <button
                            type="button"
                            onClick={() => setFaceRoleInput('whitelist')}
                            className={`px-1.5 py-0.5 rounded text-[9px] font-bold cursor-pointer ${
                              faceRoleInput === 'whitelist'
                                ? 'bg-emerald-600 text-white'
                                : 'bg-zinc-950 text-zinc-400'
                            }`}
                          >
                            WHITELIST
                          </button>
                          <button
                            type="button"
                            onClick={() => setFaceRoleInput('watchlist')}
                            className={`px-1.5 py-0.5 rounded text-[9px] font-bold cursor-pointer ${
                              faceRoleInput === 'watchlist'
                                ? 'bg-red-600 text-white'
                                : 'bg-zinc-950 text-zinc-400'
                            }`}
                          >
                            WATCHLIST
                          </button>
                        </div>
                      </div>

                      <div className="flex items-center gap-1">
                        <input
                          type="text"
                          value={faceNameInput}
                          onChange={(e) => setFaceNameInput(e.target.value)}
                          placeholder="ISAAC, JENNIFER, PROWLER..."
                          className="flex-1 bg-zinc-950 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-100 uppercase"
                        />
                        <button
                          type="submit"
                          className="bg-sky-600 hover:bg-sky-500 text-white px-2 py-1 rounded text-[10px] font-bold flex items-center gap-1 cursor-pointer"
                        >
                          <Check className="w-3 h-3" />
                          SAVE
                        </button>
                      </div>
                      <div className="flex gap-1">
                        {['ISAAC', 'JENNIFER', 'USPS_DRIVER'].map((preset) => (
                          <button
                            key={preset}
                            type="button"
                            onClick={() => setFaceNameInput(preset)}
                            className="text-[9px] bg-zinc-950 hover:bg-zinc-800 border border-zinc-700 px-1.5 py-0.5 rounded text-zinc-300 cursor-pointer"
                          >
                            {preset}
                          </button>
                        ))}
                      </div>
                      {enrollStatus && (
                        <div className="text-[10px] text-amber-300">{enrollStatus}</div>
                      )}
                    </form>
                  )}

                  {ev.ai_notes && (
                    <div
                      className={`rounded px-2 py-1.5 text-[10px] font-mono leading-relaxed border ${
                        isBreach
                          ? 'bg-red-950/60 border-red-500/40 text-red-200'
                          : 'bg-emerald-950/30 border-emerald-500/30 text-emerald-200'
                      }`}
                    >
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
          className="fixed inset-0 z-50 bg-black/85 backdrop-blur-xs flex items-center justify-center p-2 sm:p-6"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="bg-zinc-950 border border-zinc-700 rounded-lg max-w-4xl w-full overflow-hidden shadow-2xl"
          >
            <div className="px-3 sm:px-4 py-3 border-b border-zinc-800 flex flex-wrap items-center justify-between gap-2 font-mono">
              <div className="flex items-center gap-2 sm:gap-3 flex-wrap">
                <span className="text-xs font-bold text-emerald-400 uppercase">
                  EVENT #{selectedEvent.id}
                </span>
                <span className="text-xs text-zinc-200 font-bold">
                  {CAM_NAMES[selectedEvent.camera_id] || selectedEvent.camera_id} // {selectedEvent.full_label}
                </span>
              </div>
              <div className="flex flex-wrap items-center gap-1.5">
                {selectedEvent.clip_url && (
                  <button
                    onClick={() => setViewMode(viewMode === 'clip' ? 'still' : 'clip')}
                    className={`px-2 py-1 rounded text-[11px] font-mono font-bold border flex items-center gap-1 cursor-pointer ${
                      viewMode === 'clip'
                        ? 'bg-emerald-600 text-white border-emerald-400'
                        : 'bg-zinc-900 text-emerald-300 border-zinc-700'
                    }`}
                  >
                    {viewMode === 'clip' ? <ImageIcon className="w-3 h-3" /> : <Film className="w-3 h-3" />}
                    {viewMode === 'clip' ? 'VIEW STILL + BBOX' : '🎬 5s MP4 REPLAY'}
                  </button>
                )}
                {selectedEvent.crop_url && (
                  <a
                    href={`${apiBase}${selectedEvent.crop_url}`}
                    download={`crop_${selectedEvent.id}.jpg`}
                    className="px-2 py-1 rounded text-[11px] font-mono bg-zinc-900 hover:bg-zinc-800 text-zinc-200 border border-zinc-700 flex items-center gap-1"
                  >
                    <Download className="w-3 h-3 text-emerald-400" />
                    CROP
                  </a>
                )}
                <button
                  disabled={isSelectedWhitelisted}
                  onClick={() => !isSelectedWhitelisted && launchOsintFromEvent(selectedEvent.crop_url, 'pimeyes')}
                  title={
                    isSelectedWhitelisted
                      ? 'OpSec Lock: OSINT disabled for Whitelisted family members'
                      : 'Download crop & launch PimEyes'
                  }
                  className="px-2 py-1 rounded text-[11px] font-mono bg-zinc-900 hover:bg-zinc-800 disabled:bg-zinc-950 disabled:opacity-35 disabled:text-zinc-600 disabled:cursor-not-allowed text-sky-300 border border-zinc-700 flex items-center gap-1 cursor-pointer"
                >
                  {isSelectedWhitelisted ? <Lock className="w-3 h-3" /> : null}
                  PIMEYES
                  {!isSelectedWhitelisted && <ExternalLink className="w-3 h-3" />}
                </button>
                <button
                  disabled={isSelectedWhitelisted}
                  onClick={() => !isSelectedWhitelisted && launchOsintFromEvent(selectedEvent.crop_url, 'yandex')}
                  title={
                    isSelectedWhitelisted
                      ? 'OpSec Lock: OSINT disabled for Whitelisted family members'
                      : 'Download crop & launch Yandex'
                  }
                  className="px-2 py-1 rounded text-[11px] font-mono bg-zinc-900 hover:bg-zinc-800 disabled:bg-zinc-950 disabled:opacity-35 disabled:text-zinc-600 disabled:cursor-not-allowed text-amber-300 border border-zinc-700 flex items-center gap-1 cursor-pointer"
                >
                  {isSelectedWhitelisted ? <Lock className="w-3 h-3" /> : null}
                  YANDEX
                  {!isSelectedWhitelisted && <ExternalLink className="w-3 h-3" />}
                </button>
                {viewMode === 'still' && (
                  <button
                    onClick={() => setShowSnapshotBbox((prev) => !prev)}
                    className={`px-2 py-1 rounded text-[11px] font-mono font-bold border flex items-center gap-1 cursor-pointer ${
                      showSnapshotBbox
                        ? 'bg-amber-500/20 text-amber-300 border-amber-500/50'
                        : 'bg-zinc-900 text-zinc-400 border-zinc-700'
                    }`}
                  >
                    <Crosshair className="w-3 h-3" />
                    {showSnapshotBbox ? 'BBOX: ON' : 'BBOX: OFF'}
                  </button>
                )}
                <button
                  onClick={(e) => handleDeepInspect(e, selectedEvent.id)}
                  disabled={inspectingId === selectedEvent.id}
                  className="px-2.5 py-1 rounded text-[11px] font-mono font-bold bg-emerald-600 hover:bg-emerald-500 text-white flex items-center gap-1 cursor-pointer"
                >
                  <Cpu className="w-3 h-3" />
                  {inspectingId === selectedEvent.id ? 'SCANNING...' : 'INSPECT'}
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

            <div className="bg-black aspect-video relative flex items-center justify-center overflow-hidden">
              {viewMode === 'clip' && selectedEvent.clip_url ? (
                <video
                  src={`${apiBase}${selectedEvent.clip_url}`}
                  controls
                  autoPlay
                  loop
                  playsInline
                  className="w-full h-full object-contain"
                />
              ) : selectedEvent.full_url ? (
                <>
                  <img
                    src={`${apiBase}${selectedEvent.full_url}`}
                    alt="Full frame capture"
                    className="w-full h-full object-contain"
                  />
                  {showSnapshotBbox && selectedEvent.bbox_w > 0 && (
                    <div
                      className="absolute border-2 border-red-500 bg-red-500/15 pointer-events-none"
                      style={{
                        left: `${selectedEvent.bbox_x * 100}%`,
                        top: `${selectedEvent.bbox_y * 100}%`,
                        width: `${selectedEvent.bbox_w * 100}%`,
                        height: `${selectedEvent.bbox_h * 100}%`,
                      }}
                    >
                      <span className="absolute -top-5 left-0 bg-red-600 text-white text-[10px] font-mono font-bold px-1.5 py-0.5 rounded-xs whitespace-nowrap">
                        {selectedEvent.full_label} · {Math.round(selectedEvent.confidence * 100)}%
                      </span>
                    </div>
                  )}
                </>
              ) : (
                <span className="text-xs font-mono text-zinc-500">
                  No snapshot image stored for this event
                </span>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
