import React, { useEffect, useState } from 'react';
import type { KnownFaceItem } from '../types/telemetry';
import {
  AlertTriangle,
  Check,
  Clock,
  Download,
  Edit3,
  ExternalLink,
  Globe,
  Lock,
  Plus,
  ShieldCheck,
  Trash2,
  Upload,
  UserCheck,
  Users,
  X,
} from 'lucide-react';

interface FaceGalleryProps {
  hostIp: string;
}

const ROLE_STYLES: Record<string, { label: string; badge: string; border: string }> = {
  whitelist: {
    label: 'WHITELIST (TRUSTED)',
    badge: 'bg-emerald-950 text-emerald-300 border-emerald-500/50',
    border: 'border-emerald-500/40',
  },
  visitor: {
    label: 'REPEAT VISITOR',
    badge: 'bg-amber-950 text-amber-300 border-amber-500/50',
    border: 'border-amber-500/40',
  },
  watchlist: {
    label: '🚨 WATCHLIST ALERT',
    badge: 'bg-red-950 text-red-300 border-red-500/70',
    border: 'border-red-500/70 shadow-[0_0_15px_rgba(239,68,68,0.2)]',
  },
};

export function FaceGallery({ hostIp }: FaceGalleryProps) {
  const apiBase = `http://${hostIp}:8080`;
  const [faces, setFaces] = useState<KnownFaceItem[]>([]);
  const [filterRole, setFilterRole] = useState<'all' | 'whitelist' | 'visitor' | 'watchlist'>('all');
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editName, setEditName] = useState<string>('');
  const [editRole, setEditRole] = useState<'whitelist' | 'visitor' | 'watchlist'>('whitelist');

  const [showImportDrawer, setShowImportDrawer] = useState<boolean>(false);
  const [importName, setImportName] = useState<string>('');
  const [importRole, setImportRole] = useState<'whitelist' | 'watchlist'>('watchlist');
  const [importUrl, setImportUrl] = useState<string>('');
  const [importStatus, setImportStatus] = useState<string | null>(null);

  const fetchFaces = async () => {
    try {
      const res = await fetch(`${apiBase}/api/faces`);
      if (res.ok) {
        const data: KnownFaceItem[] = await res.json();
        setFaces(data);
      }
    } catch (err) {
      console.error('Failed to fetch face gallery:', err);
    }
  };

  useEffect(() => {
    fetchFaces();
    const timer = window.setInterval(fetchFaces, 5000);
    return () => window.clearInterval(timer);
  }, [apiBase]);

  const handleSaveEdit = async (id: number) => {
    if (!editName.trim()) return;
    try {
      const res = await fetch(`${apiBase}/api/faces/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: editName.trim(), role: editRole }),
      });
      if (res.ok) {
        setFaces(await res.json());
        setEditingId(null);
      }
    } catch (err) {
      console.error('Failed to update face:', err);
    }
  };

  const handleDeleteFace = async (id: number) => {
    try {
      const res = await fetch(`${apiBase}/api/faces/${id}`, { method: 'DELETE' });
      if (res.ok) {
        setFaces(await res.json());
      }
    } catch (err) {
      console.error('Failed to delete face vector:', err);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!importName.trim()) {
      setImportStatus('Enter a Name/Label first before choosing a photo.');
      return;
    }
    setImportStatus('Extracting 128-D SFace vector from photo...');
    const reader = new FileReader();
    reader.onload = async () => {
      try {
        const res = await fetch(`${apiBase}/api/faces/enroll_image`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name: importName.trim(),
            role: importRole,
            image_base64: String(reader.result),
          }),
        });
        const data = await res.json();
        if (res.ok) {
          setFaces(data);
          setImportStatus(`Enrolled ${importName.toUpperCase()} (${importRole.toUpperCase()})!`);
          setImportName('');
        } else {
          setImportStatus(data.detail || 'No clear face found in photo.');
        }
      } catch (err) {
        setImportStatus('Upload failed.');
      }
    };
    reader.readAsDataURL(file);
  };

  const handleUrlScrape = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!importName.trim() || !importUrl.trim()) return;
    setImportStatus('Fetching URL & extracting YuNet/SFace biometric vector...');
    try {
      const res = await fetch(`${apiBase}/api/faces/enroll_url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: importName.trim(),
          role: importRole,
          image_url: importUrl.trim(),
        }),
      });
      const data = await res.json();
      if (res.ok) {
        setFaces(data);
        setImportStatus(`Scraped & enrolled ${importName.toUpperCase()}!`);
        setImportUrl('');
        setImportName('');
      } else {
        setImportStatus(data.detail || 'Could not extract face from URL.');
      }
    } catch (err) {
      setImportStatus('Failed to scrape image URL.');
    }
  };

  const launchOsintLookup = (cropUrl: string | null, engine: 'pimeyes' | 'yandex') => {
    if (cropUrl) {
      const a = document.createElement('a');
      a.href = `${apiBase}${cropUrl}`;
      a.download = `osint_face_crop.jpg`;
      a.target = '_blank';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
    const urls = {
      pimeyes: 'https://pimeyes.com/en',
      yandex: 'https://yandex.com/images/',
    };
    window.open(urls[engine], '_blank', 'noopener,noreferrer');
  };

  const filteredFaces = faces.filter((f) => (filterRole === 'all' ? true : f.role === filterRole));
  const counts = {
    all: faces.length,
    whitelist: faces.filter((f) => f.role === 'whitelist').length,
    visitor: faces.filter((f) => f.role === 'visitor').length,
    watchlist: faces.filter((f) => f.role === 'watchlist').length,
  };

  return (
    <div className="flex-1 flex flex-col gap-4 font-mono">
      <div className="bg-zinc-900/80 border border-zinc-800 rounded-lg p-3 sm:p-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <Users className="w-5 h-5 text-emerald-400" />
          <div>
            <h2 className="text-xs sm:text-sm font-bold text-zinc-100 uppercase">
              YUNET + SFACE BIOMETRIC GALLERY ({faces.length} VECTORS)
            </h2>
            <p className="text-[10px] text-zinc-400">
              Persistent 128-D vectors survive Kill-Feed PURGE · OpSec Lock active on Whitelist OSINT
            </p>
          </div>
        </div>

        <button
          onClick={() => {
            setShowImportDrawer((prev) => !prev);
            setImportStatus(null);
          }}
          className="bg-emerald-600 hover:bg-emerald-500 text-white px-3 py-1.5 rounded text-xs font-bold flex items-center gap-1.5 cursor-pointer"
        >
          <Plus className="w-4 h-4" />
          + ENROLL PHOTO / WATCHLIST URL
        </button>
      </div>

      {showImportDrawer && (
        <div className="bg-zinc-950 border border-emerald-500/50 rounded-lg p-4 space-y-3">
          <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
            <span className="text-xs font-bold text-emerald-400 uppercase">
              IMPORT FACE TO WHITELIST OR LOCAL WATCHLIST
            </span>
            <button onClick={() => setShowImportDrawer(false)} className="text-zinc-400 hover:text-white cursor-pointer">
              <X className="w-4 h-4" />
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div>
              <label className="text-[10px] text-zinc-400 block mb-1">1. SUBJECT NAME / LABEL:</label>
              <input
                type="text"
                value={importName}
                onChange={(e) => setImportName(e.target.value)}
                placeholder="e.g. ISAAC, JENNIFER, PROWLER_01..."
                className="w-full bg-zinc-900 border border-zinc-700 rounded px-2.5 py-1.5 text-xs text-zinc-100 uppercase"
              />
            </div>

            <div>
              <label className="text-[10px] text-zinc-400 block mb-1">2. SECURITY ROLE:</label>
              <div className="grid grid-cols-2 gap-1.5">
                <button
                  type="button"
                  onClick={() => setImportRole('whitelist')}
                  className={`py-1.5 rounded text-[10px] font-bold border flex items-center justify-center gap-1 cursor-pointer ${
                    importRole === 'whitelist'
                      ? 'bg-emerald-600 text-white border-emerald-400'
                      : 'bg-zinc-900 text-zinc-400 border-zinc-800'
                  }`}
                >
                  <ShieldCheck className="w-3.5 h-3.5" />
                  WHITELIST
                </button>
                <button
                  type="button"
                  onClick={() => setImportRole('watchlist')}
                  className={`py-1.5 rounded text-[10px] font-bold border flex items-center justify-center gap-1 cursor-pointer ${
                    importRole === 'watchlist'
                      ? 'bg-red-600 text-white border-red-400'
                      : 'bg-zinc-900 text-zinc-400 border-zinc-800'
                  }`}
                >
                  <AlertTriangle className="w-3.5 h-3.5" />
                  WATCHLIST
                </button>
              </div>
            </div>

            <div>
              <label className="text-[10px] text-zinc-400 block mb-1">3A. UPLOAD PHOTO (PC / PHONE):</label>
              <label className="w-full bg-sky-600 hover:bg-sky-500 text-white py-1.5 px-3 rounded text-xs font-bold flex items-center justify-center gap-1.5 cursor-pointer">
                <Upload className="w-3.5 h-3.5" />
                CHOOSE PHOTO FILE
                <input type="file" accept="image/*" onChange={handleFileUpload} className="hidden" />
              </label>
            </div>
          </div>

          <form onSubmit={handleUrlScrape} className="flex flex-wrap items-center gap-2 pt-2 border-t border-zinc-800">
            <Globe className="w-4 h-4 text-amber-400 shrink-0" />
            <input
              type="url"
              value={importUrl}
              onChange={(e) => setImportUrl(e.target.value)}
              placeholder="3B. Or paste public mugshot / offender / photo URL (https://...jpg) to scrape into database..."
              className="flex-1 min-w-[220px] bg-zinc-900 border border-zinc-700 rounded px-2.5 py-1.5 text-xs text-zinc-100"
            />
            <button
              type="submit"
              className="bg-amber-600 hover:bg-amber-500 text-white px-3 py-1.5 rounded text-xs font-bold cursor-pointer"
            >
              SCRAPE URL
            </button>
          </form>

          {importStatus && <div className="text-xs text-amber-300 font-bold">{importStatus}</div>}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2">
        {[
          { key: 'all', label: `ALL VECTORS (${counts.all})` },
          { key: 'whitelist', label: `🛡️ WHITELIST (${counts.whitelist})` },
          { key: 'visitor', label: `👤 REPEAT VISITORS (${counts.visitor})` },
          { key: 'watchlist', label: `🚨 WATCHLIST (${counts.watchlist})` },
        ].map((tab) => (
          <button
            key={tab.key}
            onClick={() => setFilterRole(tab.key as 'all' | 'whitelist' | 'visitor' | 'watchlist')}
            className={`px-3 py-1.5 rounded text-xs font-bold border cursor-pointer transition-colors ${
              filterRole === tab.key
                ? 'bg-emerald-600 text-white border-emerald-400'
                : 'bg-zinc-900 hover:bg-zinc-800 text-zinc-300 border-zinc-800'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {filteredFaces.length === 0 ? (
        <div className="bg-zinc-900/40 border border-zinc-800 rounded-lg py-16 text-center text-xs text-zinc-500">
          No face vectors in this category yet. Click [ID FACE] on any Kill-Feed card or use [+ ENROLL PHOTO / WATCHLIST URL] above.
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {filteredFaces.map((face) => {
            const style = ROLE_STYLES[face.role] || ROLE_STYLES.whitelist;
            const isWhitelisted = face.role === 'whitelist';
            const lastSeenStr = face.last_seen
              ? new Date(face.last_seen).toLocaleString([], {
                  month: 'short',
                  day: 'numeric',
                  hour: '2-digit',
                  minute: '2-digit',
                })
              : 'Unknown';

            return (
              <div
                key={face.id}
                className={`bg-zinc-950 border rounded-lg p-3 flex flex-col justify-between gap-3 ${style.border}`}
              >
                <div className="flex gap-3 items-start">
                  <div className="w-20 h-20 rounded-md overflow-hidden bg-zinc-900 border border-zinc-800 shrink-0 flex items-center justify-center">
                    {face.sample_crop ? (
                      <img
                        src={`${apiBase}${face.sample_crop}`}
                        alt={face.name}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <UserCheck className="w-7 h-7 text-zinc-700" />
                    )}
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-1">
                      <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded border ${style.badge}`}>
                        {style.label}
                      </span>
                      <span className="text-[10px] text-zinc-500">#{face.id}</span>
                    </div>

                    <div className="text-sm font-bold text-zinc-100 truncate mt-1.5">{face.name}</div>

                    <div className="text-[11px] text-emerald-400 font-semibold mt-0.5">
                      Seen {face.sightings} {face.sightings === 1 ? 'time' : 'times'}
                    </div>
                    <div className="text-[10px] text-zinc-500 flex items-center gap-1 mt-0.5">
                      <Clock className="w-3 h-3" />
                      {lastSeenStr}
                    </div>
                  </div>
                </div>

                {editingId === face.id ? (
                  <div className="bg-zinc-900 border border-zinc-700 rounded p-2 space-y-2">
                    <input
                      type="text"
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                      className="w-full bg-zinc-950 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-100 uppercase"
                    />
                    <div className="grid grid-cols-3 gap-1">
                      {(['whitelist', 'visitor', 'watchlist'] as const).map((r) => (
                        <button
                          key={r}
                          type="button"
                          onClick={() => setEditRole(r)}
                          className={`py-1 rounded text-[9px] font-bold uppercase border cursor-pointer ${
                            editRole === r
                              ? 'bg-sky-600 text-white border-sky-400'
                              : 'bg-zinc-950 text-zinc-400 border-zinc-800'
                          }`}
                        >
                          {r}
                        </button>
                      ))}
                    </div>
                    <div className="flex justify-end gap-1">
                      <button
                        onClick={() => setEditingId(null)}
                        className="px-2 py-1 rounded text-[10px] bg-zinc-800 text-zinc-300 cursor-pointer"
                      >
                        CANCEL
                      </button>
                      <button
                        onClick={() => handleSaveEdit(face.id)}
                        className="px-2.5 py-1 rounded text-[10px] font-bold bg-emerald-600 text-white flex items-center gap-1 cursor-pointer"
                      >
                        <Check className="w-3 h-3" />
                        SAVE
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-2 pt-2 border-t border-zinc-900">
                    {/* Layer 3: OSINT Bar (Greyed out when role === 'whitelist') */}
                    <div className="flex items-center justify-between gap-1">
                      {face.sample_crop && (
                        <a
                          href={`${apiBase}${face.sample_crop}`}
                          download={`face_${face.name}.jpg`}
                          title="Download aligned face crop locally"
                          className="bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 px-2 py-1 rounded text-[10px] text-zinc-300 flex items-center gap-1"
                        >
                          <Download className="w-3 h-3 text-emerald-400" />
                          CROP
                        </a>
                      )}
                      <button
                        disabled={isWhitelisted}
                        onClick={() => !isWhitelisted && launchOsintLookup(face.sample_crop, 'pimeyes')}
                        title={
                          isWhitelisted
                            ? 'OpSec Lock: External OSINT disabled for Whitelisted family members'
                            : 'Download crop & launch PimEyes Face Search'
                        }
                        className="flex-1 bg-zinc-900 hover:bg-zinc-800 disabled:bg-zinc-950 disabled:opacity-35 disabled:text-zinc-600 disabled:cursor-not-allowed border border-zinc-800 py-1 rounded text-[10px] text-sky-300 flex items-center justify-center gap-1 cursor-pointer"
                      >
                        {isWhitelisted ? <Lock className="w-2.5 h-2.5" /> : null}
                        PIMEYES
                        {!isWhitelisted && <ExternalLink className="w-2.5 h-2.5" />}
                      </button>
                      <button
                        disabled={isWhitelisted}
                        onClick={() => !isWhitelisted && launchOsintLookup(face.sample_crop, 'yandex')}
                        title={
                          isWhitelisted
                            ? 'OpSec Lock: External OSINT disabled for Whitelisted family members'
                            : 'Download crop & launch Yandex Reverse Face Search'
                        }
                        className="flex-1 bg-zinc-900 hover:bg-zinc-800 disabled:bg-zinc-950 disabled:opacity-35 disabled:text-zinc-600 disabled:cursor-not-allowed border border-zinc-800 py-1 rounded text-[10px] text-amber-300 flex items-center justify-center gap-1 cursor-pointer"
                      >
                        {isWhitelisted ? <Lock className="w-2.5 h-2.5" /> : null}
                        YANDEX
                        {!isWhitelisted && <ExternalLink className="w-2.5 h-2.5" />}
                      </button>
                    </div>

                    <div className="flex items-center justify-between gap-1.5">
                      <button
                        onClick={() => {
                          setEditingId(face.id);
                          setEditName(face.name);
                          setEditRole(face.role);
                        }}
                        className="flex-1 bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 py-1 rounded text-[10px] text-zinc-200 flex items-center justify-center gap-1 cursor-pointer"
                      >
                        <Edit3 className="w-3 h-3" />
                        RENAME / ROLE
                      </button>
                      <button
                        onClick={() => handleDeleteFace(face.id)}
                        title="Delete this specific 128-D face vector"
                        className="bg-red-950/70 hover:bg-red-900 border border-red-500/40 px-2.5 py-1 rounded text-[10px] text-red-300 flex items-center gap-1 cursor-pointer"
                      >
                        <Trash2 className="w-3 h-3" />
                        DELETE
                      </button>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
