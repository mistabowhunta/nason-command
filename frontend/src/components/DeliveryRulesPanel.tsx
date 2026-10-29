import React, { useEffect, useState } from 'react';

interface DeliveryRule {
  id: string;
  name: string;
  kind: string;
  alert_mode: 'alert' | 'silent';
  cues: string;
  sightings: number;
  last_seen: string | null;
}

export const DeliveryRulesPanel: React.FC<{ apiBase?: string }> = ({ apiBase }) => {
  const base = apiBase || `http://${window.location.hostname}:8080`;
  const [rules, setRules] = useState<DeliveryRule[]>([]);
  const [expanded, setExpanded] = useState<boolean>(true);
  const [newName, setNewName] = useState('');
  const [newCues, setNewCues] = useState('');
  const [newMode, setNewMode] = useState<'alert' | 'silent'>('alert');

  const loadRules = async () => {
    try {
      const r = await fetch(`${base}/api/delivery-rules`);
      if (r.ok) {
        const d = await r.json();
        setRules(d.rules || []);
      }
    } catch {}
  };

  useEffect(() => {
    loadRules();
  }, []);

  const toggleMode = async (rule: DeliveryRule) => {
    const nextMode = rule.alert_mode === 'alert' ? 'silent' : 'alert';
    const r = await fetch(`${base}/api/delivery-rules`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: rule.name,
        kind: rule.kind,
        alert_mode: nextMode,
        cues: rule.cues,
      }),
    });
    if (r.ok) {
      const d = await r.json();
      setRules(d.rules || []);
    }
  };

  const addRule = async () => {
    if (!newName.trim()) return;
    const r = await fetch(`${base}/api/delivery-rules`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: newName.trim().toUpperCase(),
        kind: newMode === 'silent' ? 'service' : 'carrier',
        alert_mode: newMode,
        cues: newCues.trim(),
      }),
    });
    if (r.ok) {
      const d = await r.json();
      setRules(d.rules || []);
      setNewName('');
      setNewCues('');
    }
  };

  const removeRule = async (id: string) => {
    const r = await fetch(`${base}/api/delivery-rules/${id}`, { method: 'DELETE' });
    if (r.ok) {
      const d = await r.json();
      setRules(d.rules || []);
    }
  };

  return (
    <div className="col-span-full mb-2 p-3 rounded-lg border border-amber-500/30 bg-zinc-950/90 self-start">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-3">
          <span className="text-xs font-mono font-bold text-amber-300 uppercase tracking-wider">
            📦 DELIVERY & SERVICE RULES ({rules.length})
          </span>
          <span className="text-[11px] font-mono text-zinc-400 hidden sm:inline">
            Logo / Uniform / Vehicle / Gear detection — No facial vector required
          </span>
        </div>
        <button
          onClick={() => setExpanded((prev) => !prev)}
          className="px-2.5 py-1 rounded text-[10px] font-mono font-bold border border-zinc-700 bg-zinc-900 hover:bg-zinc-800 text-zinc-300 cursor-pointer"
        >
          {expanded ? '▲ COLLAPSE RULES' : '▼ CONFIGURE RULES'}
        </button>
      </div>

      {expanded && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2.5 mt-3 items-start">
            {rules.map((rule) => (
              <div
                key={rule.id}
                className="p-2.5 rounded border border-zinc-800 bg-zinc-900/70 flex flex-col justify-between gap-2 h-full"
              >
                <div className="flex items-center justify-between gap-1">
                  <span className="font-mono font-bold text-xs text-zinc-100 truncate">{rule.name}</span>
                  <div className="flex items-center gap-1 shrink-0">
                    <button
                      onClick={() => toggleMode(rule)}
                      className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold border cursor-pointer whitespace-nowrap ${
                        rule.alert_mode === 'alert'
                          ? 'bg-amber-950/80 text-amber-300 border-amber-500/50 hover:bg-amber-900/60'
                          : 'bg-emerald-950/80 text-emerald-300 border-emerald-500/50 hover:bg-emerald-900/60'
                      }`}
                      title="Toggle between Alert on Arrival vs Silent Log"
                    >
                      {rule.alert_mode === 'alert' ? '🔔 ALERT' : '🔇 SILENT'}
                    </button>
                    <button
                      onClick={() => removeRule(rule.id)}
                      className="text-zinc-500 hover:text-red-400 text-xs font-mono px-1 cursor-pointer"
                      title="Delete rule"
                    >
                      ✕
                    </button>
                  </div>
                </div>
                <div className="text-[10px] font-mono text-zinc-400 line-clamp-2 leading-relaxed" title={rule.cues}>
                  {rule.cues}
                </div>
                <div className="text-[10px] font-mono text-zinc-500 flex justify-between pt-1 border-t border-zinc-800/60">
                  <span>Seen: {rule.sightings || 0}x</span>
                  {rule.last_seen && (
                    <span>{new Date(rule.last_seen).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                  )}
                </div>
              </div>
            ))}
          </div>

          <div className="mt-3 pt-2.5 border-t border-zinc-800/80 flex flex-wrap items-center gap-2">
            <input
              type="text"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="Company/Service (e.g. DHL, GARBAGE)"
              className="bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1 text-xs font-mono text-zinc-200 w-48"
            />
            <input
              type="text"
              value={newCues}
              onChange={(e) => setNewCues(e.target.value)}
              placeholder="Visual cues (e.g. yellow van, uniform, mower, logo)"
              className="bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1 text-xs font-mono text-zinc-200 flex-1 min-w-[220px]"
            />
            <select
              value={newMode}
              onChange={(e) => setNewMode(e.target.value as 'alert' | 'silent')}
              className="bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1 text-xs font-mono text-zinc-200"
            >
              <option value="alert">🔔 Alert on Arrival</option>
              <option value="silent">🔇 Silent (No Alert)</option>
            </select>
            <button
              onClick={addRule}
              className="bg-amber-500 hover:bg-amber-400 text-black font-mono font-bold text-xs px-3 py-1 rounded cursor-pointer"
            >
              + ADD RULE
            </button>
          </div>
        </>
      )}
    </div>
  );
};
