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
    <div className="mb-4 p-3 rounded-lg border border-amber-500/30 bg-zinc-900/90">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
        <div>
          <div className="text-xs font-mono font-bold text-amber-300 uppercase tracking-wider">
            📦 DELIVERY & SERVICE RULES (LOGO / VEHICLE / GEAR — NO FACE VECTOR NEEDED)
          </div>
          <div className="text-[11px] font-mono text-zinc-400">
            Carriers (Amazon/UPS/FedEx/USPS) alert on logo/van/vest. Landscapers log silently with zero alerts or visitor face clutter.
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2 mt-2.5">
        {rules.map((rule) => (
          <div key={rule.id} className="p-2.5 rounded border border-zinc-800 bg-zinc-950/90 flex flex-col justify-between gap-1.5">
            <div className="flex items-center justify-between gap-1.5">
              <span className="font-mono font-bold text-xs text-zinc-100">{rule.name}</span>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => toggleMode(rule)}
                  className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold border cursor-pointer ${
                    rule.alert_mode === 'alert'
                      ? 'bg-amber-950/80 text-amber-300 border-amber-500/50 hover:bg-amber-900/60'
                      : 'bg-emerald-950/80 text-emerald-300 border-emerald-500/50 hover:bg-emerald-900/60'
                  }`}
                >
                  {rule.alert_mode === 'alert' ? '🔔 ALERT ON ARRIVAL' : '🔇 SILENT (NO ALERT)'}
                </button>
                <button
                  onClick={() => removeRule(rule.id)}
                  className="text-zinc-500 hover:text-red-400 text-xs font-mono px-1 cursor-pointer"
                  title="Remove rule"
                >
                  ✕
                </button>
              </div>
            </div>
            <div className="text-[10px] font-mono text-zinc-400 line-clamp-2">{rule.cues}</div>
            <div className="text-[10px] font-mono text-zinc-500 flex justify-between">
              <span>Sightings: {rule.sightings || 0}</span>
              {rule.last_seen && (
                <span>Last: {new Date(rule.last_seen).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-2.5 pt-2 border-t border-zinc-800/80 flex flex-wrap items-center gap-2">
        <input
          type="text"
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          placeholder="Company/Service (e.g. DHL, PEST CONTROL)"
          className="bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1 text-xs font-mono text-zinc-200 flex-1 min-w-[160px]"
        />
        <input
          type="text"
          value={newCues}
          onChange={(e) => setNewCues(e.target.value)}
          placeholder="Visual cues (e.g. yellow van, uniform, mower)"
          className="bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1 text-xs font-mono text-zinc-200 flex-1 min-w-[180px]"
        />
        <select
          value={newMode}
          onChange={(e) => setNewMode(e.target.value as 'alert' | 'silent')}
          className="bg-zinc-950 border border-zinc-800 rounded px-2 py-1 text-xs font-mono text-zinc-200"
        >
          <option value="alert">🔔 Alert on Arrival</option>
          <option value="silent">🔇 Silent (No Alert)</option>
        </select>
        <button
          onClick={addRule}
          className="bg-amber-600 hover:bg-amber-500 text-black font-mono font-bold text-xs px-3 py-1 rounded cursor-pointer"
        >
          + ADD RULE
        </button>
      </div>
    </div>
  );
};
