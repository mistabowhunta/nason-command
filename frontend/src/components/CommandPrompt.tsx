import React, { useState } from 'react';
import { Cpu, Send, Sparkles, Terminal } from 'lucide-react';

interface CommandPromptProps {
  hostIp: string;
  onFocusStream: (stream: string | null) => void;
  onRefreshRail: () => void;
}

const QUICK_CHIPS = [
  { label: '📊 DB SUMMARY', prompt: 'Give me a telemetry log summary' },
  { label: '🔍 INSPECT LATEST', prompt: 'Inspect latest detection' },
  { label: '🎥 DRIVEWAY', prompt: 'Focus driveway' },
  { label: '🎥 LOFT', prompt: 'Focus loft' },
  { label: '🎥 LIVING ROOM', prompt: 'Focus living room' },
  { label: '🖥️ 3-CAM GRID', prompt: 'Restore grid view' },
];

export function CommandPrompt({ hostIp, onFocusStream, onRefreshRail }: CommandPromptProps) {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [lastReply, setLastReply] = useState<{ text: string; engine: string } | null>(null);

  const runPrompt = async (promptText: string) => {
    if (!promptText.trim()) return;
    setLoading(true);
    try {
      const res = await fetch(`http://${hostIp}:8080/api/agent/command`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: promptText }),
      });
      if (res.ok) {
        const data = await res.json();
        setLastReply({ text: data.reply, engine: data.engine });
        if (data.ui_action?.type === 'SET_FOCUS') {
          onFocusStream(data.ui_action.stream);
        } else if (data.ui_action?.type === 'REFRESH_RAIL') {
          onRefreshRail();
        }
      }
    } catch (err) {
      setLastReply({ text: 'Failed to reach Command AI orchestrator.', engine: 'ERROR' });
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    runPrompt(input);
    setInput('');
  };

  return (
    <footer className="border-t border-zinc-800 bg-zinc-900/95 px-3 sm:px-6 py-2.5 flex flex-col gap-2 z-20">
      {lastReply && (
        <div className="bg-zinc-950 border border-emerald-500/40 rounded px-3 py-2 flex flex-wrap items-center justify-between gap-2 font-mono text-xs">
          <div className="flex items-center gap-2 text-emerald-300">
            <Sparkles className="w-4 h-4 shrink-0 text-emerald-400" />
            <span>{lastReply.text}</span>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <span className="text-[10px] px-2 py-0.5 rounded bg-zinc-900 border border-zinc-700 text-amber-300 flex items-center gap-1">
              <Cpu className="w-3 h-3" />
              {lastReply.engine}
            </span>
            <button
              onClick={() => setLastReply(null)}
              className="text-zinc-500 hover:text-zinc-300 text-[10px] cursor-pointer"
            >
              [CLEAR]
            </button>
          </div>
        </div>
      )}

      <div className="flex flex-wrap items-center justify-between gap-2">
        <form onSubmit={handleSubmit} className="flex-1 flex items-center gap-2 min-w-[240px]">
          <div className="hidden sm:flex items-center gap-1.5 text-emerald-400 font-mono text-xs font-bold shrink-0">
            <Terminal className="w-4 h-4" />
            <span>COMMAND &gt;</span>
          </div>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Command directive (e.g., 'Focus driveway', 'Inspect latest')..."
            className="flex-1 bg-zinc-950 border border-zinc-800 focus:border-emerald-500/60 rounded px-3 py-1.5 text-xs font-mono text-zinc-100 outline-none"
          />
          <button
            type="submit"
            disabled={loading}
            className="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 px-3 py-1.5 rounded text-xs font-mono font-bold text-white flex items-center gap-1.5 cursor-pointer"
          >
            <Send className="w-3.5 h-3.5" />
            {loading ? '...' : 'SEND'}
          </button>
        </form>

        <div className="flex items-center gap-1.5 overflow-x-auto max-w-full pb-0.5">
          {QUICK_CHIPS.map((chip) => (
            <button
              key={chip.label}
              onClick={() => runPrompt(chip.prompt)}
              className="bg-zinc-950 hover:bg-zinc-800 border border-zinc-800 px-2 py-1 rounded text-[10px] font-mono text-zinc-300 whitespace-nowrap cursor-pointer transition-colors"
            >
              {chip.label}
            </button>
          ))}
        </div>
      </div>
    </footer>
  );
}
