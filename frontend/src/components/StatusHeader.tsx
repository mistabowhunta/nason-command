import type { TelemetryHeartbeat, ConnectionState } from '../types/telemetry';
import { Activity, Cpu, Radio, ShieldAlert } from 'lucide-react';

interface StatusHeaderProps {
  telemetry: TelemetryHeartbeat | null;
  status: ConnectionState;
}

export function StatusHeader({ telemetry, status }: StatusHeaderProps) {
  return (
    <header className="border-b border-zinc-800 bg-zinc-900/90 px-6 py-3 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <ShieldAlert className="w-6 h-6 text-emerald-500" />
        <h1 className="text-lg font-bold tracking-wider uppercase text-zinc-100">
          NasonNation <span className="text-emerald-500 font-mono">| COMMAND</span>
        </h1>
      </div>

      <div className="flex items-center gap-6 text-xs font-mono">
        <div className="flex items-center gap-2">
          <Radio className={`w-4 h-4 ${status === 'ONLINE' ? 'text-emerald-400 animate-pulse' : 'text-red-500'}`} />
          <span className="text-zinc-400">WS LINK:</span>
          <span className={status === 'ONLINE' ? 'text-emerald-400 font-bold' : 'text-red-400 font-bold'}>
            {status}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-sky-400" />
          <span className="text-zinc-400">UPTIME:</span>
          <span className="text-zinc-200">{telemetry ? `${telemetry.uptime_seconds}s` : '--'}</span>
        </div>

        <div className="flex items-center gap-2">
          <Cpu className="w-4 h-4 text-amber-400" />
          <span className="text-zinc-400">AGENT:</span>
          <span className="text-amber-300">{telemetry?.agent_mode ?? 'INITIALIZING'}</span>
        </div>
      </div>
    </header>
  );
}
