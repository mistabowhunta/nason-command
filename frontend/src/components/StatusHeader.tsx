import type { ConnectionState, TelemetryHeartbeat } from '../types/telemetry';
import { Activity, Cpu, Shield, Wifi } from 'lucide-react';

interface StatusHeaderProps {
  telemetry: TelemetryHeartbeat | null;
  status: ConnectionState;
}

function formatUptime(totalSeconds?: number): string {
  if (!totalSeconds || totalSeconds < 0) return '0 d : 0 h : 0 m : 0 s';
  const d = Math.floor(totalSeconds / 86400);
  const h = Math.floor((totalSeconds % 86400) / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = totalSeconds % 60;
  return `${d} d : ${h} h : ${m} m : ${s} s`;
}

export function StatusHeader({ telemetry, status }: StatusHeaderProps) {
  return (
    <header className="border-b border-zinc-800 bg-zinc-900/90 backdrop-blur-md px-3 sm:px-6 py-2.5 flex flex-wrap items-center justify-between gap-2 sticky top-0 z-30">
      <div className="flex items-center gap-2.5">
        <Shield className="w-5 h-5 text-emerald-400 shrink-0" />
        <div className="flex items-center gap-2">
          <h1 className="font-mono font-bold tracking-wider text-xs sm:text-sm uppercase text-zinc-100">
            NasonNation <span className="text-emerald-400">//</span> Command
          </h1>
          <span className="hidden sm:inline-block text-[10px] font-mono bg-zinc-800 text-zinc-300 px-2 py-0.5 rounded">
            NODE: BOSSMAN
          </span>
        </div>
      </div>

      <div className="flex items-center gap-3 sm:gap-5 text-[11px] sm:text-xs font-mono">
        <div className="flex items-center gap-1.5 text-zinc-300">
          <Activity className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
          <span className="text-zinc-500 hidden md:inline">UPTIME:</span>
          <span className="text-zinc-100 font-semibold">{formatUptime(telemetry?.uptime_seconds)}</span>
        </div>

        <div className="hidden sm:flex items-center gap-1.5 text-zinc-300">
          <Cpu className="w-3.5 h-3.5 text-amber-400 shrink-0" />
          <span className="text-zinc-400">AGENT:</span>
          <span className="text-amber-300 font-semibold">{telemetry?.agent_mode ?? 'YOLOv8s // RTX 3060 Ti'}</span>
        </div>

        <div className="flex items-center gap-1.5">
          <Wifi className="w-3.5 h-3.5 text-zinc-400 shrink-0" />
          <span
            className={`px-2 py-0.5 rounded font-bold ${
              status === 'ONLINE'
                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                : status === 'CONNECTING'
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                : 'bg-red-500/20 text-red-400 border border-red-500/30'
            }`}
          >
            {status}
          </span>
        </div>
      </div>
    </header>
  );
}
