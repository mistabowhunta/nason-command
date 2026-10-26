import { useEffect, useState } from 'react';
import type { ConnectionState, TelemetryHeartbeat } from '../types/telemetry';
import { Activity, Bell, Check, Cpu, Send, Shield, Wifi, X } from 'lucide-react';

interface StatusHeaderProps {
  telemetry: TelemetryHeartbeat | null;
  status: ConnectionState;
  hostIp?: string;
}

function formatUptime(totalSeconds?: number): string {
  if (!totalSeconds || totalSeconds < 0) return '0 d : 0 h : 0 m : 0 s';
  const d = Math.floor(totalSeconds / 86400);
  const h = Math.floor((totalSeconds % 86400) / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = totalSeconds % 60;
  return `${d} d : ${h} h : ${m} m : ${s} s`;
}

export function StatusHeader({ telemetry, status, hostIp = window.location.hostname || '192.168.1.27' }: StatusHeaderProps) {
  const apiBase = `http://${hostIp}:8080`;
  const [showPushModal, setShowPushModal] = useState(false);
  const [enabled, setEnabled] = useState(true);
  const [ntfyTopic, setNtfyTopic] = useState('nason_command_97113');
  const [discordWebhook, setDiscordWebhook] = useState('');
  const [notifyBreach, setNotifyBreach] = useState(true);
  const [notifyWatchlist, setNotifyWatchlist] = useState(true);
  const [notifyDelivery, setNotifyDelivery] = useState(true);
  const [pushStatus, setPushStatus] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${apiBase}/api/push/settings`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data) {
          setEnabled(data.enabled);
          setNtfyTopic(data.ntfy_topic || '');
          setDiscordWebhook(data.discord_webhook || '');
          setNotifyBreach(data.notify_breach);
          setNotifyWatchlist(data.notify_watchlist);
          setNotifyDelivery(data.notify_delivery);
        }
      })
      .catch(() => {});
  }, [apiBase]);

  const savePushSettings = async () => {
    setPushStatus('Saving...');
    try {
      const res = await fetch(`${apiBase}/api/push/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          enabled,
          ntfy_topic: ntfyTopic,
          discord_webhook: discordWebhook,
          notify_breach: notifyBreach,
          notify_watchlist: notifyWatchlist,
          notify_delivery: notifyDelivery,
        }),
      });
      if (res.ok) setPushStatus('Saved Push Settings!');
    } catch {
      setPushStatus('Failed to save.');
    }
  };

  const sendTestAlert = async () => {
    await savePushSettings();
    setPushStatus('Sending test alert...');
    try {
      const res = await fetch(`${apiBase}/api/push/test`, { method: 'POST' });
      const data = await res.json();
      if (data.sent) {
        setPushStatus(`Sent via ${data.channels.join(', ')}!`);
      } else {
        setPushStatus('Push disabled or no channel configured.');
      }
    } catch {
      setPushStatus('Test push failed.');
    }
  };

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

      <div className="flex items-center gap-2.5 sm:gap-4 text-[11px] sm:text-xs font-mono">
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

        <button
          onClick={() => {
            setShowPushModal((prev) => !prev);
            setPushStatus(null);
          }}
          className={`px-2 py-1 rounded text-[10px] font-bold border flex items-center gap-1 cursor-pointer ${
            enabled
              ? 'bg-emerald-950/80 text-emerald-300 border-emerald-500/40'
              : 'bg-zinc-900 text-zinc-400 border-zinc-700'
          }`}
        >
          <Bell className="w-3 h-3" />
          <span>ALERTS</span>
        </button>

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

      {showPushModal && (
        <div className="fixed top-12 right-3 sm:right-6 z-50 bg-zinc-950 border border-emerald-500/50 rounded-lg p-4 w-80 shadow-2xl font-mono text-xs space-y-3">
          <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
            <span className="font-bold text-emerald-400 uppercase">📲 MOBILE PUSH ALERTS</span>
            <button onClick={() => setShowPushModal(false)} className="text-zinc-400 hover:text-white cursor-pointer">
              <X className="w-4 h-4" />
            </button>
          </div>

          <div className="text-[10px] text-zinc-400 leading-relaxed">
            Install the free <b>ntfy</b> app on Android/iOS and subscribe to your private topic below for instant photo alerts (Whitelisted family never trigger alerts).
          </div>

          <div className="space-y-2">
            <div>
              <label className="text-[10px] text-zinc-400 block mb-1">NTFY.SH PRIVATE TOPIC:</label>
              <input
                type="text"
                value={ntfyTopic}
                onChange={(e) => setNtfyTopic(e.target.value)}
                placeholder="nason_command_97113"
                className="w-full bg-zinc-900 border border-zinc-700 rounded px-2 py-1.5 text-xs text-zinc-100"
              />
            </div>

            <div>
              <label className="text-[10px] text-zinc-400 block mb-1">OPTIONAL DISCORD WEBHOOK URL:</label>
              <input
                type="text"
                value={discordWebhook}
                onChange={(e) => setDiscordWebhook(e.target.value)}
                placeholder="https://discord.com/api/webhooks/..."
                className="w-full bg-zinc-900 border border-zinc-700 rounded px-2 py-1.5 text-xs text-zinc-100"
              />
            </div>

            <div className="space-y-1 pt-1">
              <label className="flex items-center gap-2 cursor-pointer text-[11px]">
                <input type="checkbox" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} />
                <span>Master Push Alerts Enabled</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer text-[11px] text-red-300">
                <input type="checkbox" checked={notifyWatchlist} onChange={(e) => setNotifyWatchlist(e.target.checked)} />
                <span>🚨 Watchlist Biometric Match</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer text-[11px] text-amber-300">
                <input type="checkbox" checked={notifyBreach} onChange={(e) => setNotifyBreach(e.target.checked)} />
                <span>⚠️ Perimeter Tripwire Breach</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer text-[11px] text-sky-300">
                <input type="checkbox" checked={notifyDelivery} onChange={(e) => setNotifyDelivery(e.target.checked)} />
                <span>📦 Delivery Van (Amazon/UPS/FedEx)</span>
              </label>
            </div>
          </div>

          <div className="flex gap-2 pt-2 border-t border-zinc-800">
            <button
              onClick={savePushSettings}
              className="flex-1 bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-1.5 rounded text-[11px] flex items-center justify-center gap-1 cursor-pointer"
            >
              <Check className="w-3.5 h-3.5" />
              SAVE
            </button>
            <button
              onClick={sendTestAlert}
              className="flex-1 bg-sky-600 hover:bg-sky-500 text-white font-bold py-1.5 rounded text-[11px] flex items-center justify-center gap-1 cursor-pointer"
            >
              <Send className="w-3.5 h-3.5" />
              TEST PUSH
            </button>
          </div>

          {pushStatus && <div className="text-[11px] text-amber-300 font-bold">{pushStatus}</div>}
        </div>
      )}
    </header>
  );
}
