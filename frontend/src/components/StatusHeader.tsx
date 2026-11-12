import { useEffect, useState } from 'react';
import type { CameraScheduleConfig, ConnectionState, TelemetryHeartbeat } from '../types/telemetry';
import { Activity, Bell, BellOff, BookOpen, Bot, Camera, Check, Cpu, Home, Plane, Send, Shield, Wifi, X } from 'lucide-react';

interface StatusHeaderProps {
  telemetry: TelemetryHeartbeat | null;
  status: ConnectionState;
  hostIp?: string;
  onSchedulesChanged?: (schedules: Record<string, CameraScheduleConfig>) => void;
}

function formatUptime(totalSeconds?: number): string {
  if (!totalSeconds || totalSeconds < 0) return '0 d : 0 h : 0 m : 0 s';
  const d = Math.floor(totalSeconds / 86400);
  const h = Math.floor((totalSeconds % 86400) / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = totalSeconds % 60;
  return `${d} d : ${h} h : ${m} m : ${s} s`;
}

function formatMmSs(sec?: number): string {
  const safe = Math.max(0, sec || 0);
  const m = Math.floor(safe / 60);
  const s = safe % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}

export function StatusHeader({
  telemetry,
  status,
  hostIp = window.location.hostname || '192.168.1.27',
  onSchedulesChanged,
}: StatusHeaderProps) {
  const apiBase = `http://${hostIp}:8080`;
  const [showPushModal, setShowPushModal] = useState(false);
  const [enabled, setEnabled] = useState(true);
  const [ntfyTopic, setNtfyTopic] = useState('nason_command_97113');
  const [discordWebhook, setDiscordWebhook] = useState('');
  const [discordBotToken, setDiscordBotToken] = useState('');
  const [notifyBreach, setNotifyBreach] = useState(true);
  const [notifyWatchlist, setNotifyWatchlist] = useState(true);
  const [notifyDelivery, setNotifyDelivery] = useState(true);
  const [dailySitrepEnabled, setDailySitrepEnabled] = useState(true);
  const [dailySitrepTime, setDailySitrepTime] = useState('05:00');
  const [localAway, setLocalAway] = useState(false);
  const [botUsername, setBotUsername] = useState<string | null>(null);
  const [botError, setBotError] = useState<string | null>(null);
  const [pushStatus, setPushStatus] = useState<string | null>(null);

  const loadSettings = () => {
    fetch(`${apiBase}/api/push/settings`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data) {
          setEnabled(Boolean(data.enabled));
          setNtfyTopic(data.ntfy_topic || '');
          setDiscordWebhook(data.discord_webhook || '');
          setDiscordBotToken(data.discord_bot_token || '');
          setNotifyBreach(Boolean(data.notify_breach));
          setNotifyWatchlist(Boolean(data.notify_watchlist));
          setNotifyDelivery(Boolean(data.notify_delivery));
          setDailySitrepEnabled(data.daily_sitrep_enabled !== false);
          setDailySitrepTime(data.daily_sitrep_time || '05:00');
          setLocalAway(Boolean(data.away_mode));
          setBotUsername(data.bot_status?.bot_username || null);
          setBotError(data.bot_status?.last_error || null);
        }
      })
      .catch(() => {});
  };

  useEffect(() => {
    loadSettings();
  }, [apiBase]);

  const isAway = telemetry?.away_mode !== undefined ? telemetry.away_mode : localAway;
  const awayGraceSec = telemetry?.away_grace_remaining || 0;
  const muteRemSec = telemetry?.mute_remaining || 0;
  const isBotConnected = Boolean(telemetry?.bot_connected);

  const toggleAwayMode = async (targetAway: boolean, graceSeconds = 180) => {
    setLocalAway(targetAway);
    try {
      const res = await fetch(`${apiBase}/api/away`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ away_mode: targetAway, grace_seconds: graceSeconds }),
      });
      if (res.ok) {
        const data = await res.json();
        setLocalAway(Boolean(data.away_mode));
        if (data.schedules && onSchedulesChanged) {
          onSchedulesChanged(data.schedules);
        }
      }
    } catch {
      // ignore transient error
    }
  };

  const toggleMute = async (minutes: number) => {
    try {
      await fetch(`${apiBase}/api/push/mute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ minutes }),
      });
      setPushStatus(minutes > 0 ? `Snoozed push alerts for ${minutes}m.` : 'Push alerts resumed!');
    } catch {
      setPushStatus('Failed to update mute timer.');
    }
  };

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
          discord_bot_token: discordBotToken,
          notify_breach: notifyBreach,
          notify_watchlist: notifyWatchlist,
          notify_delivery: notifyDelivery,
          daily_sitrep_enabled: dailySitrepEnabled,
          daily_sitrep_time: dailySitrepTime,
        }),
      });
      if (res.ok) {
        const updated = await res.json();
        setBotUsername(updated.bot_status?.bot_username || null);
        setBotError(updated.bot_status?.last_error || null);
        setPushStatus('Saved Push & Discord Bot Settings!');
      }
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
        setPushStatus(data.reason || 'Push disabled or no channel configured.');
      }
    } catch {
      setPushStatus('Test push failed.');
    }
  };

  const sendManualSitrep = async () => {
    await savePushSettings();
    setPushStatus('Capturing 3-cam silent sitrep...');
    try {
      const res = await fetch(`${apiBase}/api/push/sitrep`, { method: 'POST' });
      const data = await res.json();
      if (data.sent) {
        setPushStatus('Silent 3-Cam Sitrep posted to Discord!');
      } else {
        setPushStatus(data.reason || 'Sitrep failed.');
      }
    } catch {
      setPushStatus('Sitrep request failed.');
    }
  };

  const sendCheatsheet = async () => {
    await savePushSettings();
    setPushStatus('Posting Command Cheat Sheet...');
    try {
      const res = await fetch(`${apiBase}/api/push/cheatsheet`, { method: 'POST' });
      const data = await res.json();
      if (data.sent) {
        setPushStatus(
          data.pinned
            ? 'Cheat Sheet posted & auto-pinned in Discord!'
            : 'Cheat Sheet posted to Discord! (Tap & hold -> Pin Message)'
        );
      } else {
        setPushStatus(data.reason || 'Cheat Sheet post failed.');
      }
    } catch {
      setPushStatus('Cheat Sheet request failed.');
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

      <div className="flex items-center gap-2 sm:gap-3 text-[11px] sm:text-xs font-mono flex-wrap">
        <div className="hidden xl:flex items-center gap-1.5 text-zinc-300">
          <Activity className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
          <span className="text-zinc-500">UPTIME:</span>
          <span className="text-zinc-100 font-semibold">{formatUptime(telemetry?.uptime_seconds)}</span>
        </div>

        <div className="hidden md:flex items-center gap-1.5 text-zinc-300">
          <Cpu className="w-3.5 h-3.5 text-amber-400 shrink-0" />
          <span className="text-amber-300 font-semibold">{telemetry?.agent_mode ?? 'YOLOv8s // RTX 3060 Ti'}</span>
        </div>

        {/* 1-Click HOME / AWAY Master Toggle */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => toggleAwayMode(!isAway, !isAway ? 180 : 0)}
            title={
              isAway
                ? 'AWAY MODE ACTIVE (All 3 cameras 24/7 + full alerts) — Click to restore HOME schedules'
                : 'HOME MODE — Click to arm 24/7 AWAY MODE across all 3 cameras (with 3m exit grace)'
            }
            className={`px-2.5 py-1 rounded text-[10px] font-bold border flex items-center gap-1 cursor-pointer transition-all ${
              isAway
                ? 'bg-amber-600 hover:bg-amber-500 text-white border-amber-300 shadow-[0_0_12px_rgba(245,158,11,0.5)]'
                : 'bg-zinc-900 hover:bg-zinc-800 text-emerald-300 border-zinc-700'
            }`}
          >
            {isAway ? <Plane className="w-3 h-3" /> : <Home className="w-3 h-3" />}
            <span>
              {isAway
                ? awayGraceSec > 0
                  ? `AWAY (${formatMmSs(awayGraceSec)})`
                  : 'AWAY · 24/7'
                : 'HOME'}
            </span>
          </button>

          {isAway && awayGraceSec > 0 && (
            <button
              onClick={() => toggleAwayMode(true, 0)}
              title="Skip 3-minute exit countdown and arm alerts immediately"
              className="px-1.5 py-1 rounded text-[9px] font-bold bg-red-600 hover:bg-red-500 text-white border border-red-400 cursor-pointer"
            >
              ARM NOW
            </button>
          )}
        </div>

        {muteRemSec > 0 && (
          <button
            onClick={() => toggleMute(0)}
            title="Push alerts are snoozed — Click to resume immediately"
            className="px-2 py-1 rounded text-[10px] font-bold bg-amber-950/90 text-amber-300 border border-amber-500/50 flex items-center gap-1 cursor-pointer"
          >
            <BellOff className="w-3 h-3" />
            <span>MUTED {Math.max(1, Math.ceil(muteRemSec / 60))}m</span>
          </button>
        )}

        {isBotConnected && (
          <span
            title="2-Way Discord Bot Connected (/snap, /away, /mute, /status)"
            className="hidden sm:flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold bg-indigo-950/90 text-indigo-300 border border-indigo-500/40"
          >
            <Bot className="w-3 h-3" />
            <span>BOT</span>
          </span>
        )}

        <button
          onClick={() => {
            const next = !showPushModal;
            setShowPushModal(next);
            setPushStatus(null);
            if (next) loadSettings();
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
        <div className="fixed top-12 right-3 sm:right-6 z-50 bg-zinc-950 border border-emerald-500/50 rounded-lg p-4 w-88 max-w-[calc(100vw-1.5rem)] shadow-2xl font-mono text-xs space-y-3">
          <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
            <span className="font-bold text-emerald-400 uppercase">📲 DISCORD & PUSH CONTROLS</span>
            <button onClick={() => setShowPushModal(false)} className="text-zinc-400 hover:text-white cursor-pointer">
              <X className="w-4 h-4" />
            </button>
          </div>

          <div className="space-y-2.5">
            <div>
              <label className="text-[10px] text-zinc-400 block mb-1">DISCORD WEBHOOK URL (OUTBOUND ALERTS):</label>
              <input
                type="password"
                value={discordWebhook}
                onChange={(e) => setDiscordWebhook(e.target.value)}
                placeholder="https://discord.com/api/webhooks/..."
                className="w-full bg-zinc-900 border border-zinc-700 rounded px-2 py-1.5 text-xs text-zinc-100"
              />
            </div>

            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="text-[10px] text-indigo-300 font-bold">2-WAY DISCORD BOT TOKEN (/SNAP, /AWAY):</label>
                <span className={`text-[9px] font-bold ${isBotConnected ? 'text-emerald-400' : 'text-zinc-500'}`}>
                  {isBotConnected ? `● LIVE${botUsername ? ` (@${botUsername})` : ''}` : '○ OFFLINE'}
                </span>
              </div>
              <input
                type="password"
                value={discordBotToken}
                onChange={(e) => setDiscordBotToken(e.target.value)}
                placeholder="Paste Bot Token from Discord Developer Portal"
                className="w-full bg-zinc-900 border border-zinc-700 rounded px-2 py-1.5 text-xs text-zinc-100"
              />
              {botError && !isBotConnected && (
                <div className="text-[9px] text-amber-400 mt-1">{botError}</div>
              )}
            </div>

            <div>
              <label className="text-[10px] text-zinc-400 block mb-1">OPTIONAL NTFY.SH TOPIC:</label>
              <input
                type="text"
                value={ntfyTopic}
                onChange={(e) => setNtfyTopic(e.target.value)}
                placeholder="nason_command_97113"
                className="w-full bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-100"
              />
            </div>

            <div className="space-y-1 pt-1 border-t border-zinc-800/80">
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
                <span>⚠️ Tripwire / Indoor Intrusion</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer text-[11px] text-sky-300">
                <input type="checkbox" checked={notifyDelivery} onChange={(e) => setNotifyDelivery(e.target.checked)} />
                <span>📦 Delivery Carrier (Amazon/UPS/FedEx)</span>
              </label>
              <div className="flex items-center justify-between gap-2 pt-1">
                <label className="flex items-center gap-2 cursor-pointer text-[11px] text-emerald-300">
                  <input
                    type="checkbox"
                    checked={dailySitrepEnabled}
                    onChange={(e) => setDailySitrepEnabled(e.target.checked)}
                  />
                  <span>🌅 Silent Daily Sitrep (PT):</span>
                </label>
                <input
                  type="time"
                  value={dailySitrepTime}
                  onChange={(e) => setDailySitrepTime(e.target.value)}
                  className="bg-zinc-900 border border-zinc-700 rounded px-1.5 py-0.5 text-[11px] text-zinc-100"
                />
              </div>
            </div>
          </div>

          <div className="flex items-center gap-1.5 pt-1 border-t border-zinc-800">
            <button
              onClick={() => toggleMute(muteRemSec > 0 ? 0 : 60)}
              className="flex-1 bg-zinc-800 hover:bg-zinc-700 text-amber-300 font-bold py-1.5 rounded text-[10px] flex items-center justify-center gap-1 cursor-pointer"
            >
              <BellOff className="w-3 h-3" />
              {muteRemSec > 0 ? 'UNMUTE' : 'SNOOZE 1H'}
            </button>
            <button
              onClick={sendManualSitrep}
              className="flex-1 bg-indigo-600 hover:bg-indigo-500 text-white font-bold py-1.5 rounded text-[10px] flex items-center justify-center gap-1 cursor-pointer"
            >
              <Camera className="w-3 h-3" />
              SITREP
            </button>
            <button
              onClick={sendCheatsheet}
              title="Post or refresh the Command Reference Cheat Sheet in your Discord channel"
              className="flex-1 bg-teal-700 hover:bg-teal-600 text-white font-bold py-1.5 rounded text-[10px] flex items-center justify-center gap-1 cursor-pointer"
            >
              <BookOpen className="w-3 h-3" />
              CHEAT SHEET
            </button>
          </div>

          <div className="flex gap-2 pt-1">
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
