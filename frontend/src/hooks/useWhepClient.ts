import { useEffect, useRef, useState } from 'react';

export type StreamState = 'CONNECTING' | 'LIVE' | 'OFFLINE';

export function useWhepClient(whepUrl: string) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [streamState, setStreamState] = useState<StreamState>('CONNECTING');

  useEffect(() => {
    let pc: RTCPeerConnection | null = null;
    let reconnectTimer: number | null = null;
    let watchdogTimer: number | null = null;
    let unmounted = false;
    let lastVideoTime = -1;
    let stallCount = 0;

    const cleanup = () => {
      if (reconnectTimer) window.clearTimeout(reconnectTimer);
      if (watchdogTimer) window.clearInterval(watchdogTimer);
      if (pc) {
        pc.onconnectionstatechange = null;
        pc.ontrack = null;
        pc.close();
        pc = null;
      }
    };

    const scheduleReconnect = (delayMs = 3000) => {
      if (unmounted) return;
      cleanup();
      setStreamState('CONNECTING');
      reconnectTimer = window.setTimeout(() => {
        if (!unmounted) connectWhep();
      }, delayMs);
    };

    const connectWhep = async () => {
      try {
        setStreamState('CONNECTING');
        pc = new RTCPeerConnection({
          bundlePolicy: 'max-bundle',
        });

        pc.addTransceiver('video', { direction: 'recvonly' });
        pc.addTransceiver('audio', { direction: 'recvonly' });

        pc.ontrack = (event) => {
          if (event.track.kind === 'video' && videoRef.current && event.streams[0]) {
            const vid = videoRef.current;
            vid.srcObject = event.streams[0];
            vid.onloadeddata = () => {
              if (!unmounted) setStreamState('LIVE');
            };
            vid.play().catch(() => {});
          }
        };

        pc.onconnectionstatechange = () => {
          if (!pc || unmounted) return;
          if (
            pc.connectionState === 'failed' ||
            pc.connectionState === 'disconnected' ||
            pc.connectionState === 'closed'
          ) {
            scheduleReconnect(2500);
          }
        };

        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        const res = await fetch(whepUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/sdp' },
          body: offer.sdp,
        });

        if (!res.ok) {
          scheduleReconnect(3500);
          return;
        }

        const answerSdp = await res.text();
        if (pc && !unmounted) {
          await pc.setRemoteDescription(
            new RTCSessionDescription({ type: 'answer', sdp: answerSdp })
          );
        }

        watchdogTimer = window.setInterval(() => {
          const vid = videoRef.current;
          if (!vid || unmounted) return;
          if (vid.videoWidth > 0 && !vid.paused) {
            setStreamState('LIVE');
            if (vid.currentTime === lastVideoTime) {
              stallCount += 1;
              if (stallCount >= 3) {
                stallCount = 0;
                scheduleReconnect(1500);
              }
            } else {
              lastVideoTime = vid.currentTime;
              stallCount = 0;
            }
          }
        }, 5500);
      } catch (err) {
        scheduleReconnect(3500);
      }
    };

    connectWhep();

    return () => {
      unmounted = true;
      cleanup();
    };
  }, [whepUrl]);

  return { videoRef, streamState };
}
