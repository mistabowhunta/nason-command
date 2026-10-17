import { useEffect, useRef, useState } from 'react';

export function useWhepClient(whepUrl: string) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [streamState, setStreamState] = useState<'CONNECTING' | 'LIVE' | 'ERROR'>('CONNECTING');

  useEffect(() => {
    const pc = new RTCPeerConnection({ bundlePolicy: 'max-bundle' });

    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.addTransceiver('audio', { direction: 'recvonly' });

    pc.ontrack = (event) => {
      if (videoRef.current && event.streams[0]) {
        videoRef.current.srcObject = event.streams[0];
        setStreamState('LIVE');
      }
    };

    pc.onconnectionstatechange = () => {
      if (pc.connectionState === 'failed' || pc.connectionState === 'disconnected') {
        setStreamState('ERROR');
      }
    };

    const startWhep = async () => {
      try {
        setStreamState('CONNECTING');
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        const res = await fetch(whepUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/sdp' },
          body: offer.sdp,
        });

        if (!res.ok) throw new Error(`WHEP HTTP ${res.status}`);

        const answerSdp = await res.text();
        await pc.setRemoteDescription(new RTCSessionDescription({ type: 'answer', sdp: answerSdp }));
      } catch (err) {
        console.error('WHEP signaling failed:', err);
        setStreamState('ERROR');
      }
    };

    startWhep();

    return () => pc.close();
  }, [whepUrl]);

  return { videoRef, streamState };
}