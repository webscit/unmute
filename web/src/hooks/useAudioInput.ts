import { useState, useRef, useCallback, useEffect } from "react";
import {
  SAMPLE_RATE,
  float32ToInt16,
  pcm16ToBase64,
  resample,
} from "@/lib/audio/audio-utils";

interface UseAudioInputOptions {
  onAudioChunk: (base64: string) => void;
}

interface UseAudioInputReturn {
  startCapture: () => Promise<void>;
  stopCapture: () => void;
  isCapturing: boolean;
  isMuted: boolean;
  setMuted: (muted: boolean) => void;
  analyserNode: AnalyserNode | null;
}

export function useAudioInput({
  onAudioChunk,
}: UseAudioInputOptions): UseAudioInputReturn {
  const [isCapturing, setIsCapturing] = useState(false);
  const [isMuted, setIsMutedState] = useState(false);
  const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const isMutedRef = useRef(false);
  const onAudioChunkRef = useRef(onAudioChunk);

  useEffect(() => {
    onAudioChunkRef.current = onAudioChunk;
  }, [onAudioChunk]);

  const setMuted = useCallback((muted: boolean) => {
    isMutedRef.current = muted;
    setIsMutedState(muted);
  }, []);

  const startCapture = useCallback(async () => {
    if (audioContextRef.current) return;

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: true,
      video: false,
    });
    streamRef.current = stream;

    const audioContext = new AudioContext();
    audioContextRef.current = audioContext;

    await audioContext.audioWorklet.addModule("/pcm-capture-worklet.js");

    const source = audioContext.createMediaStreamSource(stream);
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;

    const workletNode = new AudioWorkletNode(
      audioContext,
      "pcm-capture-processor"
    );
    workletNodeRef.current = workletNode;

    workletNode.port.onmessage = (event: MessageEvent<{ samples: Float32Array }>) => {
      if (isMutedRef.current) return;
      const { samples } = event.data;
      const resampled = resample(samples, audioContext.sampleRate, SAMPLE_RATE);
      const int16 = float32ToInt16(resampled);
      const base64 = pcm16ToBase64(int16);
      onAudioChunkRef.current(base64);
    };

    source.connect(analyser);
    analyser.connect(workletNode);

    setAnalyserNode(analyser);
    setIsCapturing(true);
  }, []);

  const stopCapture = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect();
      workletNodeRef.current.port.close();
      workletNodeRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    setAnalyserNode(null);
    setIsCapturing(false);
  }, []);

  return { startCapture, stopCapture, isCapturing, isMuted, setMuted, analyserNode };
}
