import { useState, useRef, useCallback } from "react";
import {
  SAMPLE_RATE,
  base64ToPcm16,
  int16ToFloat32,
  resample,
} from "@/lib/audio/audio-utils";

interface UseAudioOutputReturn {
  playAudio: (base64Pcm: string) => Promise<void>;
  stopPlayback: () => void;
  isPlaying: boolean;
  analyserNode: AnalyserNode | null;
}

export function useAudioOutput(): UseAudioOutputReturn {
  const [isPlaying, setIsPlaying] = useState(false);
  const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const initPromiseRef = useRef<Promise<void> | null>(null);

  const ensureInitialized = useCallback(() => {
    if (initPromiseRef.current) return initPromiseRef.current;

    initPromiseRef.current = (async () => {
      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;

      await audioContext.audioWorklet.addModule("/pcm-playback-worklet.js");

      const workletNode = new AudioWorkletNode(
        audioContext,
        "pcm-playback-processor"
      );
      workletNodeRef.current = workletNode;

      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;

      workletNode.connect(analyser);
      analyser.connect(audioContext.destination);

      setAnalyserNode(analyser);
    })();

    return initPromiseRef.current;
  }, []);

  const playAudio = useCallback(
    async (base64Pcm: string) => {
      await ensureInitialized();

      const audioContext = audioContextRef.current!;
      const workletNode = workletNodeRef.current!;

      if (audioContext.state === "suspended") {
        await audioContext.resume();
      }

      const int16 = base64ToPcm16(base64Pcm);
      const float32 = int16ToFloat32(int16);
      const resampled = resample(float32, SAMPLE_RATE, audioContext.sampleRate);

      workletNode.port.postMessage({ samples: resampled });
      setIsPlaying(true);
    },
    [ensureInitialized]
  );

  const stopPlayback = useCallback(() => {
    if (workletNodeRef.current) {
      workletNodeRef.current.port.postMessage({ type: "clear" });
    }
    setIsPlaying(false);
  }, []);

  return { playAudio, stopPlayback, isPlaying, analyserNode };
}
