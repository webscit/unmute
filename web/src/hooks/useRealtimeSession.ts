import { useState, useRef, useCallback, useEffect } from "react";
import type { OpenAIRealtimeWebSocket } from "openai/realtime/websocket";
import type { ChatMessageData } from "@/components/chat/ChatMessage";
import type { SessionConfig } from "@/lib/types/events";
import type { SettingsValues } from "@/components/settings/SettingsDialog";
import {
  createRealtimeClient,
  buildSessionUpdate,
  onUnmuteEvent,
} from "@/lib/realtime-client";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export type ConnectionStatus = "disconnected" | "connecting" | "connected";

export interface DebugEvent {
  timestamp: Date;
  direction: "client" | "server";
  type: string;
  payload: unknown;
}

const MAX_DEBUG_EVENTS = 500;

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export interface UseRealtimeSessionReturn {
  connect: (settings: SettingsValues) => Promise<void>;
  disconnect: () => void;
  connectionStatus: ConnectionStatus;
  messages: ChatMessageData[];
  sendAudio: (base64: string) => void;
  commitAudioBuffer: () => void;
  debugEvents: DebugEvent[];
  clearDebugEvents: () => void;
  rtClient: OpenAIRealtimeWebSocket | null;
  isSpeechActive: boolean;
  isResponding: boolean;
  onAudioDelta: React.MutableRefObject<((base64: string) => void) | null>;
}

export function useRealtimeSession(): UseRealtimeSessionReturn {
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("disconnected");
  const [messages, setMessages] = useState<ChatMessageData[]>([]);
  const [debugEvents, setDebugEvents] = useState<DebugEvent[]>([]);
  const [isSpeechActive, setIsSpeechActive] = useState(false);
  const [isResponding, setIsResponding] = useState(false);

  const rtRef = useRef<OpenAIRealtimeWebSocket | null>(null);
  const onAudioDelta = useRef<((base64: string) => void) | null>(null);
  const cleanupUnmuteRef = useRef<(() => void) | null>(null);

  // Track in-flight response timing per response_id
  const responseTimingRef = useRef<
    Map<string, { firstTokenAt?: number; createdAt: number }>
  >(new Map());

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  const clearDebugEvents = useCallback(() => {
    setDebugEvents([]);
  }, []);

  const pushDebug = useCallback(
    (direction: "client" | "server", type: string, payload: unknown) => {
      setDebugEvents((prev) => {
        const next = [
          ...prev,
          { timestamp: new Date(), direction, type, payload },
        ];
        if (next.length > MAX_DEBUG_EVENTS) return next.slice(-MAX_DEBUG_EVENTS);
        return next;
      });
    },
    [],
  );

  // ---------------------------------------------------------------------------
  // connect
  // ---------------------------------------------------------------------------

  const connect = useCallback(
    async (settings: SettingsValues) => {
      if (rtRef.current) return;
      setConnectionStatus("connecting");

      try {
        const rt = await createRealtimeClient({
          apiKey: "placeholder",
          baseURL: settings.backendUrl || undefined,
          model: "gpt-4o-realtime-preview",
        });
        rtRef.current = rt;

        // --- Wire up SDK event listeners ---

        // Catch-all: log every server event to debug
        rt.on("event", (event) => {
          pushDebug("server", event.type, event);
        });

        // Session created → send session.update with config from settings
        rt.on("session.created", () => {
          const sessionCfg: SessionConfig = {
            modalities: ["text", "audio"],
            voice: settings.voice || "alloy",
            input_audio_format: "pcm16",
            output_audio_format: "pcm16",
            input_audio_transcription: { model: "whisper-1" },
            turn_detection: { type: "server_vad" },
          };
          if (settings.customInstructions) {
            sessionCfg.instructions = settings.customInstructions;
          }

          const updatePayload = buildSessionUpdate(sessionCfg);
          rt.send({
            type: "session.update",
            session: updatePayload,
          } as unknown as Parameters<typeof rt.send>[0]);
          pushDebug("client", "session.update", updatePayload);
        });

        rt.on("session.updated", () => {
          setConnectionStatus("connected");
        });

        // --- Transcription deltas → user messages ---
        const userTranscriptBuffers = new Map<string, string>();

        rt.on(
          "conversation.item.input_audio_transcription.delta" as Parameters<typeof rt.on>[0],
          (event: { item_id: string; delta: string }) => {
            const prev = userTranscriptBuffers.get(event.item_id) ?? "";
            userTranscriptBuffers.set(event.item_id, prev + event.delta);

            setMessages((msgs) => {
              const idx = msgs.findIndex((m) => m.id === event.item_id);
              if (idx >= 0) {
                const updated = [...msgs];
                updated[idx] = {
                  ...updated[idx],
                  content: userTranscriptBuffers.get(event.item_id)!,
                  isStreaming: true,
                };
                return updated;
              }
              return [
                ...msgs,
                {
                  id: event.item_id,
                  role: "user" as const,
                  content: event.delta,
                  timestamp: new Date(),
                  isStreaming: true,
                },
              ];
            });
          },
        );

        rt.on(
          "conversation.item.input_audio_transcription.completed" as Parameters<typeof rt.on>[0],
          (event: { item_id: string; transcript: string }) => {
            userTranscriptBuffers.delete(event.item_id);
            setMessages((msgs) => {
              const idx = msgs.findIndex((m) => m.id === event.item_id);
              if (idx >= 0) {
                const updated = [...msgs];
                updated[idx] = {
                  ...updated[idx],
                  content: event.transcript,
                  isStreaming: false,
                };
                return updated;
              }
              return msgs;
            });
          },
        );

        // --- Response lifecycle ---
        rt.on("response.created", (event) => {
          setIsResponding(true);
          const respId = event.response?.id;
          if (respId) {
            responseTimingRef.current.set(respId, { createdAt: Date.now() });
          }
        });

        rt.on("response.done", (event) => {
          setIsResponding(false);
          const respId = event.response?.id;
          if (respId) {
            const timing = responseTimingRef.current.get(respId);
            if (timing) {
              const responseDurationMs = Date.now() - timing.createdAt;
              const thinkingTimeMs = timing.firstTokenAt
                ? timing.firstTokenAt - timing.createdAt
                : undefined;

              // Patch timing into the assistant message for this response
              setMessages((msgs) =>
                msgs.map((m) =>
                  m.id === respId
                    ? { ...m, isStreaming: false, responseDurationMs, thinkingTimeMs }
                    : m,
                ),
              );
              responseTimingRef.current.delete(respId);
            }
          }
        });

        // --- Text deltas → assistant messages ---
        const assistantTextBuffers = new Map<string, string>();

        rt.on(
          "response.output_text.delta" as Parameters<typeof rt.on>[0],
          (event: { response_id: string; delta: string }) => {
            const respId = event.response_id;
            const timing = responseTimingRef.current.get(respId);
            if (timing && !timing.firstTokenAt) {
              timing.firstTokenAt = Date.now();
            }

            const prev = assistantTextBuffers.get(respId) ?? "";
            assistantTextBuffers.set(respId, prev + event.delta);

            setMessages((msgs) => {
              const idx = msgs.findIndex((m) => m.id === respId);
              if (idx >= 0) {
                const updated = [...msgs];
                updated[idx] = {
                  ...updated[idx],
                  content: assistantTextBuffers.get(respId)!,
                  isStreaming: true,
                };
                return updated;
              }
              return [
                ...msgs,
                {
                  id: respId,
                  role: "assistant" as const,
                  content: event.delta,
                  timestamp: new Date(),
                  isStreaming: true,
                },
              ];
            });
          },
        );

        rt.on(
          "response.output_text.done" as Parameters<typeof rt.on>[0],
          (event: { response_id: string; text: string }) => {
            assistantTextBuffers.delete(event.response_id);
            setMessages((msgs) =>
              msgs.map((m) =>
                m.id === event.response_id
                  ? { ...m, content: event.text, isStreaming: false }
                  : m,
              ),
            );
          },
        );

        // --- Audio deltas → callback ---
        rt.on(
          "response.output_audio.delta" as Parameters<typeof rt.on>[0],
          (event: { response_id: string; delta: string }) => {
            const timing = responseTimingRef.current.get(event.response_id);
            if (timing && !timing.firstTokenAt) {
              timing.firstTokenAt = Date.now();
            }
            onAudioDelta.current?.(event.delta);
          },
        );

        // --- Audio transcript deltas → assistant messages (for audio-only responses) ---
        const assistantAudioTranscriptBuffers = new Map<string, string>();

        rt.on(
          "response.output_audio_transcript.delta" as Parameters<typeof rt.on>[0],
          (event: { response_id: string; delta: string }) => {
            const respId = event.response_id;
            const prev = assistantAudioTranscriptBuffers.get(respId) ?? "";
            assistantAudioTranscriptBuffers.set(respId, prev + event.delta);

            setMessages((msgs) => {
              const idx = msgs.findIndex((m) => m.id === respId);
              if (idx >= 0) {
                const updated = [...msgs];
                updated[idx] = {
                  ...updated[idx],
                  content: assistantAudioTranscriptBuffers.get(respId)!,
                  isStreaming: true,
                };
                return updated;
              }
              return [
                ...msgs,
                {
                  id: respId,
                  role: "assistant" as const,
                  content: event.delta,
                  timestamp: new Date(),
                  isStreaming: true,
                },
              ];
            });
          },
        );

        rt.on(
          "response.output_audio_transcript.done" as Parameters<typeof rt.on>[0],
          (event: { response_id: string; transcript: string }) => {
            assistantAudioTranscriptBuffers.delete(event.response_id);
            setMessages((msgs) =>
              msgs.map((m) =>
                m.id === event.response_id
                  ? { ...m, content: event.transcript, isStreaming: false }
                  : m,
              ),
            );
          },
        );

        // --- Speech activity (VAD) ---
        rt.on("input_audio_buffer.speech_started", () => {
          setIsSpeechActive(true);
        });

        rt.on("input_audio_buffer.speech_stopped", () => {
          setIsSpeechActive(false);
        });

        // --- Error handling ---
        rt.on("error", (error) => {
          console.error("[realtime] error:", error);
        });

        // --- Unmute extension events (via raw socket) ---
        cleanupUnmuteRef.current = onUnmuteEvent(rt, (event) => {
          pushDebug("server", event.type, event);
        });
      } catch (err) {
        console.error("[realtime] connection failed:", err);
        setConnectionStatus("disconnected");
        rtRef.current = null;
      }
    },
    [pushDebug],
  );

  // ---------------------------------------------------------------------------
  // disconnect
  // ---------------------------------------------------------------------------

  const disconnect = useCallback(() => {
    cleanupUnmuteRef.current?.();
    cleanupUnmuteRef.current = null;

    if (rtRef.current) {
      rtRef.current.close();
      rtRef.current = null;
    }
    setConnectionStatus("disconnected");
    setIsSpeechActive(false);
    setIsResponding(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cleanupUnmuteRef.current?.();
      if (rtRef.current) {
        rtRef.current.close();
        rtRef.current = null;
      }
    };
  }, []);

  // ---------------------------------------------------------------------------
  // sendAudio / commitAudioBuffer
  // ---------------------------------------------------------------------------

  const sendAudio = useCallback((base64: string) => {
    const rt = rtRef.current;
    if (!rt) return;
    rt.send({
      type: "input_audio_buffer.append",
      audio: base64,
    } as Parameters<typeof rt.send>[0]);
  }, []);

  const commitAudioBuffer = useCallback(() => {
    const rt = rtRef.current;
    if (!rt) return;
    rt.send({
      type: "input_audio_buffer.commit",
    } as Parameters<typeof rt.send>[0]);
    pushDebug("client", "input_audio_buffer.commit", {});
  }, [pushDebug]);

  // ---------------------------------------------------------------------------
  // Return
  // ---------------------------------------------------------------------------

  return {
    connect,
    disconnect,
    connectionStatus,
    messages,
    sendAudio,
    commitAudioBuffer,
    debugEvents,
    clearDebugEvents,
    rtClient: rtRef.current,
    isSpeechActive,
    isResponding,
    onAudioDelta,
  };
}
