import OpenAI from "openai";
import { OpenAIRealtimeWebSocket } from "openai/realtime/websocket";
import type { SessionConfig } from "@/lib/types/events";

export interface RealtimeClientConfig {
  /** OpenAI API key (or placeholder for proxied setups). */
  apiKey: string;
  /** Base URL for the OpenAI-compatible API. */
  baseURL?: string;
  /** Model identifier sent in the WebSocket handshake. */
  model?: string;
}

const DEFAULT_MODEL = "gpt-4o-realtime-preview";

/**
 * Create an {@link OpenAIRealtimeWebSocket} connected to the given (or proxied)
 * backend. Returns a promise that resolves once the socket is open.
 */
export async function createRealtimeClient(
  config: RealtimeClientConfig,
): Promise<OpenAIRealtimeWebSocket> {
  const client = new OpenAI({
    apiKey: config.apiKey,
    baseURL: config.baseURL,
    dangerouslyAllowBrowser: true,
  });

  const rt = await OpenAIRealtimeWebSocket.create(client, {
    model: config.model ?? DEFAULT_MODEL,
    dangerouslyAllowBrowser: true,
  });

  return rt;
}

/**
 * Build the `session.update` payload from our {@link SessionConfig}.
 */
export function buildSessionUpdate(cfg: SessionConfig): Record<string, unknown> {
  const session: Record<string, unknown> = {};

  if (cfg.modalities) session.modalities = cfg.modalities;
  if (cfg.instructions) session.instructions = cfg.instructions;
  if (cfg.voice) session.voice = cfg.voice;
  if (cfg.input_audio_format) session.input_audio_format = cfg.input_audio_format;
  if (cfg.output_audio_format) session.output_audio_format = cfg.output_audio_format;
  if (cfg.input_audio_transcription)
    session.input_audio_transcription = cfg.input_audio_transcription;
  if (cfg.turn_detection) session.turn_detection = cfg.turn_detection;
  if (cfg.tools) session.tools = cfg.tools;
  if (cfg.tool_choice) session.tool_choice = cfg.tool_choice;
  if (cfg.temperature !== undefined) session.temperature = cfg.temperature;
  if (cfg.max_response_output_tokens !== undefined)
    session.max_response_output_tokens = cfg.max_response_output_tokens;

  return session;
}

/**
 * Listen for unmute extension events (`unmute.*`) on the raw underlying socket.
 * The SDK emitter only dispatches known OpenAI event types, so we attach a
 * listener directly on the WebSocket `message` event.
 */
export function onUnmuteEvent(
  rt: OpenAIRealtimeWebSocket,
  callback: (event: { type: string; [key: string]: unknown }) => void,
): () => void {
  const handler = (ev: MessageEvent) => {
    try {
      const data = JSON.parse(typeof ev.data === "string" ? ev.data : "{}");
      if (typeof data.type === "string" && data.type.startsWith("unmute.")) {
        callback(data);
      }
    } catch {
      // ignore non-JSON frames
    }
  };

  rt.socket.addEventListener("message", handler);
  return () => rt.socket.removeEventListener("message", handler);
}
