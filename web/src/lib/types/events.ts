/**
 * TypeScript event types mirroring backend events from
 * unmute/openai_realtime_api_events.py
 *
 * See OpenAI's docs: https://platform.openai.com/docs/api-reference/realtime
 */

// ---------------------------------------------------------------------------
// Supporting types
// ---------------------------------------------------------------------------

export interface ContentPart {
  type: string;
}

export interface TextContentPart extends ContentPart {
  type: "text";
  text: string;
}

export interface AudioContentPart extends ContentPart {
  type: "audio";
  audio?: string; // Base64-encoded audio
  transcript?: string;
}

export interface InputTextContentPart extends ContentPart {
  type: "input_text";
  text: string;
}

export interface InputAudioContentPart extends ContentPart {
  type: "input_audio";
  audio?: string; // Base64-encoded audio
  transcript?: string;
}

export type AnyContentPart =
  | TextContentPart
  | AudioContentPart
  | InputTextContentPart
  | InputAudioContentPart
  | ContentPart;

export interface ToolDefinition {
  type: "function";
  name: string;
  description?: string;
  parameters?: Record<string, unknown>;
}

export interface TurnDetectionConfig {
  type?: string;
  threshold?: number;
  prefix_padding_ms?: number;
  silence_duration_ms?: number;
  [key: string]: unknown;
}

export interface InputAudioTranscriptionConfig {
  model?: string;
  [key: string]: unknown;
}

/** Session configuration (maps to Python `Session` model). */
export interface SessionConfig {
  id?: string;
  object?: "realtime.session";
  type?: "realtime" | "transcription";
  model?: string;
  modalities?: string[];
  instructions?: string | Record<string, unknown>;
  voice?: string;
  input_audio_format?: string;
  output_audio_format?: string;
  input_audio_transcription?: InputAudioTranscriptionConfig;
  turn_detection?: TurnDetectionConfig;
  tools?: ToolDefinition[];
  tool_choice?: string;
  temperature?: number;
  max_response_output_tokens?: number | "inf";
  /** Unmute extension */
  allow_recording?: boolean;
}

export interface UsageStats {
  total_tokens: number;
  input_tokens: number;
  output_tokens: number;
}

export interface ErrorDetails {
  type: string;
  code?: string;
  message: string;
  param?: string;
  /** Unmute extension */
  details?: unknown;
}

/** Conversation or response item (maps to Python `Item` model). */
export interface ItemObject {
  id: string;
  object: "realtime.item";
  type: "message" | "function_call" | "function_call_output";
  status?: "in_progress" | "completed" | "incomplete";
  role?: "system" | "user" | "assistant";
  content?: Record<string, unknown>[];
  call_id?: string;
  name?: string;
  arguments?: string;
  output?: string;
}

/** Response object (maps to Python `Response` model). */
export interface ResponseObject {
  id?: string;
  object: "realtime.response";
  status: "in_progress" | "completed" | "cancelled" | "failed" | "incomplete";
  status_details?: Record<string, unknown>;
  output: ItemObject[];
  usage?: UsageStats;
  /** Unmute extension */
  voice?: string;
  /** Unmute extension */
  chat_history?: Record<string, unknown>[];
}

// ---------------------------------------------------------------------------
// Base event
// ---------------------------------------------------------------------------

export interface BaseEvent {
  type: string;
  event_id: string;
}

// ---------------------------------------------------------------------------
// Client events (sent from client to server)
// ---------------------------------------------------------------------------

export interface SessionUpdateEvent extends BaseEvent {
  type: "session.update";
  session: SessionConfig | Record<string, unknown>;
}

export interface InputAudioBufferAppendEvent extends BaseEvent {
  type: "input_audio_buffer.append";
  /** Base64-encoded Opus data */
  audio: string;
}

export interface InputAudioBufferCommitEvent extends BaseEvent {
  type: "input_audio_buffer.commit";
}

export interface InputAudioBufferClearEvent extends BaseEvent {
  type: "input_audio_buffer.clear";
}

export interface ResponseCreateEvent extends BaseEvent {
  type: "response.create";
  response?: Record<string, unknown>;
}

export interface ResponseCancelEvent extends BaseEvent {
  type: "response.cancel";
}

export interface ConversationItemCreateEvent extends BaseEvent {
  type: "conversation.item.create";
  item: Record<string, unknown>;
  previous_item_id?: string;
}

export interface ConversationItemDeleteEvent extends BaseEvent {
  type: "conversation.item.delete";
  item_id: string;
}

export interface ConversationItemTruncateEvent extends BaseEvent {
  type: "conversation.item.truncate";
  item_id: string;
  content_index: number;
  audio_end_ms: number;
}

export type ClientEvent =
  | SessionUpdateEvent
  | InputAudioBufferAppendEvent
  | InputAudioBufferCommitEvent
  | InputAudioBufferClearEvent
  | ResponseCreateEvent
  | ResponseCancelEvent
  | ConversationItemCreateEvent
  | ConversationItemDeleteEvent
  | ConversationItemTruncateEvent;

// ---------------------------------------------------------------------------
// Server events (sent from server to client)
// ---------------------------------------------------------------------------

export interface SessionCreatedEvent extends BaseEvent {
  type: "session.created";
  session: SessionConfig;
}

export interface SessionUpdatedEvent extends BaseEvent {
  type: "session.updated";
  session: SessionConfig;
}

export interface ErrorEvent extends BaseEvent {
  type: "error";
  error: ErrorDetails;
}

export interface ConversationItemAddedEvent extends BaseEvent {
  type: "conversation.item.added";
  item: ItemObject;
  previous_item_id?: string;
}

export interface ConversationItemDoneEvent extends BaseEvent {
  type: "conversation.item.done";
  item: ItemObject;
  previous_item_id?: string;
}

export interface InputAudioBufferCommittedEvent extends BaseEvent {
  type: "input_audio_buffer.committed";
  item_id: string;
  previous_item_id?: string;
}

export interface InputAudioBufferClearedEvent extends BaseEvent {
  type: "input_audio_buffer.cleared";
}

export interface InputAudioBufferSpeechStartedEvent extends BaseEvent {
  type: "input_audio_buffer.speech_started";
  item_id?: string;
  audio_start_ms?: number;
}

export interface InputAudioBufferSpeechStoppedEvent extends BaseEvent {
  type: "input_audio_buffer.speech_stopped";
  item_id?: string;
  audio_end_ms?: number;
}

export interface ResponseCreatedEvent extends BaseEvent {
  type: "response.created";
  response: ResponseObject;
}

export interface ResponseDoneEvent extends BaseEvent {
  type: "response.done";
  response: ResponseObject;
}

export interface ResponseOutputItemAddedEvent extends BaseEvent {
  type: "response.output_item.added";
  response_id: string;
  output_index: number;
  item: ItemObject;
}

export interface ResponseOutputItemDoneEvent extends BaseEvent {
  type: "response.output_item.done";
  response_id: string;
  output_index: number;
  item: ItemObject;
}

/** Incremental text response chunk (response.output_text.delta). */
export interface ResponseTextDeltaEvent extends BaseEvent {
  type: "response.output_text.delta";
  delta: string;
  response_id?: string;
  item_id?: string;
  output_index?: number;
  content_index?: number;
}

export interface ResponseTextDoneEvent extends BaseEvent {
  type: "response.output_text.done";
  text: string;
  response_id?: string;
  item_id?: string;
  output_index?: number;
  content_index?: number;
}

/** Incremental audio response chunk (response.output_audio.delta). */
export interface ResponseAudioDeltaEvent extends BaseEvent {
  type: "response.output_audio.delta";
  /** Base64-encoded Opus audio data */
  delta: string;
  response_id?: string;
  item_id?: string;
  output_index?: number;
  content_index?: number;
}

export interface ResponseAudioDoneEvent extends BaseEvent {
  type: "response.output_audio.done";
  response_id?: string;
  item_id?: string;
  output_index?: number;
  content_index?: number;
}

export interface ResponseAudioTranscriptDeltaEvent extends BaseEvent {
  type: "response.output_audio_transcript.delta";
  delta: string;
  response_id?: string;
  item_id?: string;
  output_index?: number;
  content_index?: number;
}

export interface ResponseAudioTranscriptDoneEvent extends BaseEvent {
  type: "response.output_audio_transcript.done";
  transcript: string;
  response_id?: string;
  item_id?: string;
  output_index?: number;
  content_index?: number;
}

/** Incremental transcription of user audio input. */
export interface InputAudioTranscriptionDeltaEvent extends BaseEvent {
  type: "conversation.item.input_audio_transcription.delta";
  item_id: string;
  content_index: number;
  delta: string;
  /** Unmute extension */
  start_time?: number;
}

export interface InputAudioTranscriptionCompletedEvent extends BaseEvent {
  type: "conversation.item.input_audio_transcription.completed";
  item_id: string;
  content_index: number;
  transcript: string;
  usage?: UsageStats;
}

export interface ResponseFunctionCallArgumentsDeltaEvent extends BaseEvent {
  type: "response.function_call_arguments.delta";
  response_id: string;
  item_id: string;
  output_index: number;
  call_id: string;
  delta: string;
}

export interface ResponseFunctionCallArgumentsDoneEvent extends BaseEvent {
  type: "response.function_call_arguments.done";
  response_id: string;
  item_id: string;
  output_index: number;
  call_id: string;
  arguments: string;
}

// ---------------------------------------------------------------------------
// Unmute extension events
// ---------------------------------------------------------------------------

export interface UnmuteAdditionalOutputsEvent extends BaseEvent {
  type: "unmute.additional_outputs";
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  args: unknown;
}

export interface UnmuteInterruptedByVADEvent extends BaseEvent {
  type: "unmute.interrupted_by_vad";
}

// ---------------------------------------------------------------------------
// Union types
// ---------------------------------------------------------------------------

export type ServerEvent =
  // Session events
  | SessionCreatedEvent
  | SessionUpdatedEvent
  // Error events
  | ErrorEvent
  // Conversation events
  | ConversationItemAddedEvent
  | ConversationItemDoneEvent
  // Input audio buffer events
  | InputAudioBufferCommittedEvent
  | InputAudioBufferClearedEvent
  | InputAudioBufferSpeechStartedEvent
  | InputAudioBufferSpeechStoppedEvent
  // Transcription events
  | InputAudioTranscriptionDeltaEvent
  | InputAudioTranscriptionCompletedEvent
  // Response events
  | ResponseCreatedEvent
  | ResponseDoneEvent
  | ResponseOutputItemAddedEvent
  | ResponseOutputItemDoneEvent
  // Response text events
  | ResponseTextDeltaEvent
  | ResponseTextDoneEvent
  // Response audio events
  | ResponseAudioDeltaEvent
  | ResponseAudioDoneEvent
  | ResponseAudioTranscriptDeltaEvent
  | ResponseAudioTranscriptDoneEvent
  // Function call events
  | ResponseFunctionCallArgumentsDeltaEvent
  | ResponseFunctionCallArgumentsDoneEvent
  // Unmute extensions
  | UnmuteAdditionalOutputsEvent
  | UnmuteInterruptedByVADEvent;

export type RealtimeEvent = ClientEvent | ServerEvent;
