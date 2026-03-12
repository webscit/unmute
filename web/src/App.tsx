import { useState, useCallback, useEffect, useRef } from "react";
import { useRealtimeSession } from "@/hooks/useRealtimeSession";
import { useAudioInput } from "@/hooks/useAudioInput";
import { useAudioOutput } from "@/hooks/useAudioOutput";
import { useSessionHistory } from "@/hooks/useSessionHistory";
import { onUnmuteEvent } from "@/lib/realtime-client";
import { AppLayout } from "@/components/layout/AppLayout";
import { Header } from "@/components/layout/Header";
import { SessionSidebar } from "@/components/layout/SessionSidebar";
import { ChatView } from "@/components/chat/ChatView";
import { AudioControls } from "@/components/audio/AudioControls";
import {
  SettingsDialog,
  type SettingsValues,
} from "@/components/settings/SettingsDialog";
import type { ChatMessageRecord } from "@/lib/db/chat-db";

const DEFAULT_SETTINGS: SettingsValues = {
  voice: "alloy",
  customInstructions: "",
  backendUrl: "/api",
  apiKey: "",
  model: "gpt-4o-realtime-preview",
};

function App() {
  // ---- State ----
  const [debugMode, setDebugMode] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState<SettingsValues>(DEFAULT_SETTINGS);

  // ---- Hooks ----
  const session = useRealtimeSession();
  const audioOutput = useAudioOutput();
  const audioInput = useAudioInput({ onAudioChunk: session.sendAudio });
  const history = useSessionHistory();

  // ---- Wire audio output to session's onAudioDelta ----
  useEffect(() => {
    session.onAudioDelta.current = audioOutput.playAudio;
    return () => {
      session.onAudioDelta.current = null;
    };
  }, [session.onAudioDelta, audioOutput.playAudio]);

  // ---- Handle VAD interruption (unmute.interrupted_by_vad → stopPlayback) ----
  const stopPlaybackRef = useRef(audioOutput.stopPlayback);
  stopPlaybackRef.current = audioOutput.stopPlayback;

  useEffect(() => {
    const rt = session.rtClient;
    if (!rt) return;

    return onUnmuteEvent(rt, (event) => {
      if (event.type === "unmute.interrupted_by_vad") {
        stopPlaybackRef.current();
      }
    });
  }, [session.rtClient]);

  // ---- Save session when messages change ----
  useEffect(() => {
    if (session.messages.length === 0) return;
    if (!history.currentSessionId) return;

    const messageRecords: ChatMessageRecord[] = session.messages.map((m) => ({
      id: m.id,
      role: m.role,
      content: m.content,
      timestamp: m.timestamp,
      thinkingTimeMs: m.thinkingTimeMs,
      responseDurationMs: m.responseDurationMs,
    }));

    history.saveCurrentSession(messageRecords, settings);
  }, [session.messages, history.currentSessionId, history.saveCurrentSession, settings]);

  // ---- Connect flow: createNewSession → startCapture → session.connect ----
  const handleConnect = useCallback(async () => {
    history.createNewSession();
    await audioInput.startCapture();
    await session.connect(settings);
  }, [history, audioInput, session, settings]);

  // ---- Disconnect flow: session.disconnect → stopCapture → stopPlayback ----
  const handleDisconnect = useCallback(() => {
    session.disconnect();
    audioInput.stopCapture();
    audioOutput.stopPlayback();
  }, [session, audioInput, audioOutput]);

  // ---- Session sidebar callbacks ----
  const handleNewSession = useCallback(() => {
    // Disconnect current session if connected, then start fresh
    if (session.connectionStatus !== "disconnected") {
      handleDisconnect();
    }
    history.createNewSession();
  }, [session.connectionStatus, handleDisconnect, history]);

  const handleLoadSession = useCallback(
    async (id: string) => {
      if (session.connectionStatus !== "disconnected") {
        handleDisconnect();
      }
      await history.loadSession(id);
    },
    [session.connectionStatus, handleDisconnect, history],
  );

  const handleDeleteSession = useCallback(
    async (id: string) => {
      await history.deleteSession(id);
    },
    [history],
  );

  const handleToggleMute = useCallback(() => {
    audioInput.setMuted(!audioInput.isMuted);
  }, [audioInput]);

  const isConnected = session.connectionStatus === "connected";

  return (
    <>
    <AppLayout
      sidebar={
        <SessionSidebar
          sessions={history.sessions}
          currentSessionId={history.currentSessionId}
          onNewSession={handleNewSession}
          onLoadSession={handleLoadSession}
          onDeleteSession={handleDeleteSession}
        />
      }
      header={
        <Header
          debugMode={debugMode}
          onDebugModeChange={setDebugMode}
          onSettingsOpen={() => setSettingsOpen(true)}
        />
      }
      chatArea={
        <ChatView
          messages={session.messages}
          isSpeaking={session.isResponding}
          debugMode={debugMode}
          debugEvents={session.debugEvents}
          onClearDebugEvents={session.clearDebugEvents}
        />
      }
      audioControls={
        <AudioControls
          isConnected={isConnected}
          onConnect={handleConnect}
          onDisconnect={handleDisconnect}
          isMuted={audioInput.isMuted}
          onToggleMute={handleToggleMute}
        />
      }
      debugMode={debugMode}
      onDebugModeChange={setDebugMode}
      debugEvents={session.debugEvents}
      onClearDebugEvents={session.clearDebugEvents}
    />

    <SettingsDialog
      open={settingsOpen}
      onOpenChange={setSettingsOpen}
      values={settings}
      onChange={setSettings}
    />
    </>
  );
}

export default App;
