import { useState, useEffect, useRef, useCallback } from "react";
import {
  getAllSessions,
  createSession,
  updateSession,
  deleteSession as dbDeleteSession,
  getSession,
  type ChatSessionRecord,
  type ChatMessageRecord,
  type SessionConfigRecord,
} from "@/lib/db/chat-db";

export interface SessionSummary {
  id: string;
  title: string;
  updatedAt: Date;
  messageCount: number;
}

export interface UseSessionHistoryReturn {
  sessions: SessionSummary[];
  currentSessionId: string | null;
  isLoading: boolean;
  createNewSession: () => string;
  loadSession: (id: string) => Promise<ChatSessionRecord | undefined>;
  deleteSession: (id: string) => Promise<void>;
  saveCurrentSession: (
    messages: ChatMessageRecord[],
    config: SessionConfigRecord
  ) => void;
}

export function useSessionHistory(): UseSessionHistoryReturn {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    getAllSessions()
      .then((all) => {
        const summaries: SessionSummary[] = all
          .map((s) => ({
            id: s.id,
            title: s.title,
            updatedAt: s.updatedAt,
            messageCount: s.messages.length,
          }))
          .sort((a, b) => b.updatedAt.getTime() - a.updatedAt.getTime());
        setSessions(summaries);
      })
      .finally(() => setIsLoading(false));
  }, []);

  const createNewSession = useCallback((): string => {
    const id = crypto.randomUUID();
    const now = new Date();
    const session: ChatSessionRecord = {
      id,
      title: "New Chat",
      createdAt: now,
      updatedAt: now,
      messages: [],
      config: { voice: "", customInstructions: "", backendUrl: "" },
    };
    createSession(session).then(() => {
      setSessions((prev) => [
        { id, title: "New Chat", updatedAt: now, messageCount: 0 },
        ...prev,
      ]);
    });
    setCurrentSessionId(id);
    return id;
  }, []);

  const loadSession = useCallback(
    async (id: string): Promise<ChatSessionRecord | undefined> => {
      const session = await getSession(id);
      setCurrentSessionId(id);
      return session;
    },
    []
  );

  const deleteSession = useCallback(
    async (id: string): Promise<void> => {
      await dbDeleteSession(id);
      setSessions((prev) => prev.filter((s) => s.id !== id));
      if (currentSessionId === id) {
        setCurrentSessionId(null);
      }
    },
    [currentSessionId]
  );

  const saveCurrentSession = useCallback(
    (messages: ChatMessageRecord[], config: SessionConfigRecord): void => {
      if (!currentSessionId) return;
      if (saveTimerRef.current) {
        clearTimeout(saveTimerRef.current);
      }
      const sessionId = currentSessionId;
      saveTimerRef.current = setTimeout(async () => {
        const existing = await getSession(sessionId);
        if (!existing) return;

        // Derive title from first user message
        const firstUserMsg = messages.find((m) => m.role === "user");
        const title = firstUserMsg?.content.slice(0, 60) || existing.title;

        const updated: ChatSessionRecord = {
          ...existing,
          title,
          messages,
          config,
          updatedAt: new Date(),
        };
        await updateSession(updated);
        setSessions((prev) =>
          prev
            .map((s) =>
              s.id === sessionId
                ? {
                    id: s.id,
                    title,
                    updatedAt: updated.updatedAt,
                    messageCount: messages.length,
                  }
                : s
            )
            .sort((a, b) => b.updatedAt.getTime() - a.updatedAt.getTime())
        );
      }, 1000);
    },
    [currentSessionId]
  );

  return {
    sessions,
    currentSessionId,
    isLoading,
    createNewSession,
    loadSession,
    deleteSession,
    saveCurrentSession,
  };
}
