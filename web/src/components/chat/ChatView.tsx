import { useEffect, useRef, useMemo } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ChatMessage, type ChatMessageData } from "./ChatMessage";
import { DebugEventItem } from "./DebugEvent";
import type { DebugEvent } from "@/hooks/useRealtimeSession";

interface ChatViewProps {
  messages: ChatMessageData[];
  /** Whether the assistant is currently speaking (speech activity indicator) */
  isSpeaking?: boolean;
  /** When true, debug events are interleaved with messages */
  debugMode?: boolean;
  /** Debug events to interleave (only used when debugMode is true) */
  debugEvents?: DebugEvent[];
}

type TimelineEntry =
  | { kind: "message"; data: ChatMessageData }
  | { kind: "event"; data: DebugEvent };

export function ChatView({
  messages,
  isSpeaking = false,
  debugMode = false,
  debugEvents = [],
}: ChatViewProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const isNearBottomRef = useRef(true);

  // Track whether user has scrolled away from bottom
  function handleScroll(e: React.UIEvent<HTMLDivElement>) {
    const el = e.currentTarget;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    isNearBottomRef.current = distanceFromBottom < 80;
  }

  // Build interleaved timeline when debug mode is on
  const timeline = useMemo<TimelineEntry[]>(() => {
    if (!debugMode || debugEvents.length === 0) {
      return messages.map((m) => ({ kind: "message" as const, data: m }));
    }

    const entries: TimelineEntry[] = [
      ...messages.map((m) => ({ kind: "message" as const, data: m })),
      ...debugEvents.map((e) => ({ kind: "event" as const, data: e })),
    ];

    entries.sort((a, b) => {
      const ta = a.data.timestamp.getTime();
      const tb = b.data.timestamp.getTime();
      if (ta !== tb) return ta - tb;
      // Messages before events at same timestamp
      if (a.kind !== b.kind) return a.kind === "message" ? -1 : 1;
      return 0;
    });

    return entries;
  }, [messages, debugEvents, debugMode]);

  // Auto-scroll only if near bottom
  useEffect(() => {
    if (isNearBottomRef.current) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [timeline]);

  return (
    <div className="flex flex-col h-full">
      {isSpeaking && (
        <div className="flex items-center gap-2 px-4 py-1.5 bg-primary/10 text-primary text-xs font-medium border-b border-primary/20">
          <span className="flex gap-0.5">
            {[0, 1, 2].map((i) => (
              <span
                key={i}
                className="inline-block w-1 h-3 bg-primary rounded-full animate-bounce"
                style={{ animationDelay: `${i * 0.15}s` }}
              />
            ))}
          </span>
          Assistant is speaking…
        </div>
      )}

      <ScrollArea className="flex-1">
        <div
          ref={scrollAreaRef}
          className="px-4 py-4"
          onScroll={handleScroll}
        >
          {timeline.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-48 text-muted-foreground text-sm gap-2">
              <p>No messages yet.</p>
              <p>Start a session and speak to begin.</p>
            </div>
          ) : (
            timeline.map((entry, i) =>
              entry.kind === "message" ? (
                <ChatMessage key={entry.data.id} message={entry.data} />
              ) : (
                <DebugEventItem
                  key={`debug-${entry.data.timestamp.getTime()}-${i}`}
                  event={entry.data}
                />
              ),
            )
          )}
          <div ref={bottomRef} />
        </div>
      </ScrollArea>
    </div>
  );
}
