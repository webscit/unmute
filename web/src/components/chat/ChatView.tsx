import { useEffect, useRef } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ChatMessage, type ChatMessageData } from "./ChatMessage";

interface ChatViewProps {
  messages: ChatMessageData[];
  /** Whether the assistant is currently speaking (speech activity indicator) */
  isSpeaking?: boolean;
}

export function ChatView({ messages, isSpeaking = false }: ChatViewProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const isNearBottomRef = useRef(true);

  // Track whether user has scrolled away from bottom
  function handleScroll(e: React.UIEvent<HTMLDivElement>) {
    const el = e.currentTarget;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    isNearBottomRef.current = distanceFromBottom < 80;
  }

  // Auto-scroll only if near bottom
  useEffect(() => {
    if (isNearBottomRef.current) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

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
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-48 text-muted-foreground text-sm gap-2">
              <p>No messages yet.</p>
              <p>Start a session and speak to begin.</p>
            </div>
          ) : (
            messages.map((msg) => (
              <ChatMessage key={msg.id} message={msg} />
            ))
          )}
          <div ref={bottomRef} />
        </div>
      </ScrollArea>
    </div>
  );
}
