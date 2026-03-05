import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";

export interface ChatMessageData {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  /** Milliseconds the model spent "thinking" before streaming started */
  thinkingTimeMs?: number;
  /** Total milliseconds from first token to last */
  responseDurationMs?: number;
  isStreaming?: boolean;
}

interface ChatMessageProps {
  message: ChatMessageData;
}

function formatTime(date: Date): string {
  return date.toTimeString().slice(0, 8); // HH:MM:SS
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-3`}>
      <div className={`max-w-[75%] ${isUser ? "items-end" : "items-start"} flex flex-col gap-1`}>
        <Card
          className={`${
            isUser
              ? "bg-primary text-primary-foreground"
              : "bg-card text-card-foreground"
          } shadow-sm`}
        >
          <CardContent className="px-4 py-2 text-sm">
            {message.content}
            {message.isStreaming && (
              <span className="inline-block w-2 h-4 ml-1 bg-current animate-pulse align-middle" />
            )}
          </CardContent>
        </Card>

        <div className="flex items-center gap-1.5 px-1">
          <span className="text-xs text-muted-foreground">
            {formatTime(message.timestamp)}
          </span>
          {message.thinkingTimeMs !== undefined && (
            <Badge variant="secondary" className="text-xs h-4 px-1.5">
              think {formatDuration(message.thinkingTimeMs)}
            </Badge>
          )}
          {message.responseDurationMs !== undefined && (
            <Badge variant="outline" className="text-xs h-4 px-1.5">
              {formatDuration(message.responseDurationMs)}
            </Badge>
          )}
        </div>
      </div>
    </div>
  );
}
