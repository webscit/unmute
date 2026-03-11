import { ChevronRight, ArrowUp, ArrowDown } from "lucide-react";
import {
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
} from "@/components/ui/collapsible";
import type { DebugEvent as DebugEventData } from "@/hooks/useRealtimeSession";

interface DebugEventProps {
  event: DebugEventData;
}

function formatTimePrecise(date: Date): string {
  const h = String(date.getHours()).padStart(2, "0");
  const m = String(date.getMinutes()).padStart(2, "0");
  const s = String(date.getSeconds()).padStart(2, "0");
  const ms = String(date.getMilliseconds()).padStart(3, "0");
  return `${h}:${m}:${s}.${ms}`;
}

/**
 * Determine color class based on event direction and type.
 * Sent (client) = blue, received (server) = green, errors = red.
 */
function eventColorClass(event: DebugEventData): {
  border: string;
  bg: string;
  text: string;
  icon: string;
} {
  if (event.type === "error" || event.type.includes("error")) {
    return {
      border: "border-red-500/30",
      bg: "bg-red-500/5",
      text: "text-red-600 dark:text-red-400",
      icon: "text-red-500",
    };
  }
  if (event.direction === "client") {
    return {
      border: "border-blue-500/30",
      bg: "bg-blue-500/5",
      text: "text-blue-600 dark:text-blue-400",
      icon: "text-blue-500",
    };
  }
  return {
    border: "border-green-500/30",
    bg: "bg-green-500/5",
    text: "text-green-600 dark:text-green-400",
    icon: "text-green-500",
  };
}

/** Simple JSON syntax highlighter using CSS classes. */
function highlightJson(obj: unknown): string {
  const raw = JSON.stringify(obj, null, 2);
  if (!raw) return "";
  return raw
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(
      /"([^"\\]*(\\.[^"\\]*)*)"(\s*:)?/g,
      (match, _p1, _p2, colon) => {
        if (colon) {
          return `<span class="text-purple-600 dark:text-purple-400">${match}</span>`;
        }
        return `<span class="text-green-700 dark:text-green-300">${match}</span>`;
      },
    )
    .replace(/\b(\d+(\.\d+)?)\b/g, '<span class="text-amber-600 dark:text-amber-400">$1</span>')
    .replace(/\b(true|false)\b/g, '<span class="text-blue-600 dark:text-blue-400">$1</span>')
    .replace(/\bnull\b/g, '<span class="text-red-500">null</span>');
}

export function DebugEventItem({ event }: DebugEventProps) {
  const colors = eventColorClass(event);
  const Arrow = event.direction === "client" ? ArrowUp : ArrowDown;

  return (
    <Collapsible>
      <div
        className={`rounded-md border ${colors.border} ${colors.bg} text-xs font-mono mb-1`}
      >
        <CollapsibleTrigger className="flex items-center gap-1.5 w-full px-2 py-1.5 text-left group cursor-pointer">
          <ChevronRight className="h-3 w-3 shrink-0 text-muted-foreground transition-transform group-data-[state=open]:rotate-90" />
          <Arrow className={`h-3 w-3 shrink-0 ${colors.icon}`} />
          <span className={`font-semibold truncate ${colors.text}`}>
            {event.type}
          </span>
          <span className="ml-auto text-muted-foreground shrink-0">
            {formatTimePrecise(event.timestamp)}
          </span>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <div className="border-t border-border/50 px-2 py-1.5 overflow-x-auto max-h-64 overflow-y-auto">
            <pre
              className="text-[11px] leading-relaxed whitespace-pre-wrap break-all"
              dangerouslySetInnerHTML={{
                __html: highlightJson(event.payload),
              }}
            />
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
}
