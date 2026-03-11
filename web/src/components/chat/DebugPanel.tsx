import { useState, useMemo, useRef, useEffect } from "react";
import { Trash2, Filter } from "lucide-react";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { DebugEventItem } from "./DebugEvent";
import type { DebugEvent } from "@/hooks/useRealtimeSession";

interface DebugPanelProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  events: DebugEvent[];
  onClear: () => void;
}

export function DebugPanel({
  open,
  onOpenChange,
  events,
  onClear,
}: DebugPanelProps) {
  const [filter, setFilter] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const isNearBottomRef = useRef(true);

  const filtered = useMemo(() => {
    if (!filter) return events;
    const lower = filter.toLowerCase();
    return events.filter((e) => e.type.toLowerCase().includes(lower));
  }, [events, filter]);

  function handleScroll(e: React.UIEvent<HTMLDivElement>) {
    const el = e.currentTarget;
    const distanceFromBottom =
      el.scrollHeight - el.scrollTop - el.clientHeight;
    isNearBottomRef.current = distanceFromBottom < 80;
  }

  useEffect(() => {
    if (isNearBottomRef.current) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [filtered]);

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-full sm:max-w-md p-0 flex flex-col">
        <SheetHeader className="px-4 pt-4 pb-2 shrink-0">
          <div className="flex items-center justify-between">
            <SheetTitle className="text-sm">Debug Events</SheetTitle>
            <Badge variant="secondary" className="text-xs">
              {filtered.length}
              {filter && ` / ${events.length}`}
            </Badge>
          </div>
          <SheetDescription className="sr-only">
            WebSocket event log for debugging
          </SheetDescription>
        </SheetHeader>

        <div className="flex items-center gap-2 px-4 pb-2 shrink-0">
          <div className="relative flex-1">
            <Filter className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
            <Input
              placeholder="Filter by event type..."
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="pl-8 h-8 text-xs"
            />
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 shrink-0"
            onClick={onClear}
            aria-label="Clear all events"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </div>

        <ScrollArea className="flex-1 min-h-0">
          <div className="px-4 pb-4" onScroll={handleScroll}>
            {filtered.length === 0 ? (
              <p className="text-xs text-muted-foreground text-center py-8">
                {events.length === 0
                  ? "No events yet. Connect a session to see events."
                  : "No events match the filter."}
              </p>
            ) : (
              filtered.map((event, i) => (
                <DebugEventItem key={`${event.timestamp.getTime()}-${i}`} event={event} />
              ))
            )}
            <div ref={bottomRef} />
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}
