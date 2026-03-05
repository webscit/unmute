import { type ReactNode } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { MessageSquare } from "lucide-react";

interface SessionSummary {
  id: string;
  label: string;
  date: Date;
}

interface AppLayoutProps {
  header: ReactNode;
  chatArea: ReactNode;
  audioControls: ReactNode;
  /** Session history list (populated by unmute-boi) */
  sessions?: SessionSummary[];
  activeSessionId?: string;
  onSessionSelect?: (id: string) => void;
}

export function AppLayout({
  header,
  chatArea,
  audioControls,
  sessions = [],
  activeSessionId,
  onSessionSelect,
}: AppLayoutProps) {
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background text-foreground">
      {/* Sidebar */}
      <aside className="hidden md:flex flex-col w-56 shrink-0 border-r bg-sidebar text-sidebar-foreground">
        <div className="px-4 py-3 font-semibold text-sm text-sidebar-foreground/70 uppercase tracking-widest">
          Sessions
        </div>
        <Separator />
        <ScrollArea className="flex-1">
          <nav className="px-2 py-2 flex flex-col gap-1">
            {sessions.length === 0 ? (
              <p className="text-xs text-sidebar-foreground/50 px-2 py-3 text-center">
                No sessions yet
              </p>
            ) : (
              sessions.map((s) => (
                <button
                  key={s.id}
                  onClick={() => onSessionSelect?.(s.id)}
                  className={`flex items-center gap-2 w-full text-left px-3 py-2 rounded-md text-sm transition-colors ${
                    s.id === activeSessionId
                      ? "bg-sidebar-accent text-sidebar-accent-foreground"
                      : "hover:bg-sidebar-accent/50 text-sidebar-foreground"
                  }`}
                >
                  <MessageSquare className="h-3.5 w-3.5 shrink-0" />
                  <span className="truncate">{s.label}</span>
                </button>
              ))
            )}
          </nav>
        </ScrollArea>
      </aside>

      {/* Main area */}
      <div className="flex flex-col flex-1 min-w-0">
        {header}

        <main className="flex-1 min-h-0">{chatArea}</main>

        <div className="shrink-0 border-t">{audioControls}</div>
      </div>
    </div>
  );
}
