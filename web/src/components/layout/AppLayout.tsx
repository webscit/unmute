import { type ReactNode } from "react";
import { DebugPanel } from "@/components/chat/DebugPanel";
import type { DebugEvent } from "@/hooks/useRealtimeSession";

interface AppLayoutProps {
  sidebar: ReactNode;
  header: ReactNode;
  chatArea: ReactNode;
  audioControls: ReactNode;
  /** Debug panel state */
  debugMode?: boolean;
  onDebugModeChange?: (open: boolean) => void;
  debugEvents?: DebugEvent[];
  onClearDebugEvents?: () => void;
}

export function AppLayout({
  sidebar,
  header,
  chatArea,
  audioControls,
  debugMode = false,
  onDebugModeChange,
  debugEvents = [],
  onClearDebugEvents,
}: AppLayoutProps) {
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background text-foreground">
      {/* Sidebar */}
      <aside className="hidden md:flex shrink-0 border-r">
        {sidebar}
      </aside>

      {/* Main area */}
      <div className="flex flex-col flex-1 min-w-0">
        {header}

        <main className="flex-1 min-h-0">{chatArea}</main>

        <div className="shrink-0 border-t">{audioControls}</div>
      </div>

      {/* Debug drawer */}
      <DebugPanel
        open={debugMode}
        onOpenChange={(open) => onDebugModeChange?.(open)}
        events={debugEvents}
        onClear={() => onClearDebugEvents?.()}
      />
    </div>
  );
}
