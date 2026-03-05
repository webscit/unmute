import { Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuAction,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
} from "@/components/ui/sidebar";
import type { SessionSummary } from "@/hooks/useSessionHistory";

interface SessionSidebarProps {
  sessions: SessionSummary[];
  currentSessionId: string | null;
  onNewSession: () => void;
  onLoadSession: (id: string) => void;
  onDeleteSession: (id: string) => void;
}

function formatDate(date: Date): string {
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
  if (diffDays === 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString();
}

export function SessionSidebar({
  sessions,
  currentSessionId,
  onNewSession,
  onLoadSession,
  onDeleteSession,
}: SessionSidebarProps) {
  return (
    <SidebarProvider>
      <Sidebar>
        <SidebarHeader className="p-3">
          <Button
            variant="outline"
            size="sm"
            className="w-full justify-start gap-2"
            onClick={onNewSession}
          >
            <Plus className="h-4 w-4" />
            New Chat
          </Button>
        </SidebarHeader>

        <SidebarContent>
          <SidebarGroup>
            <SidebarGroupLabel>Sessions</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {sessions.length === 0 ? (
                  <p className="text-xs text-sidebar-foreground/50 px-2 py-3 text-center">
                    No sessions yet
                  </p>
                ) : (
                  sessions.map((session) => (
                    <SidebarMenuItem key={session.id}>
                      <SidebarMenuButton
                        isActive={session.id === currentSessionId}
                        onClick={() => onLoadSession(session.id)}
                        className="pr-8"
                      >
                        <div className="flex flex-col items-start gap-0.5 min-w-0">
                          <span className="truncate text-sm font-medium w-full">
                            {session.title}
                          </span>
                          <span className="text-xs text-sidebar-foreground/50">
                            {formatDate(session.updatedAt)} · {session.messageCount} msg
                          </span>
                        </div>
                      </SidebarMenuButton>
                      <SidebarMenuAction
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeleteSession(session.id);
                        }}
                        aria-label="Delete session"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </SidebarMenuAction>
                    </SidebarMenuItem>
                  ))
                )}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        </SidebarContent>
      </Sidebar>
    </SidebarProvider>
  );
}
