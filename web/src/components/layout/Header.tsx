import { Settings } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";

interface HeaderProps {
  debugMode: boolean;
  onDebugModeChange: (value: boolean) => void;
  onSettingsOpen: () => void;
}

export function Header({ debugMode, onDebugModeChange, onSettingsOpen }: HeaderProps) {
  return (
    <header className="flex items-center justify-between px-4 py-3 border-b bg-background">
      <h1 className="text-lg font-semibold tracking-tight">Unmute</h1>

      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <Switch
            id="debug-mode"
            checked={debugMode}
            onCheckedChange={onDebugModeChange}
          />
          <label htmlFor="debug-mode" className="text-sm cursor-pointer">
            Debug
          </label>
        </div>

        <Button
          variant="ghost"
          size="icon"
          aria-label="Open settings"
          onClick={onSettingsOpen}
        >
          <Settings className="h-4 w-4" />
        </Button>
      </div>
    </header>
  );
}
