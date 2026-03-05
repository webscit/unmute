import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export interface SettingsValues {
  voice: string;
  customInstructions: string;
  backendUrl: string;
}

interface SettingsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  values: SettingsValues;
  onChange: (values: SettingsValues) => void;
}

/** Available voices (subset; extend as needed) */
const VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"] as const;

export function SettingsDialog({
  open,
  onOpenChange,
  values,
  onChange,
}: SettingsDialogProps) {
  function update(patch: Partial<SettingsValues>) {
    onChange({ ...values, ...patch });
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Settings</DialogTitle>
        </DialogHeader>

        <div className="flex flex-col gap-4">
          {/* Voice selection */}
          <div className="flex flex-col gap-1.5">
            <label className="text-sm font-medium" htmlFor="voice-select">
              Voice
            </label>
            <select
              id="voice-select"
              value={values.voice}
              onChange={(e) => update({ voice: e.target.value })}
              className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors focus:outline-none focus:ring-1 focus:ring-ring"
            >
              {VOICES.map((v) => (
                <option key={v} value={v}>
                  {v.charAt(0).toUpperCase() + v.slice(1)}
                </option>
              ))}
            </select>
          </div>

          {/* Custom instructions */}
          <div className="flex flex-col gap-1.5">
            <label className="text-sm font-medium" htmlFor="custom-instructions">
              Custom instructions
            </label>
            <textarea
              id="custom-instructions"
              rows={4}
              value={values.customInstructions}
              onChange={(e) => update({ customInstructions: e.target.value })}
              placeholder="Optional system prompt additions…"
              className="flex w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring resize-none"
            />
          </div>

          {/* Backend URL override */}
          <div className="flex flex-col gap-1.5">
            <label className="text-sm font-medium" htmlFor="backend-url">
              Backend URL
            </label>
            <Input
              id="backend-url"
              type="url"
              value={values.backendUrl}
              onChange={(e) => update({ backendUrl: e.target.value })}
              placeholder={import.meta.env.VITE_BACKEND_URL ?? "http://localhost:8000"}
            />
          </div>
        </div>

        <DialogFooter showCloseButton>
          <Button onClick={() => onOpenChange(false)}>Save</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
