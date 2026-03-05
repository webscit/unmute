import { Phone, PhoneOff, Mic, MicOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

interface AudioControlsProps {
  isConnected: boolean;
  onConnect: () => void;
  onDisconnect: () => void;
  isMuted: boolean;
  onToggleMute: () => void;
}

export function AudioControls({
  isConnected,
  onConnect,
  onDisconnect,
  isMuted,
  onToggleMute,
}: AudioControlsProps) {
  return (
    <div className="flex items-center gap-3">
      <Badge variant={isConnected ? "default" : "secondary"}>
        {isConnected ? "Connected" : "Disconnected"}
      </Badge>

      <Button
        variant={isConnected ? "destructive" : "default"}
        size="icon"
        onClick={isConnected ? onDisconnect : onConnect}
        aria-label={isConnected ? "Disconnect" : "Connect"}
      >
        {isConnected ? <PhoneOff /> : <Phone />}
      </Button>

      <Button
        variant={isMuted ? "secondary" : "outline"}
        size="icon"
        onClick={onToggleMute}
        disabled={!isConnected}
        aria-label={isMuted ? "Unmute" : "Mute"}
      >
        {isMuted ? <MicOff /> : <Mic />}
      </Button>
    </div>
  );
}
