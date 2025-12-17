"use client";

import { Badge } from "@/components/ui/badge";
import { Wifi, WifiOff, Zap } from "lucide-react";

type ConnectionStatus = "mock" | "disconnected" | "connected";

interface WSStatusProps {
  status?: ConnectionStatus;
}

export function WSStatus({ status = "mock" }: WSStatusProps) {
  const statusConfig = {
    mock: {
      icon: Zap,
      label: "Mock Mode",
      className: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
    },
    disconnected: {
      icon: WifiOff,
      label: "Disconnected",
      className: "bg-destructive/20 text-destructive border-destructive/30",
    },
    connected: {
      icon: Wifi,
      label: "Connected",
      className: "bg-green-500/20 text-green-400 border-green-500/30",
    },
  };

  const config = statusConfig[status];
  const Icon = config.icon;

  return (
    <Badge
      variant="outline"
      className={`${config.className} gap-1.5 px-3 py-1.5`}
    >
      <Icon className="w-3.5 h-3.5" />
      {config.label}
    </Badge>
  );
}

