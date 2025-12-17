"use client";

import { useState } from "react";
import { CameraPanel } from "@/components/app/camera-panel";
import { WSStatus } from "@/components/app/ws-status";
import type { WebSocketStatus } from "@/hooks/use-websocket";

export default function AppPage() {
  const [wsStatus, setWsStatus] = useState<WebSocketStatus>("disconnected");
  const [lastWord, setLastWord] = useState<string | null>(null);

  // Map WebSocket status to display status
  const getDisplayStatus = (): "mock" | "disconnected" | "connected" => {
    if (!process.env.NEXT_PUBLIC_WS_URL) return "mock";
    if (wsStatus === "connected") return "connected";
    return "disconnected";
  };

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold">Dashboard</h1>
          <p className="text-muted-foreground">
            Enable your camera and start signing
          </p>
        </div>
        <WSStatus status={getDisplayStatus()} />
      </div>

      {/* Camera - Full width */}
      <CameraPanel 
        onWsStatusChange={setWsStatus}
        onWordReceived={(word) => setLastWord(word)}
      />

      {/* Last recognized word (optional display) */}
      {lastWord && (
        <div className="mt-4 glass rounded-xl p-4 text-center">
          <p className="text-sm text-muted-foreground mb-1">Dernier mot reconnu</p>
          <p className="text-xl font-semibold">{lastWord}</p>
        </div>
      )}

      {/* Instructions */}
      <div className="mt-8 glass rounded-2xl p-6">
        <h2 className="font-semibold mb-4">Quick Start Guide</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[
            {
              step: "1",
              title: "Enable Camera",
              description: "Click the enable button to allow camera access",
            },
            {
              step: "2",
              title: "Start Signing",
              description: "Position yourself in frame and begin signing",
            },
          ].map((item) => (
            <div key={item.step} className="flex gap-4">
              <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center shrink-0">
                <span className="text-primary font-semibold">{item.step}</span>
              </div>
              <div>
                <h3 className="font-medium">{item.title}</h3>
                <p className="text-sm text-muted-foreground">
                  {item.description}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Connection status notice */}
      <div className="mt-6 text-center text-sm text-muted-foreground">
        {getDisplayStatus() === "mock" ? (
          <p>
            🧪 Running in <span className="text-yellow-600 font-medium">Mock Mode</span> — Configure NEXT_PUBLIC_WS_URL to connect
          </p>
        ) : getDisplayStatus() === "connected" ? (
          <p>
            ✅ <span className="text-green-600 font-medium">Connected</span> to sign recognition backend
          </p>
        ) : (
          <p>
            ⏳ <span className="text-orange-600 font-medium">Disconnected</span> — Enable camera to connect
          </p>
        )}
      </div>
    </div>
  );
}
