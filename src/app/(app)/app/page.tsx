"use client";

import { useState, useEffect, useCallback } from "react";
import { CameraPanel } from "@/components/app/camera-panel";
import { WSStatus } from "@/components/app/ws-status";
import { TTSTestBar } from "@/components/app/tts-test-bar";
import type { WebSocketStatus } from "@/hooks/use-websocket";
import { useElevenLabs } from "@/hooks/use-elevenlabs";
import { createClient } from "@/lib/supabase/client";
import type { VoiceId } from "@/lib/config";

export default function AppPage() {
  const [wsStatus, setWsStatus] = useState<WebSocketStatus>("disconnected");
  const [lastWord, setLastWord] = useState<string | null>(null);
  const [voiceId, setVoiceId] = useState<VoiceId>("pNInz6obpgDQGcFmaJgB"); // Default: Adam
  const [autoSpeak, setAutoSpeak] = useState(true); // Active par défaut

  // Hook ElevenLabs pour la synthèse vocale automatique
  const { speak, status: ttsStatus } = useElevenLabs({
    voiceId,
    onError: (err) => console.error("TTS Error:", err),
  });

  // Charger la voix de l'utilisateur depuis ses settings
  useEffect(() => {
    async function loadUserVoice() {
      const supabase = createClient();
      const { data: { user } } = await supabase.auth.getUser();
      
      if (user) {
        const { data: profile } = await supabase
          .from("profiles")
          .select("voice_id")
          .eq("id", user.id)
          .single();
        
        if (profile?.voice_id) {
          setVoiceId(profile.voice_id as VoiceId);
        }
      }
    }
    
    loadUserVoice();
  }, []);

  // Callback quand un mot est reconnu par le backend ML
  // Note: La déduplication est déjà faite dans CameraPanel (lastWordRef)
  // Donc on ne reçoit ici QUE les nouveaux mots
  const handleWordReceived = useCallback((word: string) => {
    setLastWord(word);
    
    // Envoyer automatiquement à ElevenLabs si autoSpeak est activé
    if (autoSpeak && word) {
      console.log("🔊 Auto-speaking:", word);
      speak(word);
    }
  }, [autoSpeak, speak]);

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
        onWordReceived={handleWordReceived}
      />

      {/* Last recognized word + TTS status */}
      {lastWord && (
        <div className="mt-4 glass rounded-xl p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground mb-1">Dernier mot reconnu</p>
              <p className="text-2xl font-semibold">{lastWord}</p>
            </div>
            <div className="flex items-center gap-3">
              {/* TTS Status indicator */}
              {ttsStatus === "loading" && (
                <span className="text-xs text-yellow-500">🔄 Generating...</span>
              )}
              {ttsStatus === "playing" && (
                <span className="text-xs text-green-500">🔊 Playing</span>
              )}
              
              {/* Auto-speak toggle */}
              <button
                onClick={() => setAutoSpeak(!autoSpeak)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                  autoSpeak 
                    ? "bg-primary/20 text-primary" 
                    : "bg-muted text-muted-foreground"
                }`}
              >
                {autoSpeak ? "🔊 Auto ON" : "🔇 Auto OFF"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* TTS Test Bar */}
      <TTSTestBar voiceId={voiceId} />

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
