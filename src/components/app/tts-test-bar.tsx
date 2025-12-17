"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Volume2, Send, Loader2, Square } from "lucide-react";
import { useElevenLabs, type ElevenLabsStatus } from "@/hooks/use-elevenlabs";
import { VOICES } from "@/lib/config";

interface TTSTestBarProps {
  voiceId: string;
}

export function TTSTestBar({ voiceId }: TTSTestBarProps) {
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  
  const { status, speak, stop, isReady } = useElevenLabs({
    voiceId,
    onError: (err) => {
      setError(err);
      setTimeout(() => setError(null), 5000);
    },
  });

  // Trouver le nom de la voix
  const voiceName = VOICES.find(v => v.id === voiceId)?.name || "Unknown";

  // Envoyer le texte
  const handleSend = async () => {
    if (!text.trim()) return;
    setError(null);
    await speak(text.trim());
    setText(""); // Clear input après envoi
  };

  // Gérer Entrée pour envoyer
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Vérifier si l'API key est configurée
  const hasApiKey = !!process.env.NEXT_PUBLIC_ELEVENLABS_API_KEY;

  const getStatusColor = (s: ElevenLabsStatus) => {
    switch (s) {
      case "playing": return "text-green-500";
      case "loading": return "text-yellow-500";
      case "error": return "text-red-500";
      default: return "text-muted-foreground";
    }
  };

  const getStatusText = (s: ElevenLabsStatus) => {
    switch (s) {
      case "playing": return "Playing";
      case "loading": return "Generating...";
      case "error": return "Error";
      default: return "Ready";
    }
  };

  return (
    <Card className="mt-4 glass p-4">
      <div className="flex items-center gap-2 mb-3">
        <Volume2 className="w-5 h-5 text-primary" />
        <span className="font-medium">Test Text-to-Speech</span>
        <span className="text-xs text-muted-foreground">
          (Voice: {voiceName})
        </span>
        
        {/* Status indicator */}
        <div className={`ml-auto flex items-center gap-1.5 ${getStatusColor(status)}`}>
          {status === "loading" && <Loader2 className="w-4 h-4 animate-spin" />}
          {status === "playing" && (
            <div className="flex gap-0.5">
              <div className="w-1 h-3 bg-green-500 rounded-full animate-pulse" />
              <div className="w-1 h-3 bg-green-500 rounded-full animate-pulse" style={{ animationDelay: "0.1s" }} />
              <div className="w-1 h-3 bg-green-500 rounded-full animate-pulse" style={{ animationDelay: "0.2s" }} />
            </div>
          )}
          <span className="text-xs">{getStatusText(status)}</span>
        </div>
      </div>

      {!hasApiKey ? (
        <div className="text-center py-4">
          <p className="text-sm text-yellow-500">
            ⚠️ Configure NEXT_PUBLIC_ELEVENLABS_API_KEY dans .env.local
          </p>
        </div>
      ) : (
        <>
          {/* Input bar */}
          <div className="flex gap-2">
            <Input
              placeholder="Type a word or phrase to test..."
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={status === "loading"}
              className="flex-1"
            />
            {status === "playing" ? (
              <Button onClick={stop} size="icon" variant="destructive">
                <Square className="w-4 h-4" />
              </Button>
            ) : (
              <Button 
                onClick={handleSend}
                disabled={!isReady || !text.trim() || status === "loading"}
                size="icon"
              >
                {status === "loading" ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
              </Button>
            )}
          </div>

          {/* Error message */}
          {error && (
            <p className="mt-2 text-xs text-red-500">❌ {error}</p>
          )}

          {/* Hint */}
          <p className="mt-2 text-xs text-muted-foreground">
            💡 Press Enter to send, or click the button
          </p>
        </>
      )}
    </Card>
  );
}
