"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Volume2, VolumeX, Play, Pause, RotateCcw } from "lucide-react";

interface AudioPanelProps {
  mockMode?: boolean;
}

export function AudioPanel({ mockMode = true }: AudioPanelProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  // Handle audio state
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleLoadedMetadata = () => {
      setDuration(audio.duration);
      setIsReady(true);
    };

    const handleTimeUpdate = () => {
      setCurrentTime(audio.currentTime);
    };

    const handleEnded = () => {
      setIsPlaying(false);
      setCurrentTime(0);
    };

    audio.addEventListener("loadedmetadata", handleLoadedMetadata);
    audio.addEventListener("timeupdate", handleTimeUpdate);
    audio.addEventListener("ended", handleEnded);

    return () => {
      audio.removeEventListener("loadedmetadata", handleLoadedMetadata);
      audio.removeEventListener("timeupdate", handleTimeUpdate);
      audio.removeEventListener("ended", handleEnded);
    };
  }, []);

  const togglePlay = async () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
      setIsPlaying(false);
    } else {
      try {
        await audio.play();
        setIsPlaying(true);
      } catch (err) {
        console.error("Audio playback error:", err);
      }
    }
  };

  const toggleMute = () => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.muted = !audio.muted;
    setIsMuted(!isMuted);
  };

  const restart = () => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = 0;
    setCurrentTime(0);
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <Card className="glass overflow-hidden">
      <div className="p-4 border-b border-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Volume2 className="w-5 h-5 text-primary" />
          <span className="font-medium">Audio Output</span>
        </div>
        {mockMode && (
          <span className="text-xs text-muted-foreground bg-muted px-2 py-1 rounded">
            Mock Mode
          </span>
        )}
      </div>

      <div className="p-6">
        {mockMode ? (
          <div className="space-y-4">
            {/* Waveform visualization (mock) */}
            <div className="h-20 bg-muted rounded-lg flex items-center justify-center gap-1 px-4">
              {Array.from({ length: 40 }).map((_, i) => (
                <div
                  key={i}
                  className={`w-1 bg-primary/60 rounded-full transition-all duration-150 ${
                    isPlaying ? "animate-pulse" : ""
                  }`}
                  style={{
                    height: `${Math.random() * 60 + 20}%`,
                    animationDelay: `${i * 50}ms`,
                  }}
                />
              ))}
            </div>

            {/* Progress bar */}
            <div className="space-y-2">
              <div className="h-1 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary transition-all duration-100"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>{formatTime(currentTime)}</span>
                <span>{formatTime(duration)}</span>
              </div>
            </div>

            {/* Controls */}
            <div className="flex items-center justify-center gap-4">
              <Button
                size="icon"
                variant="ghost"
                onClick={restart}
                disabled={!isReady}
              >
                <RotateCcw className="w-4 h-4" />
              </Button>
              <Button
                size="lg"
                className="w-14 h-14 rounded-full glow"
                onClick={togglePlay}
                disabled={!isReady}
              >
                {isPlaying ? (
                  <Pause className="w-6 h-6" />
                ) : (
                  <Play className="w-6 h-6 ml-1" />
                )}
              </Button>
              <Button
                size="icon"
                variant="ghost"
                onClick={toggleMute}
                disabled={!isReady}
              >
                {isMuted ? (
                  <VolumeX className="w-4 h-4" />
                ) : (
                  <Volume2 className="w-4 h-4" />
                )}
              </Button>
            </div>

            <p className="text-center text-xs text-muted-foreground">
              {isReady
                ? "Press play to hear a demo"
                : "Loading demo audio..."}
            </p>

            {/* Hidden audio element */}
            <audio ref={audioRef} preload="metadata">
              <source src="/demo.mp3" type="audio/mpeg" />
              {/* Fallback: generate a simple tone if no demo.mp3 */}
            </audio>
          </div>
        ) : (
          <div className="text-center py-8">
            <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mx-auto mb-4">
              <Volume2 className="w-8 h-8 text-muted-foreground" />
            </div>
            <p className="text-sm text-muted-foreground">
              Waiting for backend connection...
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Audio will play here when signs are detected
            </p>
          </div>
        )}
      </div>
    </Card>
  );
}

