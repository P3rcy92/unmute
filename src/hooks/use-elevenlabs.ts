"use client";

import { useState, useRef, useCallback, useEffect } from "react";

const ELEVENLABS_API_KEY = process.env.NEXT_PUBLIC_ELEVENLABS_API_KEY || "";
const ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech";

export type ElevenLabsStatus = "idle" | "loading" | "playing" | "error";

interface UseElevenLabsOptions {
  voiceId: string;
  onError?: (error: string) => void;
  onStart?: () => void;
  onEnd?: () => void;
}

interface UseElevenLabsReturn {
  status: ElevenLabsStatus;
  speak: (text: string) => Promise<void>;
  stop: () => void;
  isReady: boolean;
}

export function useElevenLabs({
  voiceId,
  onError,
  onStart,
  onEnd,
}: UseElevenLabsOptions): UseElevenLabsReturn {
  const [status, setStatus] = useState<ElevenLabsStatus>("idle");
  
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Vérifier si l'API est prête
  const isReady = !!ELEVENLABS_API_KEY && !!voiceId;

  // Arrêter l'audio en cours
  const stop = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = "";
      audioRef.current = null;
    }
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setStatus("idle");
  }, []);

  // Synthétiser et jouer le texte
  const speak = useCallback(async (text: string) => {
    if (!ELEVENLABS_API_KEY) {
      console.error("ElevenLabs API key not configured");
      onError?.("API key not configured");
      return;
    }

    if (!voiceId) {
      console.error("Voice ID not provided");
      onError?.("Voice ID not provided");
      return;
    }

    if (!text.trim()) {
      console.warn("Empty text, skipping");
      return;
    }

    // Arrêter tout audio en cours
    stop();

    setStatus("loading");
    console.log("ElevenLabs: Generating audio for:", text);

    // Créer un AbortController pour pouvoir annuler la requête
    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch(
        `${ELEVENLABS_API_URL}/${voiceId}/stream?output_format=mp3_44100_128`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY,
          },
          body: JSON.stringify({
            text: text,
            model_id: "eleven_turbo_v2",
            voice_settings: {
              stability: 0.5,
              similarity_boost: 0.75,
              speed: 1.0,
            },
          }),
          signal: abortControllerRef.current.signal,
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail?.message || `HTTP ${response.status}`);
      }

      // Convertir la réponse en blob audio
      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);

      console.log("ElevenLabs: Audio received, playing...");

      // Créer et jouer l'audio
      const audio = new Audio(audioUrl);
      audioRef.current = audio;

      audio.onplay = () => {
        setStatus("playing");
        onStart?.();
      };

      audio.onended = () => {
        setStatus("idle");
        URL.revokeObjectURL(audioUrl);
        onEnd?.();
      };

      audio.onerror = () => {
        setStatus("error");
        URL.revokeObjectURL(audioUrl);
        onError?.("Failed to play audio");
      };

      await audio.play();

    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") {
        console.log("ElevenLabs: Request cancelled");
        setStatus("idle");
        return;
      }

      console.error("ElevenLabs error:", error);
      setStatus("error");
      onError?.(error instanceof Error ? error.message : "Unknown error");
    }
  }, [voiceId, stop, onError, onStart, onEnd]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stop();
    };
  }, [stop]);

  return {
    status,
    speak,
    stop,
    isReady,
  };
}
