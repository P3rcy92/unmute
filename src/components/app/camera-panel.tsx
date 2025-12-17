"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Camera, CameraOff, RefreshCw, Wifi, WifiOff } from "lucide-react";
import { useWebSocket, WebSocketStatus } from "@/hooks/use-websocket";

// Ajouter /ws à l'URL si elle ne se termine pas déjà par /ws
const BASE_WS_URL = process.env.NEXT_PUBLIC_WS_URL || null;
const WS_URL = BASE_WS_URL 
  ? (BASE_WS_URL.endsWith("/ws") ? BASE_WS_URL : `${BASE_WS_URL}/ws`)
  : null;
const FRAME_RATE = 15; // 15 FPS
const FRAME_INTERVAL = 1000 / FRAME_RATE; // ~66ms
const JPEG_QUALITY = 0.7; // 70% quality for good balance size/quality

interface CameraPanelProps {
  onStreamReady?: (stream: MediaStream | null) => void;
  onWordReceived?: (word: string) => void;
  onWsStatusChange?: (status: WebSocketStatus) => void;
}

export function CameraPanel({ 
  onStreamReady, 
  onWordReceived,
  onWsStatusChange 
}: CameraPanelProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  const [isEnabled, setIsEnabled] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [currentWord, setCurrentWord] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const lastWordRef = useRef<string | null>(null); // Pour filtrer les mots répétés

  // WebSocket connection
  const { status: wsStatus, send, connect, disconnect } = useWebSocket({
    url: WS_URL,
    onMessage: (data) => {
      // Handle incoming word from backend
      if (typeof data === "object" && data !== null && "type" in data) {
        const message = data as { type: string; word?: string };
        if (message.type === "word" && message.word) {
          // Filtrer les mots répétés côté frontend
          if (message.word !== lastWordRef.current) {
            lastWordRef.current = message.word;
            setCurrentWord(message.word);
            onWordReceived?.(message.word);
          }
        }
      }
    },
    onConnect: () => {
      console.log("Connected to sign language recognition backend");
    },
    onDisconnect: () => {
      console.log("Disconnected from backend");
      setIsStreaming(false);
    },
  });

  // Notify parent of WebSocket status changes
  useEffect(() => {
    onWsStatusChange?.(wsStatus);
  }, [wsStatus, onWsStatusChange]);

  // Capture and send frame
  const captureAndSendFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || wsStatus !== "connected") {
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    if (!ctx || video.readyState !== video.HAVE_ENOUGH_DATA) {
      return;
    }

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw current frame to canvas
    ctx.drawImage(video, 0, 0);

    // Convert to base64 JPEG
    const frameData = canvas.toDataURL("image/jpeg", JPEG_QUALITY);

    // Send frame to backend
    send({
      type: "frame",
      data: frameData,
      timestamp: Date.now(),
    });
  }, [wsStatus, send]);

  // Start frame streaming
  const startStreaming = useCallback(() => {
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
    }

    frameIntervalRef.current = setInterval(captureAndSendFrame, FRAME_INTERVAL);
    setIsStreaming(true);
    console.log(`Started streaming at ${FRAME_RATE} FPS`);
  }, [captureAndSendFrame]);

  // Stop frame streaming
  const stopStreaming = useCallback(() => {
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    setIsStreaming(false);
    console.log("Stopped streaming");
  }, []);

  // Auto-start streaming when camera is enabled and WebSocket is connected
  useEffect(() => {
    if (isEnabled && wsStatus === "connected" && !isStreaming) {
      startStreaming();
    } else if ((!isEnabled || wsStatus !== "connected") && isStreaming) {
      stopStreaming();
    }
  }, [isEnabled, wsStatus, isStreaming, startStreaming, stopStreaming]);

  const startCamera = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 }, // Reduced for better performance
          height: { ideal: 480 },
          facingMode: "user",
        },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }

      setStream(mediaStream);
      setIsEnabled(true);
      onStreamReady?.(mediaStream);

      // Connect to WebSocket if URL is configured
      if (WS_URL) {
        connect();
      }
    } catch (err) {
      console.error("Camera error:", err);
      if (err instanceof Error) {
        if (err.name === "NotAllowedError") {
          setError("Camera access denied. Please allow camera permissions.");
        } else if (err.name === "NotFoundError") {
          setError("No camera found. Please connect a camera.");
        } else {
          setError("Failed to access camera. Please try again.");
        }
      }
    } finally {
      setIsLoading(false);
    }
  }, [onStreamReady, connect]);

  const stopCamera = useCallback(() => {
    // Stop streaming first
    stopStreaming();
    
    // Disconnect WebSocket
    disconnect();

    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsEnabled(false);
    setCurrentWord(null);
    lastWordRef.current = null; // Reset pour permettre de redétecter le même mot
    onStreamReady?.(null);
  }, [stream, onStreamReady, stopStreaming, disconnect]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
      }
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [stream]);

  // Determine display status
  const getDisplayStatus = (): "mock" | "disconnected" | "connected" => {
    if (!WS_URL) return "mock";
    if (wsStatus === "connected") return "connected";
    return "disconnected";
  };

  return (
    <Card className="glass overflow-hidden">
      <div className="p-4 border-b border-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Camera className="w-5 h-5 text-primary" />
          <span className="font-medium">Camera</span>
          {isStreaming && (
            <span className="text-xs text-muted-foreground">
              ({FRAME_RATE} FPS)
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {/* WebSocket status indicator */}
          {isEnabled && (
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              {wsStatus === "connected" ? (
                <Wifi className="w-3.5 h-3.5 text-green-500" />
              ) : wsStatus === "connecting" ? (
                <Wifi className="w-3.5 h-3.5 text-yellow-500 animate-pulse" />
              ) : (
                <WifiOff className="w-3.5 h-3.5 text-red-500" />
              )}
            </div>
          )}
          
          {isEnabled && (
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                stopCamera();
                startCamera();
              }}
              disabled={isLoading}
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? "animate-spin" : ""}`} />
            </Button>
          )}
          <Button
            size="sm"
            variant={isEnabled ? "destructive" : "default"}
            onClick={isEnabled ? stopCamera : startCamera}
            disabled={isLoading}
          >
            {isLoading ? (
              <>
                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                Loading...
              </>
            ) : isEnabled ? (
              <>
                <CameraOff className="w-4 h-4 mr-2" />
                Disable
              </>
            ) : (
              <>
                <Camera className="w-4 h-4 mr-2" />
                Enable
              </>
            )}
          </Button>
        </div>
      </div>

      <div className="aspect-video bg-muted relative">
        {error ? (
          <div className="absolute inset-0 flex items-center justify-center p-6">
            <div className="text-center">
              <div className="w-16 h-16 rounded-full bg-destructive/20 flex items-center justify-center mx-auto mb-4">
                <CameraOff className="w-8 h-8 text-destructive" />
              </div>
              <p className="text-sm text-muted-foreground">{error}</p>
              <Button
                size="sm"
                variant="outline"
                className="mt-4"
                onClick={startCamera}
              >
                Try again
              </Button>
            </div>
          </div>
        ) : !isEnabled ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mx-auto mb-4">
                <Camera className="w-8 h-8 text-muted-foreground" />
              </div>
              <p className="text-sm text-muted-foreground">
                Camera is disabled
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                Enable to start signing
              </p>
            </div>
          </div>
        ) : null}

        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`w-full h-full object-cover ${
            isEnabled ? "opacity-100" : "opacity-0"
          }`}
        />

        {/* Hidden canvas for frame capture */}
        <canvas ref={canvasRef} className="hidden" />

        {/* Status indicator */}
        {isEnabled && (
          <div className="absolute top-4 left-4 flex items-center gap-2 glass-subtle rounded-full px-3 py-1.5">
            <div className={`w-2 h-2 rounded-full ${
              isStreaming ? "bg-green-500 animate-pulse" : "bg-yellow-500"
            }`} />
            <span className="text-xs font-medium">
              {isStreaming ? "Streaming" : "Live"}
            </span>
          </div>
        )}

        {/* Current recognized word overlay */}
        {isEnabled && currentWord && (
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 glass rounded-xl px-6 py-3">
            <p className="text-2xl font-bold text-center">{currentWord}</p>
          </div>
        )}

        {/* No WebSocket URL warning */}
        {isEnabled && !WS_URL && (
          <div className="absolute bottom-4 right-4 glass-subtle rounded-lg px-3 py-2">
            <p className="text-xs text-yellow-500">
              ⚠️ Mode démo - Configurez NEXT_PUBLIC_WS_URL
            </p>
          </div>
        )}
      </div>
    </Card>
  );
}
