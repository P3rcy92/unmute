"use client";

import { useState, useEffect, useRef, useCallback } from "react";

export type WebSocketStatus = "disconnected" | "connecting" | "connected";

interface UseWebSocketOptions {
  url: string | null;
  onMessage?: (data: unknown) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Event) => void;
  reconnectAttempts?: number;
  reconnectInterval?: number;
}

interface UseWebSocketReturn {
  status: WebSocketStatus;
  send: (data: unknown) => void;
  connect: () => void;
  disconnect: () => void;
  lastMessage: unknown | null;
}

export function useWebSocket({
  url,
  onMessage,
  onConnect,
  onDisconnect,
  onError,
  reconnectAttempts = 5,
  reconnectInterval = 3000,
}: UseWebSocketOptions): UseWebSocketReturn {
  const [status, setStatus] = useState<WebSocketStatus>("disconnected");
  const [lastMessage, setLastMessage] = useState<unknown | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectCountRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isManualDisconnectRef = useRef(false);

  const clearReconnectTimeout = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
  }, []);

  const disconnect = useCallback(() => {
    isManualDisconnectRef.current = true;
    clearReconnectTimeout();
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    setStatus("disconnected");
  }, [clearReconnectTimeout]);

  const connect = useCallback(() => {
    if (!url) {
      console.warn("WebSocket URL is not defined");
      return;
    }

    // Clean up existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    isManualDisconnectRef.current = false;
    setStatus("connecting");

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("WebSocket connected");
        setStatus("connected");
        reconnectCountRef.current = 0;
        onConnect?.();
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setLastMessage(data);
          onMessage?.(data);
        } catch {
          // If not JSON, pass raw data
          setLastMessage(event.data);
          onMessage?.(event.data);
        }
      };

      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        onError?.(error);
      };

      ws.onclose = () => {
        console.log("WebSocket disconnected");
        setStatus("disconnected");
        wsRef.current = null;
        onDisconnect?.();

        // Auto-reconnect if not manual disconnect
        if (!isManualDisconnectRef.current && reconnectCountRef.current < reconnectAttempts) {
          reconnectCountRef.current++;
          console.log(`Reconnecting... attempt ${reconnectCountRef.current}/${reconnectAttempts}`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        }
      };
    } catch (error) {
      console.error("Failed to create WebSocket:", error);
      setStatus("disconnected");
    }
  }, [url, onConnect, onDisconnect, onMessage, onError, reconnectAttempts, reconnectInterval]);

  const send = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const message = typeof data === "string" ? data : JSON.stringify(data);
      wsRef.current.send(message);
    } else {
      console.warn("WebSocket is not connected. Cannot send message.");
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      isManualDisconnectRef.current = true;
      clearReconnectTimeout();
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [clearReconnectTimeout]);

  return {
    status,
    send,
    connect,
    disconnect,
    lastMessage,
  };
}

