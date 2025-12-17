"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";
import { LANGUAGES, VOICES, type LanguageId, type VoiceId } from "@/lib/config";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Loader2, Save, ArrowLeft, Volume2, Play, Square } from "lucide-react";
import { toast } from "sonner";
import Link from "next/link";

export default function SettingsPage() {
  const router = useRouter();
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [language, setLanguage] = useState<LanguageId>("en");
  const [voiceId, setVoiceId] = useState<VoiceId>("pNInz6obpgDQGcFmaJgB");
  const [playingVoiceId, setPlayingVoiceId] = useState<string | null>(null);

  // Load user settings
  useEffect(() => {
    async function loadSettings() {
      const supabase = createClient();
      const { data: { user } } = await supabase.auth.getUser();
      
      if (!user) {
        router.push("/login");
        return;
      }

      const { data: profile } = await supabase
        .from("profiles")
        .select("language, voice_id")
        .eq("id", user.id)
        .single();

      if (profile) {
        setLanguage(profile.language as LanguageId);
        setVoiceId(profile.voice_id as VoiceId);
      }

      setIsLoading(false);
    }

    loadSettings();
  }, [router]);

  // Cleanup audio on unmount
  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
    };
  }, []);

  function playVoiceSample(voice: typeof VOICES[number]) {
    // If already playing this voice, stop it
    if (playingVoiceId === voice.id) {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      setPlayingVoiceId(null);
      return;
    }

    // Stop any currently playing audio
    if (audioRef.current) {
      audioRef.current.pause();
    }

    // Play the new voice sample
    const audio = new Audio(voice.audioFile);
    audioRef.current = audio;
    setPlayingVoiceId(voice.id);

    audio.play().catch((err) => {
      console.error("Failed to play audio:", err);
      setPlayingVoiceId(null);
    });

    audio.onended = () => {
      setPlayingVoiceId(null);
    };
  }

  function selectVoice(voice: typeof VOICES[number]) {
    setVoiceId(voice.id);
    playVoiceSample(voice);
  }

  async function saveSettings() {
    setIsSaving(true);

    try {
      const supabase = createClient();
      const { data: { user } } = await supabase.auth.getUser();

      if (!user) {
        toast.error("Not authenticated");
        return;
      }

      const { error } = await supabase
        .from("profiles")
        .upsert({
          id: user.id,
          email: user.email,
          language,
          voice_id: voiceId,
          updated_at: new Date().toISOString(),
        });

      if (error) {
        toast.error("Failed to save settings");
        return;
      }

      toast.success("Settings saved!");
    } catch {
      toast.error("An error occurred");
    } finally {
      setIsSaving(false);
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Back button */}
      <Button variant="ghost" asChild className="mb-6">
        <Link href="/app">
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to app
        </Link>
      </Button>

      <Card className="glass">
        <CardHeader>
          <CardTitle>Settings</CardTitle>
          <CardDescription>
            Customize your Unmute experience
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Language */}
          <div className="space-y-2">
            <Label htmlFor="language">Language</Label>
            <Select value={language} onValueChange={(v) => setLanguage(v as LanguageId)}>
              <SelectTrigger id="language">
                <SelectValue placeholder="Select language" />
              </SelectTrigger>
              <SelectContent>
                {LANGUAGES.map((lang) => (
                  <SelectItem key={lang.id} value={lang.id}>
                    <span className="flex items-center gap-2">
                      <span>{lang.flag}</span>
                      <span>{lang.name}</span>
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-sm text-muted-foreground">
              The language for sign language recognition
            </p>
          </div>

          <Separator />

          {/* Voice */}
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Voice</Label>
              <p className="text-sm text-muted-foreground">
                Click on a voice to listen and select it
              </p>
            </div>

            {/* Voice preview cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {VOICES.map((voice) => (
                <button
                  key={voice.id}
                  type="button"
                  onClick={() => selectVoice(voice)}
                  className={`p-4 rounded-xl border text-left transition-all ${
                    voiceId === voice.id
                      ? "border-primary bg-primary/10"
                      : "border-border hover:border-primary/50"
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div className={`w-12 h-12 rounded-full flex items-center justify-center transition-all ${
                      voiceId === voice.id ? "bg-primary/20" : "bg-muted"
                    } ${playingVoiceId === voice.id ? "animate-pulse" : ""}`}>
                      {playingVoiceId === voice.id ? (
                        <Square className={`w-5 h-5 ${
                          voiceId === voice.id ? "text-primary" : "text-muted-foreground"
                        }`} />
                      ) : (
                        <Play className={`w-5 h-5 ml-0.5 ${
                          voiceId === voice.id ? "text-primary" : "text-muted-foreground"
                        }`} />
                      )}
                    </div>
                    <div>
                      <p className="font-medium">{voice.name}</p>
                      <p className="text-xs text-muted-foreground">
                        {playingVoiceId === voice.id ? "Playing..." : "Click to preview"}
                      </p>
                    </div>
                    {voiceId === voice.id && (
                      <div className="ml-auto">
                        <Volume2 className="w-5 h-5 text-primary" />
                      </div>
                    )}
                  </div>
                </button>
              ))}
            </div>
          </div>

          <Separator />

          {/* Save button */}
          <div className="flex justify-end">
            <Button onClick={saveSettings} disabled={isSaving} className="glow">
              {isSaving ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  <Save className="w-4 h-4 mr-2" />
                  Save changes
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Account info */}
      <Card className="glass mt-6">
        <CardHeader>
          <CardTitle className="text-lg">Account</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Account management and billing options will be available soon.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
