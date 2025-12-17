import type { LanguageId, VoiceId } from "@/lib/config";

export interface Profile {
  id: string;
  email: string;
  voice_id: VoiceId;
  language: LanguageId;
  created_at: string;
  updated_at: string;
}

export interface ProfileUpdate {
  voice_id?: VoiceId;
  language?: LanguageId;
}

