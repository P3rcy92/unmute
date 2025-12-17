export const APP_CONFIG = {
  name: "Unmute",
  description: "Transform sign language into natural voice for calls and meetings.",
  url: process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000",
} as const;

export const LANGUAGES = [
  { id: "en", name: "English", flag: "🇬🇧" },
  { id: "fr", name: "Français", flag: "🇫🇷" },
] as const;

export type LanguageId = (typeof LANGUAGES)[number]["id"];

export const VOICES = [
  { id: "pNInz6obpgDQGcFmaJgB", name: "Adam", audioFile: "/adam-pNInz6obpgDQGcFmaJgB.mp3" },
  { id: "Xb7hH8MSUJpSbSDYk0k2", name: "Alice", audioFile: "/alice-Xb7hH8MSUJpSbSDYk0k2.mp3" },
  { id: "pqHfZKP75CvOlQylNhV4", name: "Bill", audioFile: "/bill-pqHfZKP75CvOlQylNhV4.mp3" },
  { id: "cgSgspJ2msm6clMCkdW9", name: "Jessica", audioFile: "/jessica-cgSgspJ2msm6clMCkdW9.mp3" },
] as const;

export type VoiceId = (typeof VOICES)[number]["id"];

export const PRICING_PLANS = [
  {
    id: "free",
    name: "Free",
    price: 0,
    period: "forever",
    description: "Perfect for trying out Unmute",
    features: [
      "5 minutes per day",
      "1 voice option",
      "Basic support",
      "Web app access",
    ],
    cta: "Get Started",
    popular: false,
  },
  {
    id: "pro",
    name: "Pro",
    price: 20,
    period: "month",
    description: "For individuals who need more",
    features: [
      "Unlimited usage",
      "All 5 voice options",
      "Priority support",
      "Web & desktop app",
      "Custom voice tuning",
      "HD audio quality",
    ],
    cta: "Start Pro",
    popular: true,
  },
  {
    id: "team",
    name: "Team",
    price: 20,
    period: "user/month",
    description: "For teams and organizations",
    features: [
      "Everything in Pro",
      "Team dashboard",
      "Admin controls",
      "API access",
      "SSO integration",
      "Dedicated support",
    ],
    cta: "Contact Sales",
    popular: false,
  },
] as const;

export const WS_CONFIG = {
  url: process.env.NEXT_PUBLIC_WS_URL || "wss://placeholder.example/ws",
  reconnectAttempts: 5,
  reconnectDelay: 1000,
} as const;

