"use client";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import Link from "next/link";
import { ArrowRight, Play, Sparkles } from "lucide-react";

export function Hero() {
  return (
    <section className="relative pt-32 pb-20 px-4 sm:px-6 lg:px-8 overflow-hidden">
      {/* Background effects */}
      <div className="absolute inset-0 -z-10 flex items-start justify-center pt-40">
        <div className="w-[500px] h-[500px] bg-primary/25 rounded-full blur-3xl animate-pulse" />
      </div>

      <div className="max-w-7xl mx-auto">
        <div className="text-center max-w-4xl mx-auto">
          {/* Badge */}
          <Badge
            variant="secondary"
            className="mb-6 px-4 py-2 text-sm glass animate-in fade-in slide-in-from-bottom-4 duration-700"
          >
            <Sparkles className="w-4 h-4 mr-2 text-primary" />
            Now in public beta
          </Badge>

          {/* Main headline */}
          <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight mb-6 animate-in fade-in slide-in-from-bottom-4 duration-700 delay-100">
            <span className="gradient-text">Unmute</span> sign language.
            <br />
            <span className="text-foreground">Instantly.</span>
          </h1>

          {/* Subtitle */}
          <p className="text-lg sm:text-xl text-muted-foreground max-w-2xl mx-auto mb-10 animate-in fade-in slide-in-from-bottom-4 duration-700 delay-200">
            Turn signing into natural voice for calls and meetings.
            Break communication barriers with real-time AI translation.
          </p>

          {/* CTAs */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 animate-in fade-in slide-in-from-bottom-4 duration-700 delay-300">
            <Button size="lg" asChild className="glow text-base px-8">
              <Link href="/signup">
                Start for free
                <ArrowRight className="ml-2 w-4 h-4" />
              </Link>
            </Button>
            <Button size="lg" variant="outline" className="glass text-base px-8">
              <Play className="mr-2 w-4 h-4" />
              Watch demo
            </Button>
          </div>

          {/* Social proof */}
          <div className="mt-16 animate-in fade-in slide-in-from-bottom-4 duration-700 delay-500">
            <p className="text-sm text-muted-foreground mb-4">
              Trusted by accessibility advocates worldwide
            </p>
            <div className="flex items-center justify-center gap-8 opacity-60">
              {/* Placeholder logos */}
              {["Microsoft", "Google", "Zoom", "Slack"].map((company) => (
                <div
                  key={company}
                  className="text-muted-foreground font-semibold text-sm"
                >
                  {company}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* App preview */}
        <div className="mt-20 relative animate-in fade-in slide-in-from-bottom-8 duration-1000 delay-700">
          <div className="glass rounded-2xl p-2 max-w-5xl mx-auto glow">
            <div className="bg-muted rounded-xl aspect-video flex items-center justify-center">
              <div className="text-center">
                <div className="w-20 h-20 rounded-full bg-primary/20 flex items-center justify-center mx-auto mb-4">
                  <Play className="w-8 h-8 text-primary ml-1" />
                </div>
                <p className="text-muted-foreground">App preview</p>
              </div>
            </div>
          </div>
          
          {/* Floating elements */}
          <div className="absolute -top-4 -left-4 glass rounded-xl p-4 hidden lg:block animate-in fade-in slide-in-from-left-4 duration-700 delay-1000">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-green-500/20 flex items-center justify-center">
                <span className="text-green-400 text-lg">✓</span>
              </div>
              <div>
                <p className="font-medium text-sm">Real-time</p>
                <p className="text-xs text-muted-foreground">{"< 200ms latency"}</p>
              </div>
            </div>
          </div>
          
          <div className="absolute -bottom-4 -right-4 glass rounded-xl p-4 hidden lg:block animate-in fade-in slide-in-from-right-4 duration-700 delay-1000">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center">
                <span className="text-primary text-lg">🎙</span>
              </div>
              <div>
                <p className="font-medium text-sm">5 Voices</p>
                <p className="text-xs text-muted-foreground">Natural AI voices</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

