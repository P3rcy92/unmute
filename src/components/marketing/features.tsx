import { Camera, Mic, Zap, Globe, Shield, Smartphone } from "lucide-react";

const features = [
  {
    icon: Camera,
    title: "Sign Recognition",
    description:
      "Advanced AI recognizes sign language in real-time through your camera with high accuracy.",
  },
  {
    icon: Mic,
    title: "Natural Voice Output",
    description:
      "Choose from 5 distinct AI voices to represent you naturally in any conversation.",
  },
  {
    icon: Zap,
    title: "Instant Translation",
    description:
      "Sub-200ms latency ensures your signs are spoken almost instantly, keeping conversations flowing.",
  },
  {
    icon: Globe,
    title: "Multi-language Support",
    description:
      "Support for English and French, with more languages coming soon.",
  },
  {
    icon: Shield,
    title: "Privacy First",
    description:
      "Your video never leaves your device. All processing happens locally with encrypted connections.",
  },
  {
    icon: Smartphone,
    title: "Works Everywhere",
    description:
      "Use Unmute in any video call or meeting platform - Zoom, Meet, Teams, and more.",
  },
];

export function Features() {
  return (
    <section id="features" className="py-24 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Section header */}
        <div className="text-center max-w-3xl mx-auto mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold mb-4">
            Everything you need to{" "}
            <span className="gradient-text">communicate freely</span>
          </h2>
          <p className="text-lg text-muted-foreground">
            Powerful features designed to make sign language communication
            seamless and natural.
          </p>
        </div>

        {/* Features grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <div
              key={feature.title}
              className="glass rounded-2xl p-6 hover:bg-card/80 transition-all duration-300 group"
              style={{
                animationDelay: `${index * 100}ms`,
              }}
            >
              <div className="w-12 h-12 rounded-xl bg-primary/20 flex items-center justify-center mb-4 group-hover:bg-primary/30 transition-colors">
                <feature.icon className="w-6 h-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
              <p className="text-muted-foreground">{feature.description}</p>
            </div>
          ))}
        </div>

        {/* How it works */}
        <div className="mt-24">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">
              How it <span className="gradient-text">works</span>
            </h2>
            <p className="text-lg text-muted-foreground">
              Three simple steps to start communicating.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                step: "01",
                title: "Enable Camera",
                description:
                  "Allow Unmute to access your camera. Your video stays private.",
              },
              {
                step: "02",
                title: "Start Signing",
                description:
                  "Sign naturally. Our AI recognizes your gestures in real-time.",
              },
              {
                step: "03",
                title: "Speak Out",
                description:
                  "Your chosen voice speaks your words. Join any call confidently.",
              },
            ].map((item, index) => (
              <div key={item.step} className="relative">
                {/* Connector line */}
                {index < 2 && (
                  <div className="hidden md:block absolute top-12 left-full w-full h-px bg-gradient-to-r from-primary/50 to-transparent -translate-x-1/2" />
                )}
                
                <div className="text-center">
                  <div className="w-24 h-24 rounded-full glass flex items-center justify-center mx-auto mb-6 gradient-border">
                    <span className="text-3xl font-bold gradient-text">
                      {item.step}
                    </span>
                  </div>
                  <h3 className="text-xl font-semibold mb-2">{item.title}</h3>
                  <p className="text-muted-foreground">{item.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

