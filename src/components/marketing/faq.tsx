"use client";

import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

const faqs = [
  {
    question: "What sign languages does Unmute support?",
    answer:
      "Currently, Unmute supports American Sign Language (ASL) and French Sign Language (LSF). We're actively working on adding more sign languages including BSL, Auslan, and others. Stay tuned for updates!",
  },
  {
    question: "How accurate is the sign language recognition?",
    answer:
      "Our AI achieves over 95% accuracy for common signs and phrases in controlled conditions. Accuracy can vary based on lighting, camera quality, and signing speed. We continuously improve our models with user feedback.",
  },
  {
    question: "Is my video data stored or shared?",
    answer:
      "No. Your privacy is paramount. Video processing happens in real-time and no video data is stored on our servers. Only the generated text is transmitted securely for voice synthesis.",
  },
  {
    question: "Can I use Unmute with Zoom, Google Meet, or Teams?",
    answer:
      "Yes! Unmute works as a virtual audio device that integrates with any video conferencing platform. Simply select Unmute as your microphone input in your meeting app settings.",
  },
  {
    question: "What equipment do I need?",
    answer:
      "All you need is a computer with a webcam and a stable internet connection. For best results, we recommend good lighting and positioning your camera at chest height to capture your signs clearly.",
  },
  {
    question: "Is there a mobile app?",
    answer:
      "We're currently focused on the web and desktop experience for the best quality. Mobile apps for iOS and Android are on our roadmap for 2025.",
  },
  {
    question: "Can I customize my voice?",
    answer:
      "Yes! Pro and Team plans include access to all 5 AI voices, plus the ability to adjust speaking speed and tone. We're also working on custom voice cloning for enterprise customers.",
  },
  {
    question: "What's the latency like?",
    answer:
      "Unmute typically achieves sub-200ms latency from sign to speech, making conversations feel natural and fluid. Actual latency may vary based on your internet connection.",
  },
];

export function FAQ() {
  const [openIndex, setOpenIndex] = useState<number | null>(0);

  return (
    <section id="faq" className="py-24 px-4 sm:px-6 lg:px-8">
      <div className="max-w-3xl mx-auto">
        {/* Section header */}
        <div className="text-center mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold mb-4">
            Frequently asked <span className="gradient-text">questions</span>
          </h2>
          <p className="text-lg text-muted-foreground">
            Everything you need to know about Unmute.
          </p>
        </div>

        {/* FAQ list */}
        <div className="space-y-4">
          {faqs.map((faq, index) => (
            <div
              key={index}
              className="glass rounded-xl overflow-hidden"
            >
              <button
                className="w-full px-6 py-4 flex items-center justify-between text-left"
                onClick={() => setOpenIndex(openIndex === index ? null : index)}
              >
                <span className="font-medium pr-4">{faq.question}</span>
                <ChevronDown
                  className={cn(
                    "w-5 h-5 text-muted-foreground shrink-0 transition-transform duration-200",
                    openIndex === index && "rotate-180"
                  )}
                />
              </button>
              <div
                className={cn(
                  "overflow-hidden transition-all duration-200",
                  openIndex === index ? "max-h-96" : "max-h-0"
                )}
              >
                <p className="px-6 pb-4 text-muted-foreground">{faq.answer}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Contact CTA */}
        <div className="mt-12 text-center">
          <p className="text-muted-foreground">
            Still have questions?{" "}
            <a href="mailto:support@unmute.app" className="text-primary hover:underline">
              Contact our support team
            </a>
          </p>
        </div>
      </div>
    </section>
  );
}

