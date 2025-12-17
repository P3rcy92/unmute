import { Button } from "@/components/ui/button";
import Link from "next/link";
import { ArrowRight } from "lucide-react";

export function CTA() {
  return (
    <section className="py-24 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <div className="glass rounded-3xl p-12 text-center relative overflow-hidden">
          {/* Background gradient */}
          <div className="absolute inset-0 -z-10">
            <div className="absolute top-0 left-1/4 w-64 h-64 bg-primary/30 rounded-full blur-3xl" />
            <div className="absolute bottom-0 right-1/4 w-48 h-48 bg-accent/30 rounded-full blur-3xl" />
          </div>

          <h2 className="text-3xl sm:text-4xl font-bold mb-4">
            Ready to <span className="gradient-text">unmute</span> yourself?
          </h2>
          <p className="text-lg text-muted-foreground max-w-xl mx-auto mb-8">
            Join thousands of users who have already transformed their
            communication. Start for free today.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Button size="lg" asChild className="glow text-base px-8">
              <Link href="/signup">
                Get started free
                <ArrowRight className="ml-2 w-4 h-4" />
              </Link>
            </Button>
            <Button size="lg" variant="outline" className="glass text-base px-8" asChild>
              <Link href="#pricing">View pricing</Link>
            </Button>
          </div>
          <p className="mt-6 text-sm text-muted-foreground">
            No credit card required • Free forever plan available
          </p>
        </div>
      </div>
    </section>
  );
}

