import Link from "next/link";
import { APP_CONFIG } from "@/lib/config";

export default function AuthLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen flex flex-col">
      {/* Simple header */}
      <header className="p-4">
        <Link href="/" className="flex items-center gap-2 w-fit">
          <img src="/logounmute.png" alt="Unmute" className="h-8 w-auto" />
          <span className="font-semibold text-lg">{APP_CONFIG.name}</span>
        </Link>
      </header>

      {/* Auth content */}
      <main className="flex-1 flex items-center justify-center p-4">
        {children}
      </main>

      {/* Simple footer */}
      <footer className="p-4 text-center text-sm text-muted-foreground">
        © {new Date().getFullYear()} {APP_CONFIG.name}. All rights reserved.
      </footer>
    </div>
  );
}

