"use client";

import Image from "next/image";
import { useAuth } from "@/lib/auth-context";
import { useTheme } from "@/lib/theme-context";
import { useRouter } from "next/navigation";
import { Moon, Sun, LogOut } from "lucide-react";

export function Navbar() {
  const { user, logout } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const router = useRouter();

  const handleLogout = () => {
    logout();
    router.replace("/login");
  };

  return (
    <nav className="border-b border-border bg-card px-6 py-4 flex items-center justify-between sticky top-0 z-50">
      <div className="flex items-center gap-3">
        <div className="text-3xl">🤖</div>
        <div className="text-2xl font-bold text-primary">TechnoBot</div>
      </div>

      <div className="flex items-center gap-6">
        <button
          onClick={toggleTheme}
          className="p-2 hover:bg-muted rounded-lg transition-colors"
          aria-label="Toggle theme"
        >
          {theme === "dark" ? (
            <Sun className="w-5 h-5 text-muted-foreground" />
          ) : (
            <Moon className="w-5 h-5 text-muted-foreground" />
          )}
        </button>

        {user && (
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-3">
              {/* Google profile photo (when available) or emoji avatar */}
              {user.picture ? (
                <Image
                  src={user.picture}
                  alt={user.username}
                  width={44}
                  height={44}
                  className="w-11 h-11 rounded-full object-cover ring-2 ring-primary/30"
                  referrerPolicy="no-referrer"
                />
              ) : (
                <div className="w-12 h-12 rounded-full bg-linear-to-br from-primary to-accent flex items-center justify-center text-xl font-semibold ring-2 ring-primary/30">
                  {user.emoji}
                </div>
              )}
              <span className="text-sm font-medium text-foreground">
                {user.username}
              </span>
            </div>
            <button
              onClick={handleLogout}
              type="button"
              className="p-2 hover:bg-muted rounded-lg transition-colors"
              aria-label="Logout"
            >
              <LogOut className="w-5 h-5 text-muted-foreground" />
            </button>
          </div>
        )}
      </div>
    </nav>
  );
}
