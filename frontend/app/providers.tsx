"use client";

import type React from "react";

import { GoogleOAuthProvider } from "@react-oauth/google";
import { AuthProvider } from "@/lib/auth-context";
import { ThemeProvider } from "@/lib/theme-context";
import { ChatProvider } from "@/lib/ChatContext";

const GOOGLE_CLIENT_ID =
  "973982653746-vd06pr6bspt5i737pvu8phmm9nnh8q5j.apps.googleusercontent.com";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
      <ThemeProvider>
        <AuthProvider>
          <ChatProvider>{children}</ChatProvider>
        </AuthProvider>
      </ThemeProvider>
    </GoogleOAuthProvider>
  );
}
