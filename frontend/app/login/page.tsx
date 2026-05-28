"use client";

import type React from "react";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";
import { GoogleLogin } from "@react-oauth/google";
import { useAuth } from "@/lib/auth-context";
import type { GoogleUserPayload } from "@/lib/auth-context";
import Link from "next/link";

export default function LoginPage() {
  const router = useRouter();
  const { login, loginWithGoogle, isLoading } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    try {
      await login(email, password);
      router.push("/chat");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    }
  };

  /**
   * Handles a successful Google popup credential response.
   * Decodes the JWT payload (public, not sensitive), extracts profile
   * fields, persists to localStorage via loginWithGoogle, then navigates.
   */
  const handleGoogleSuccess = (credentialResponse: { credential?: string }) => {
    try {
      const token = credentialResponse.credential;
      if (!token) throw new Error("No credential returned");

      // Decode the public JWT payload — no verification needed client-side
      const decoded = JSON.parse(
        atob(token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/"))
      ) as GoogleUserPayload;

      console.log("GOOGLE TOKEN:", token.slice(0, 20) + "…");
      console.log("USER:", { name: decoded.name, email: decoded.email });

      loginWithGoogle(decoded);
      router.push("/chat");
    } catch (err) {
      console.error("[GoogleLogin] decode error:", err);
      setError("Google sign-in failed — please try again.");
    }
  };

  return (
    <div className="relative min-h-screen overflow-hidden bg-slate-950 flex items-center justify-center px-4 py-8">
      <div className="absolute inset-0">
        <Image
          src="/data-analyst.jpg"
          alt="TechnoBot sign in background"
          fill
          priority
          className="object-cover object-center opacity-80"
        />
        <div className="absolute inset-0 bg-linear-to-br from-slate-950/65 via-slate-950/35 to-slate-900/50" />
      </div>

      <div className="relative w-full max-w-md">
        <div className="rounded-2xl border border-white/10 bg-slate-950/45 backdrop-blur-md shadow-2xl shadow-black/35 p-8 text-white">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-white mb-2">🤖TechnoBot</h1>
            <p className="text-cyan-300">Your Technical Analysis Mentor!</p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2 text-white/85">
                Email
              </label>
              <input
                suppressHydrationWarning
                name="email"
                autoComplete="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full rounded-lg px-4 py-2 border border-white/10 bg-white/10 text-white placeholder:text-white/40 focus:outline-none focus:ring-2 focus:ring-cyan-400"
                placeholder="demo@technobot.com"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2 text-white/85">
                Password
              </label>
              <input
                suppressHydrationWarning
                name="current-password"
                autoComplete="current-password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full rounded-lg px-4 py-2 border border-white/10 bg-white/10 text-white placeholder:text-white/40 focus:outline-none focus:ring-2 focus:ring-cyan-400"
                placeholder="demo123"
                required
              />
            </div>

            {error && (
              <div className="bg-red-500/10 border border-red-400/30 rounded-lg p-3 text-sm text-red-200">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={isLoading}
              className="w-full rounded-lg py-2 font-medium text-slate-950 bg-cyan-300 hover:bg-cyan-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isLoading ? "Signing in..." : "Sign In"}
            </button>
          </form>

          {/* ── Google Sign-In ─────────────────────────────────────────────── */}
          <div className="mt-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="flex-1 h-px bg-white/15" />
              <span className="text-xs text-white/45 font-medium tracking-wide uppercase">
                or continue with
              </span>
              <div className="flex-1 h-px bg-white/15" />
            </div>

            <div className="flex justify-center">
              <GoogleLogin
                onSuccess={handleGoogleSuccess}
                onError={() => {
                  console.log("Google Login Failed");
                  setError("Google sign-in failed — please try again.");
                }}
                theme="filled_black"
                size="large"
                text="signin_with"
                shape="rectangular"
                width="368"
              />
            </div>
          </div>

          {/* ── Create account link ─────────────────────────────────────────── */}
          <div className="mt-6 pt-6 border-t border-white/10">
            <p className="text-sm text-white/65 text-center mb-4">
              Don't have an account?
            </p>
            <Link
              href="/register"
              className="block text-center text-cyan-300 hover:underline font-medium"
            >
              Create Account
            </Link>
          </div>

          <div className="mt-8 pt-6 border-t border-white/10">
            <p className="text-xs font-semibold text-white/60 mb-3">
              DEMO CREDENTIALS:
            </p>
            <div className="space-y-3 text-xs">
              <div>
                <p className="font-medium text-white">
                  Account 1: Demo Trader 📊
                </p>
                <p className="text-white/60">demo@technobot.com</p>
                <p className="text-white/60">demo123</p>
              </div>
              <div>
                <p className="font-medium text-white">
                  Account 2: Chart Analyst 📈
                </p>
                <p className="text-white/60">trader@technobot.com</p>
                <p className="text-white/60">trader123</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
