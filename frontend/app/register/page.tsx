"use client"

import type React from "react"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { GoogleLogin } from "@react-oauth/google"
import { useAuth } from "@/lib/auth-context"
import type { GoogleUserPayload } from "@/lib/auth-context"
import Link from "next/link"

const EMOJI_OPTIONS = ["📊", "📈", "📉", "💹", "🎯", "🔍", "⚡", "💡"]

export default function RegisterPage() {
  const router = useRouter()
  const { register, loginWithGoogle, isLoading } = useAuth()
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
    confirmPassword: "",
    emoji: "📊",
  })
  const [error, setError] = useState("")

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target
    setFormData((prev) => ({ ...prev, [name]: value }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")

    if (formData.password !== formData.confirmPassword) {
      setError("Passwords do not match")
      return
    }

    if (formData.password.length < 6) {
      setError("Password must be at least 6 characters")
      return
    }

    try {
      await register(formData.username, formData.email, formData.password, formData.emoji)
      router.push("/chat")
    } catch (err) {
      setError(err instanceof Error ? err.message : "Registration failed")
    }
  }

  /**
   * Handles a successful Google popup credential during sign-up.
   * Uses the same loginWithGoogle path as the login page — Google accounts
   * don't require a separate registration step.
   */
  const handleGoogleSuccess = (credentialResponse: { credential?: string }) => {
    try {
      const token = credentialResponse.credential
      if (!token) throw new Error("No credential returned")

      const decoded = JSON.parse(
        atob(token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/"))
      ) as GoogleUserPayload

      console.log("GOOGLE TOKEN:", token.slice(0, 20) + "…")
      console.log("USER:", { name: decoded.name, email: decoded.email })

      loginWithGoogle(decoded)
      router.push("/chat")
    } catch (err) {
      console.error("[GoogleSignup] decode error:", err)
      setError("Google sign-up failed — please try again.")
    }
  }

  return (
    <div className="min-h-screen bg-background flex items-center justify-center px-4">
      <div className="w-full max-w-md">
        <div className="bg-card rounded-lg border border-border p-8">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-primary mb-2">TechnoBot</h1>
            <p className="text-muted-foreground">Create Your Account</p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Username</label>
              <input
                type="text"
                name="username"
                value={formData.username}
                onChange={handleChange}
                className="w-full bg-input text-foreground rounded-lg px-4 py-2 border border-border focus:outline-none focus:ring-2 focus:ring-primary"
                placeholder="Your trading alias"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Email</label>
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                className="w-full bg-input text-foreground rounded-lg px-4 py-2 border border-border focus:outline-none focus:ring-2 focus:ring-primary"
                placeholder="your@email.com"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Select Emoji</label>
              <div className="grid grid-cols-4 gap-2">
                {EMOJI_OPTIONS.map((emoji) => (
                  <button
                    key={emoji}
                    type="button"
                    onClick={() => setFormData((prev) => ({ ...prev, emoji }))}
                    className={`p-3 rounded-lg text-2xl border transition-colors ${
                      formData.emoji === emoji ? "bg-primary/20 border-primary" : "border-border hover:bg-muted"
                    }`}
                  >
                    {emoji}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Password</label>
              <input
                type="password"
                name="password"
                value={formData.password}
                onChange={handleChange}
                className="w-full bg-input text-foreground rounded-lg px-4 py-2 border border-border focus:outline-none focus:ring-2 focus:ring-primary"
                placeholder="Min 6 characters"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Confirm Password</label>
              <input
                type="password"
                name="confirmPassword"
                value={formData.confirmPassword}
                onChange={handleChange}
                className="w-full bg-input text-foreground rounded-lg px-4 py-2 border border-border focus:outline-none focus:ring-2 focus:ring-primary"
                placeholder="Confirm password"
                required
              />
            </div>

            {error && (
              <div className="bg-destructive/10 border border-destructive/30 rounded-lg p-3 text-sm text-destructive">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-primary text-primary-foreground rounded-lg py-2 font-medium hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isLoading ? "Creating Account..." : "Create Account"}
            </button>
          </form>

          {/* ── Google Sign-Up ─────────────────────────────────────────────── */}
          <div className="mt-6 pt-6 border-t border-border">
            <div className="flex items-center gap-3 mb-4">
              <div className="flex-1 h-px bg-border" />
              <span className="text-xs text-muted-foreground font-medium tracking-wide uppercase">
                or sign up with
              </span>
              <div className="flex-1 h-px bg-border" />
            </div>

            <div className="flex justify-center">
              <GoogleLogin
                onSuccess={handleGoogleSuccess}
                onError={() => {
                  console.log("Google Signup Failed")
                  setError("Google sign-up failed — please try again.")
                }}
                theme="outline"
                size="large"
                text="signup_with"
                shape="rectangular"
                width="368"
              />
            </div>
          </div>

          {/* ── Already have an account ─────────────────────────────────────── */}
          <div className="mt-6 pt-6 border-t border-border">
            <p className="text-sm text-muted-foreground text-center">Already have an account?</p>
            <Link href="/login" className="block text-center text-primary hover:underline font-medium mt-2">
              Sign In
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}
