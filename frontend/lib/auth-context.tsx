"use client"

import type React from "react"
import { createContext, useContext, useState, useEffect } from "react"

export interface User {
  id: string
  username: string
  email: string
  emoji: string
  /** Profile photo URL — populated for Google-authenticated users */
  picture?: string
}

/** Decoded fields from a Google ID token that we use to build a User. */
export interface GoogleUserPayload {
  sub: string
  name: string
  email: string
  picture: string
}

interface AuthContextType {
  user: User | null
  isLoading: boolean
  login: (email: string, password: string) => Promise<void>
  register: (username: string, email: string, password: string, emoji: string) => Promise<void>
  loginWithGoogle: (decoded: GoogleUserPayload) => void
  logout: () => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  // Initialize from localStorage
  useEffect(() => {
    const stored = localStorage.getItem("user")
    if (stored) {
      try {
        setUser(JSON.parse(stored))
      } catch {
        localStorage.removeItem("user")
      }
    }
    setIsLoading(false)
  }, [])

  const login = async (email: string, password: string) => {
    setIsLoading(true)
    try {
      // Demo credentials
      const demoUsers = {
        "demo@technobot.com": { password: "demo123", username: "Demo Trader", emoji: "📊" },
        "trader@technobot.com": { password: "trader123", username: "Chart Analyst", emoji: "📈" },
      }

      const demoUser = demoUsers[email as keyof typeof demoUsers]
      if (demoUser && demoUser.password === password) {
        const newUser: User = {
          id: Math.random().toString(36).substring(7),
          username: demoUser.username,
          email,
          emoji: demoUser.emoji,
        }
        setUser(newUser)
        localStorage.setItem("user", JSON.stringify(newUser))
      } else {
        throw new Error("Invalid credentials")
      }
    } finally {
      setIsLoading(false)
    }
  }

  const register = async (username: string, email: string, password: string, emoji: string) => {
    setIsLoading(true)
    try {
      const newUser: User = {
        id: Math.random().toString(36).substring(7),
        username,
        email,
        emoji,
      }
      setUser(newUser)
      localStorage.setItem("user", JSON.stringify(newUser))
    } finally {
      setIsLoading(false)
    }
  }

  /**
   * Called after a successful Google One-Tap / popup sign-in.
   * Decodes the Google ID token fields and stores a User compatible with the
   * existing email/password session shape — no backend call required.
   */
  const loginWithGoogle = (decoded: GoogleUserPayload) => {
    const newUser: User = {
      id:       decoded.sub,
      username: decoded.name,
      email:    decoded.email,
      emoji:    "🌐",
      picture:  decoded.picture,
    }
    console.log("GOOGLE AUTH:", { name: newUser.username, email: newUser.email })
    setUser(newUser)
    localStorage.setItem("user", JSON.stringify(newUser))
  }

  const logout = () => {
    setUser(null)
    localStorage.removeItem("user")
  }

  return (
    <AuthContext.Provider value={{ user, isLoading, login, register, loginWithGoogle, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error("useAuth must be used within AuthProvider")
  }
  return context
}
