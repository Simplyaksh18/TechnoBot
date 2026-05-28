import type React from "react"
import type { Metadata } from "next"

import { Analytics } from "@vercel/analytics/next"
import { Providers } from "./providers"
import "./globals.css"

import { Geist_Mono, Josefin_Sans as V0_Font_Josefin_Sans, Geist_Mono as V0_Font_Geist_Mono, Bitter as V0_Font_Bitter } from 'next/font/google'

// Initialize fonts
const _josefinSans = V0_Font_Josefin_Sans({ subsets: ['latin'], weight: ["100","200","300","400","500","600","700"] })
const _geistMono = V0_Font_Geist_Mono({ subsets: ['latin'], weight: ["100","200","300","400","500","600","700","800","900"] })
const _bitter = V0_Font_Bitter({ subsets: ['latin'], weight: ["100","200","300","400","500","600","700","800","900"] })

export const metadata: Metadata = {
  title: "TechnoBot - Technical Analysis Mentor",
  description: "Professional technical analysis guidance without trading advice",
  generator: "v0.app",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`font-sans antialiased`}>
        <Providers>{children}</Providers>
        <Analytics />
      </body>
    </html>
  )
}
