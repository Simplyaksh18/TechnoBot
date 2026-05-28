"use client";

import type React from "react";
import { useState, useRef, useEffect } from "react";
import ChatMessageComponent from "@/components/chat-message";
import { WelcomeBanner } from "@/components/welcome-banner";
import { ChatInput, type PendingUpload } from "@/components/chat-input";
import { useChat } from "@/lib/ChatContext";
import { API_BASE } from "@/lib/api";

// A1: activeChat and onUpdateChatTitle props removed — state now lives in ChatContext
export default function ChatWindow({ onNewChat }: { onNewChat: () => void }) {
  const {
    messages,
    sendMessage,
    sendUserMessage,
    isLoading,
    addMessage,
    sendHiddenMessage,
    streamingMessageId,
  } = useChat();

  const [input, setInput] = useState("");
  const [pendingUpload, setPendingUpload] = useState<PendingUpload | null>(
    null,
  );
  const bottomRef = useRef<HTMLDivElement | null>(null);

  // Scroll to bottom whenever messages change (including streaming token appends)
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ── Send logic ─────────────────────────────────────────────────────────────

  const handleSend = async () => {
    if (!input.trim() && !pendingUpload) return;

    const textToSend = input;

    try {
      if (pendingUpload) {
        // Hidden file-path message first (no user bubble), then the user's query
        await sendHiddenMessage(pendingUpload.path);
        // Small pause so backend context is ready before the follow-on query
        await new Promise((res) => setTimeout(res, 150));
        if (textToSend.trim()) {
          await sendMessage({ role: "user", content: textToSend });
        }
        setPendingUpload(null);
        setInput("");
        return;
      }

      await sendMessage({ role: "user", content: textToSend });
      setInput("");
    } catch (err) {
      console.error("[ChatWindow] send error:", err);
      addMessage({
        role: "assistant",
        content: `Upload or send failed: ${err instanceof Error ? err.message : String(err)}`,
        timestamp: Date.now(),
      });
    }
  };

  // ── Follow-up chip handler ─────────────────────────────────────────────────

  /**
   * Called when the user clicks a follow-up suggestion chip.
   * Sends the question directly as a user message — no intermediate state needed.
   */
  const handleFollowUp = (question: string) => {
    sendUserMessage(question).catch((err) =>
      console.error("[ChatWindow] follow-up error:", err),
    );
  };

  // ── Upload error ───────────────────────────────────────────────────────────

  const handleUploadError = (message: string) => {
    addMessage({ role: "assistant", content: message, timestamp: Date.now() });
  };

  const isEmpty = messages.length === 0;

  return (
    <div className="flex-1 flex flex-col h-full bg-background">
      {/* ── Scrollable message area ─────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto px-4 py-6 pb-52">
        {isEmpty ? (
          <WelcomeBanner onNewChat={onNewChat} />
        ) : (
          <div className="space-y-4 max-w-3xl mx-auto">
            {messages.map((msg) => (
              <ChatMessageComponent
                key={msg.id ?? `${msg.role}-${msg.timestamp}`}
                role={msg.role}
                content={msg.content}
                streaming={msg.id === streamingMessageId}
                onFollowUp={handleFollowUp}
                status={msg.status}
                evidence={msg.evidence} // Task 2: message-scoped structured dict
                insight={msg.insight} // Task 4: message-scoped structured dict
              />
            ))}
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* ── Fixed bottom input bar ──────────────────────────────────────────── */}
      <div className="fixed left-0 md:left-64 right-0 bottom-0 z-10">
        <div className="border-t border-border bg-background py-4">
          <div className="max-w-3xl mx-auto px-4 space-y-3">
            {/* Loading indicator */}
            {isLoading && (
              <div className="flex justify-center">
                <span className="px-3 py-1.5 rounded-full bg-muted text-sm text-primary animate-pulse">
                  🤖 TechnoBot is thinking…
                </span>
              </div>
            )}

            {/* Input component */}
            <ChatInput
              value={input}
              onChange={setInput}
              onSend={handleSend}
              onUpload={setPendingUpload}
              onRemoveUpload={() => setPendingUpload(null)}
              pendingUpload={pendingUpload}
              isLoading={isLoading}
              uploadEndpoint={`${API_BASE}/upload`}
              onUploadError={handleUploadError}
            />

            {/* Disclaimer */}
            <p className="text-center text-xs text-muted-foreground">
              TechnoBot is a TA mentor. Always conduct your own research before
              making investment decisions.
            </p>
            <p className="text-center text-xs text-muted-foreground">
              ©️2026 Developed by Akshi👩‍💻
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
