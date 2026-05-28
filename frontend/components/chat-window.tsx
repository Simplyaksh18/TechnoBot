"use client";

import type React from "react";
import { useState, useRef, useEffect } from "react";
import ChatMessageComponent from "./chat-message";
import { DataTable } from "./data-table";
import { WelcomeBanner } from "./welcome-banner";
import { Send, Paperclip } from "lucide-react";
import { Footer } from "./footer";
import { useChat } from "@/lib/ChatContext";
import { API_BASE } from "@/lib/api";

export function ChatWindow({
  activeChat,
  onUpdateChatTitle,
  onNewChat,
}: {
  activeChat: string | null;
  onUpdateChatTitle: (chatId: string, query: string) => void;
  onNewChat: () => void;
}) {
  const { messages, sendMessage, isLoading, addMessage, sendHiddenMessage } =
    useChat();
  const [input, setInput] = useState("");
  const [pendingUpload, setPendingUpload] = useState<{
    filename: string;
    path: string;
  } | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() && !pendingUpload) return;

    try {
      if (pendingUpload) {
        // Send server-side file path to backend without showing it in chat
        await sendHiddenMessage(pendingUpload.path);
        // small pause to ensure backend updated context
        await new Promise((res) => setTimeout(res, 150));
        await sendMessage({ role: "user", content: input });
        setPendingUpload(null);
        setInput("");
        return;
      }

      await sendMessage({ role: "user", content: input });
      setInput("");
    } catch (err) {
      console.error(err);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const isEmpty = messages.length === 0;

  return (
    <div className="flex-1 flex flex-col h-full bg-background">
      {/* scrollable area: add bottom padding so footer / last messages aren't covered */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-6 pb-40">
        {isEmpty ? (
          <WelcomeBanner onNewChat={onNewChat} />
        ) : (
          <div className="space-y-4">
            {messages.map((msg, idx) => (
              <ChatMessageComponent
                key={idx}
                role={msg.role}
                content={msg.content}
              />
            ))}
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      {/* end of messages scroll area */}
      {/* Footer — fixed at bottom of viewport; contains input so both are visible */}
      <div className="fixed left-0 md:left-64 right-0 bottom-0 z-10">
        <div className="border-t border-border bg-background py-6">
          <div className="max-w-7xl mx-auto px-6">
            <div className="max-w-2xl mx-auto w-full">
              <div className="pt-2">
                <div className="border-t border-border bg-background pt-4">
                  <div className="flex gap-3 items-end">
                    <textarea
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={handleKeyDown}
                      placeholder="Ask about market structure, upload CSV, or type your analysis..."
                      disabled={isLoading}
                      className={`flex-1 bg-card text-foreground rounded-lg px-4 py-3 resize-none focus:outline-none focus:ring-2 focus:ring-primary border border-border max-h-24 ${
                        isLoading ? "opacity-50 cursor-not-allowed" : ""
                      }`}
                      rows={1}
                    />
                    <button
                      onClick={() => {
                        // open hidden file picker
                        fileInputRef.current?.click();
                      }}
                      disabled={isLoading}
                      className={`p-3 rounded-lg transition-colors ${
                        isLoading
                          ? "opacity-50 cursor-not-allowed"
                          : "hover:bg-muted"
                      }`}
                      aria-label="Upload CSV"
                    >
                      <Paperclip className="w-5 h-5 text-muted-foreground" />
                    </button>

                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".csv,text/csv"
                      style={{ display: "none" }}
                      onChange={async (ev) => {
                        const f = ev.target.files?.[0];
                        if (!f) return;
                        try {
                          const form = new FormData();
                          form.append("file", f);

                          const resp = await fetch(`${API_BASE}/upload`, {
                            method: "POST",
                            body: form,
                          });

                          if (!resp.ok) {
                            const txt = await resp.text();
                            throw new Error(txt || "Upload failed");
                          }

                          const data = await resp.json();
                          const serverPath = data.path;
                          const filename = data.filename || f.name;

                          // store pending upload; do NOT post preview or send to backend yet
                          setPendingUpload({ filename, path: serverPath });
                        } catch (err) {
                          console.error("Upload failed", err);
                          addMessage({
                            role: "assistant",
                            content: `Upload failed: ${err instanceof Error ? err.message : String(err)}`,
                            timestamp: Date.now(),
                          });
                        } finally {
                          if (ev.target)
                            (ev.target as HTMLInputElement).value = "";
                        }
                      }}
                    />

                    {isLoading && (
                      <div className="px-3 py-2 bg-muted rounded-lg text-sm text-primary animate-pulse">
                        🤖 TechnoBot is Thinking 🤔
                      </div>
                    )}

                    <button
                      onClick={handleSend}
                      disabled={isLoading || (!input.trim() && !pendingUpload)}
                      className="px-4 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
                      aria-label="Send message"
                    >
                      {!isLoading ? (
                        <Send className="w-5 h-5" />
                      ) : (
                        <Send className="w-5 h-5 opacity-30" />
                      )}
                    </button>
                  </div>
                  {pendingUpload && (
                    <div className="mt-3 flex items-center gap-2">
                      <div className="px-3 py-1 bg-muted rounded-full text-sm">
                        {pendingUpload.filename}
                      </div>
                      <button
                        className="text-sm text-destructive underline"
                        onClick={() => setPendingUpload(null)}
                      >
                        Remove
                      </button>
                    </div>
                  )}
                </div>
              </div>

              <div className="pt-4 text-center">
                <p className="text-xs text-muted-foreground">
                  TechnoBot is a TA mentor. Always conduct your own research
                  before making investment decisions.
                </p>
                <p className="text-xs text-muted-foreground mt-2 whitespace-nowrap">
                  ©️2026 Developed by Akshi👩‍💻
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
