"use client";

import React, { useRef, useEffect, useCallback } from "react";
import { Send, Paperclip, X } from "lucide-react";

export interface PendingUpload {
  filename: string;
  path: string;
}

export interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onUpload: (upload: PendingUpload) => void;
  onRemoveUpload: () => void;
  pendingUpload: PendingUpload | null;
  isLoading: boolean;
  uploadEndpoint: string;
  onUploadError?: (message: string) => void;
  placeholder?: string;
}

/**
 * ChatInput — focused, self-contained textarea + upload component.
 *
 * - Auto-expands up to ~6 lines then scrolls
 * - Enter = send, Shift+Enter = newline
 * - Disabled during streaming / loading
 * - Paperclip triggers hidden file input; accepted formats: .csv
 * - Pending upload shown as dismissible chip above the input row
 */
export function ChatInput({
  value,
  onChange,
  onSend,
  onUpload,
  onRemoveUpload,
  pendingUpload,
  isLoading,
  uploadEndpoint,
  onUploadError,
  placeholder = "Ask about market structure, upload CSV, or type your analysis…",
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auto-resize the textarea to fit content (max ~6 lines ≈ 144 px)
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 144)}px`;
  }, [value]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (!isLoading && (value.trim() || pendingUpload)) onSend();
      }
    },
    [isLoading, onSend, value, pendingUpload]
  );

  const handleFileChange = useCallback(
    async (ev: React.ChangeEvent<HTMLInputElement>) => {
      const file = ev.target.files?.[0];
      if (!file) return;

      try {
        const form = new FormData();
        form.append("file", file);

        const resp = await fetch(uploadEndpoint, {
          method: "POST",
          body: form,
        });

        if (!resp.ok) {
          const txt = await resp.text();
          throw new Error(txt || `Upload failed (${resp.status})`);
        }

        const data = (await resp.json()) as { path: string; filename?: string };
        onUpload({ filename: data.filename ?? file.name, path: data.path });
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        onUploadError?.(`Upload failed: ${msg}`);
      } finally {
        // Reset so the same file can be re-selected
        if (ev.target) ev.target.value = "";
      }
    },
    [uploadEndpoint, onUpload, onUploadError]
  );

  const canSend = !isLoading && (value.trim().length > 0 || !!pendingUpload);

  return (
    <div className="space-y-2">
      {/* Pending upload chip */}
      {pendingUpload && (
        <div className="flex items-center gap-2">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 bg-muted rounded-full text-sm text-foreground">
            <Paperclip size={12} className="text-muted-foreground" />
            {pendingUpload.filename}
            <button
              type="button"
              aria-label="Remove upload"
              onClick={onRemoveUpload}
              className="ml-1 text-muted-foreground hover:text-destructive transition-colors"
            >
              <X size={12} />
            </button>
          </div>
        </div>
      )}

      {/* Input row */}
      <div className="flex gap-2 items-end">
        {/* Hidden file picker */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,text/csv"
          className="hidden"
          onChange={handleFileChange}
        />

        {/* Textarea */}
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={isLoading}
          rows={1}
          className={[
            "flex-1 bg-card text-foreground rounded-lg px-4 py-3",
            "resize-none overflow-y-auto focus:outline-none focus:ring-2 focus:ring-primary",
            "border border-border transition-colors",
            "placeholder:text-muted-foreground",
            isLoading ? "opacity-50 cursor-not-allowed" : "",
          ]
            .filter(Boolean)
            .join(" ")}
        />

        {/* Paperclip upload button */}
        <button
          type="button"
          aria-label="Upload CSV"
          disabled={isLoading}
          onClick={() => fileInputRef.current?.click()}
          className={[
            "p-3 rounded-lg transition-colors",
            isLoading
              ? "opacity-50 cursor-not-allowed"
              : "hover:bg-muted text-muted-foreground hover:text-foreground",
          ].join(" ")}
        >
          <Paperclip size={20} />
        </button>

        {/* Send button */}
        <button
          type="button"
          aria-label="Send message"
          disabled={!canSend}
          onClick={onSend}
          className="px-4 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
        >
          <Send size={20} className={isLoading ? "opacity-30" : ""} />
        </button>
      </div>
    </div>
  );
}

export default ChatInput;
