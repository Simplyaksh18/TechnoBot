"use client";

import { useState } from "react";
import { Plus, MessageSquare, Trash2 } from "lucide-react";

interface Chat {
  id: string;
  title: string;
  timestamp: string;
}

interface SidebarProps {
  chats: Chat[];
  activeChat: string | null;
  onNewChat: () => void;
  onSelectChat: (id: string) => void;
  onDeleteChat: (id: string) => void;
}

export function Sidebar({
  chats,
  activeChat,
  onNewChat,
  onSelectChat,
  onDeleteChat,
}: SidebarProps) {
  // Track which chat is pending confirmation so we can show a second-click guard
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);

  function handleDeleteClick(e: React.MouseEvent, chatId: string) {
    e.stopPropagation(); // don't accidentally switch to the chat being deleted

    if (pendingDelete === chatId) {
      // Second click — confirmed
      onDeleteChat(chatId);
      setPendingDelete(null);
    } else {
      // First click — arm the confirmation
      setPendingDelete(chatId);
      // Auto-reset after 3 s so an accidental hover doesn't leave it armed
      setTimeout(
        () => setPendingDelete((p) => (p === chatId ? null : p)),
        3000,
      );
    }
  }

  return (
    <div className="w-64 border-r border-border bg-sidebar flex flex-col">
      <div className="p-4 border-b border-border">
        <button
          onClick={onNewChat}
          className="w-full flex items-center justify-center gap-2 bg-primary text-primary-foreground rounded-lg py-2 hover:bg-primary/90 transition-colors"
        >
          <Plus className="w-4 h-4" />
          <span className="text-sm font-medium">New Chat</span>
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-2">
        {chats.length === 0 ? (
          <p className="text-sm text-muted-foreground text-center py-8">
            No chats yet
          </p>
        ) : (
          chats.map((chat) => {
            const isActive = activeChat === chat.id;
            const isArmed = pendingDelete === chat.id;

            return (
              <div
                key={chat.id}
                className={`group relative flex items-start gap-1 rounded-lg transition-colors ${
                  isActive
                    ? "bg-primary/20 border border-primary"
                    : "hover:bg-muted"
                }`}
              >
                {/* Chat select button — takes most of the row */}
                <button
                  onClick={() => {
                    setPendingDelete(null); // dismiss any pending delete
                    onSelectChat(chat.id);
                  }}
                  className={`flex-1 min-w-0 text-left p-3 pr-1 text-sm ${
                    isActive
                      ? "text-foreground"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <div className="flex items-start gap-2 min-w-0">
                    <MessageSquare className="w-4 h-4 shrink-0 mt-0.5" />
                    <div className="min-w-0 flex-1">
                      <p className="font-medium truncate block">{chat.title}</p>
                      <p className="text-xs mt-1 opacity-70">
                        {chat.timestamp}
                      </p>
                    </div>
                  </div>
                </button>

                {/* Delete button — visible on hover or when armed */}
                <button
                  onClick={(e) => handleDeleteClick(e, chat.id)}
                  title={
                    isArmed ? "Click again to confirm delete" : "Delete chat"
                  }
                  className={`shrink-0 p-2 mt-1 mr-1 rounded transition-colors ${
                    isArmed
                      ? "text-destructive bg-destructive/10 opacity-100"
                      : "text-muted-foreground opacity-0 group-hover:opacity-100 hover:text-destructive hover:bg-destructive/10"
                  }`}
                  aria-label={isArmed ? "Confirm delete" : "Delete chat"}
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>

                {/* Confirmation tooltip */}
                {isArmed && (
                  <span className="absolute right-10 top-1/2 -translate-y-1/2 bg-destructive text-destructive-foreground text-[10px] font-medium px-2 py-0.5 rounded whitespace-nowrap z-10 pointer-events-none">
                    Click again to delete
                  </span>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
